"""CLI to fetch, embed, and build LanceDB index."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import lancedb
import pyarrow as pa

from config import Settings, get_settings
from ingest.embed import build_embed_text, embed_texts_offline
from ingest.fetch import Checkpoint, fetch_patents
from search.models import PatentRecord

logger = logging.getLogger(__name__)

TABLE_NAME = "patents"


class CheckpointCompleteError(RuntimeError):
    """Raised when ingest checkpoint is complete and rebuild was not requested."""


def _schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("patent_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("abstract", pa.string()),
            pa.field("claim_snippet", pa.string()),
            pa.field("cpc_codes", pa.list_(pa.string())),
            pa.field("filing_date", pa.string()),
            pa.field("assignee", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


def _record_row(record: PatentRecord, text: str, vector: list[float]) -> dict:
    return {
        "patent_id": record.patent_id,
        "title": record.title,
        "abstract": record.abstract,
        "claim_snippet": record.claim_snippet,
        "cpc_codes": record.cpc_codes,
        "filing_date": record.filing_date_str(),
        "assignee": record.assignee or "",
        "text": text,
        "vector": vector,
    }


def _embedded_ids(db: lancedb.DBConnection, table_name: str, table_dir: Path) -> set[str]:
    if not table_dir.exists():
        return set()
    table = db.open_table(table_name)
    return set(table.to_pandas()["patent_id"].tolist())


def _append_batch(
    db: lancedb.DBConnection,
    table_name: str,
    table_dir: Path,
    records: list[PatentRecord],
    texts: list[str],
    vectors: list[list[float]],
) -> None:
    rows = [_record_row(records[i], texts[i], vectors[i]) for i in range(len(records))]
    vector_dim = len(vectors[0])
    if table_dir.exists():
        table = db.open_table(table_name)
        table.add(rows)
    else:
        db.create_table(
            table_name, data=rows, schema=_schema(vector_dim), mode="overwrite"
        )


def _checkpoint_records(checkpoint_path: Path) -> list[PatentRecord]:
    checkpoint = Checkpoint.load(checkpoint_path)
    return [PatentRecord.model_validate(raw) for raw in checkpoint.records]


def build_index(
    *,
    limit: int | None = None,
    output_path: Path | None = None,
    batch_size: int = 64,
    reset_checkpoint: bool = False,
    settings: Settings | None = None,
) -> Path:
    """Fetch patents incrementally, embed offline, append to LanceDB (resume-safe)."""
    settings = settings or get_settings()
    output_path = output_path or settings.lance_path

    if reset_checkpoint:
        if settings.checkpoint_path.exists():
            settings.checkpoint_path.unlink()

    checkpoint = Checkpoint.load(settings.checkpoint_path)
    if checkpoint.complete and not reset_checkpoint:
        raise CheckpointCompleteError(
            "Checkpoint marked complete; pass --reset-checkpoint to rebuild index."
        )

    db_dir = output_path.parent
    table_name = output_path.stem
    table_dir = db_dir / f"{table_name}.lance"
    if reset_checkpoint and table_dir.exists():
        shutil.rmtree(table_dir)

    db_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_dir))
    embedded = _embedded_ids(db, table_name, table_dir)

    def _at_limit() -> bool:
        return limit is not None and len(embedded) >= limit

    def _fetch_headroom() -> int | None:
        if limit is None:
            return None
        return max(0, limit - len(embedded))

    def flush(records: list[PatentRecord]) -> None:
        nonlocal embedded
        if not records:
            return
        if limit is not None:
            records = records[: max(0, limit - len(embedded))]
            if not records:
                return
        texts = [
            build_embed_text(r, max_tokens=settings.max_embed_tokens) for r in records
        ]
        logger.info("Embedding batch of %s patents", len(texts))
        vectors = embed_texts_offline(
            texts, settings.embed_model, batch_size=min(batch_size, len(texts))
        )
        _append_batch(db, table_name, table_dir, records, texts, vectors)
        embedded.update(r.patent_id for r in records)

    pending: list[PatentRecord] = []

    def queue(record: PatentRecord) -> bool:
        """Queue a record for embedding. Returns False when limit is reached."""
        if record.patent_id in embedded:
            return True
        if _at_limit():
            return False
        pending.append(record)
        if len(pending) >= batch_size:
            flush(pending)
            pending.clear()
        return not _at_limit()

    # Embed checkpointed records from prior failed runs (e.g. page 1 before crash).
    for record in _checkpoint_records(settings.checkpoint_path):
        if not queue(record):
            break
    flush(pending)
    pending.clear()

    new_count = 0
    fetch_headroom = _fetch_headroom()
    if fetch_headroom is None or fetch_headroom > 0:
        for record in fetch_patents(settings, limit=fetch_headroom):
            if not queue(record):
                break
            new_count += 1
    flush(pending)

    if not embedded:
        raise RuntimeError("No patents fetched — check API or checkpoint state.")

    logger.info(
        "Index at %s contains %s records (%s newly fetched this run)",
        table_dir,
        len(embedded),
        new_count,
    )
    return table_dir


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build PatentHub LanceDB index")
    parser.add_argument("--limit", type=int, default=None, help="Max patents to fetch")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output LanceDB directory (default: PATENTHUB_LANCE_PATH)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Clear ingest checkpoint before fetch",
    )
    args = parser.parse_args(argv)

    try:
        build_index(
            limit=args.limit,
            output_path=args.output,
            batch_size=args.batch_size,
            reset_checkpoint=args.reset_checkpoint,
        )
    except CheckpointCompleteError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.error("Build failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())