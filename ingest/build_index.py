"""CLI to fetch, embed, and build LanceDB index."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import lancedb
import pyarrow as pa

from config import get_settings
from ingest.embed import build_embed_text, embed_texts_offline
from ingest.fetch import fetch_patents

logger = logging.getLogger(__name__)

TABLE_NAME = "patents"


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


def build_index(
    *,
    limit: int | None = None,
    output_path: Path | None = None,
    batch_size: int = 64,
    reset_checkpoint: bool = False,
) -> Path:
    """Fetch patents, embed offline, write LanceDB table."""
    settings = get_settings()
    output_path = output_path or settings.lance_path

    if reset_checkpoint and settings.checkpoint_path.exists():
        settings.checkpoint_path.unlink()

    records = list(fetch_patents(settings, limit=limit))
    if not records:
        raise RuntimeError("No patents fetched — check API or checkpoint state.")

    texts = [
        build_embed_text(r, max_tokens=settings.max_embed_tokens) for r in records
    ]
    logger.info("Embedding %s patents with %s", len(texts), settings.embed_model)
    vectors = embed_texts_offline(texts, settings.embed_model, batch_size=batch_size)
    vector_dim = len(vectors[0])

    db_dir = output_path.parent
    table_name = output_path.stem
    table_dir = db_dir / f"{table_name}.lance"
    if table_dir.exists():
        shutil.rmtree(table_dir)

    db_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_dir))
    data = [
        {
            "patent_id": r.patent_id,
            "title": r.title,
            "abstract": r.abstract,
            "claim_snippet": r.claim_snippet,
            "cpc_codes": r.cpc_codes,
            "filing_date": r.filing_date,
            "assignee": r.assignee,
            "text": texts[i],
            "vector": vectors[i],
        }
        for i, r in enumerate(records)
    ]
    db.create_table(table_name, data=data, schema=_schema(vector_dim), mode="overwrite")
    logger.info("Index written to %s (%s records)", table_dir, len(records))
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
    except Exception as exc:
        logger.error("Build failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())