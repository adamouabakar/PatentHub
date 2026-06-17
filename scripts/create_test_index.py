"""Build a minimal LanceDB index for local testing (no API key required)."""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import shutil
import sys
from pathlib import Path

import lancedb
import pyarrow as pa

# Allow running as `python scripts/create_test_index.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import Settings, get_settings
from search.query import OFFLINE_INDEX_MARKER, _is_lance_table_dir, get_fastembed_model

logger = logging.getLogger(__name__)

DUMMY_PATENTS: list[dict[str, object]] = [
    {
        "patent_id": "10000001",
        "title": "Neural network for medical image diagnosis",
        "abstract": "A convolutional neural network classifies radiology scans.",
        "claim_snippet": "A method comprising training a deep neural network on labeled images.",
        "cpc_codes": ["G06N3/08"],
        "filing_date": "2018-03-15",
        "assignee": "HealthAI Corp",
    },
    {
        "patent_id": "10000002",
        "title": "Système d'apprentissage profond pour la traduction automatique",
        "abstract": "Un modèle transformer multilingue pour traduire du texte en temps réel.",
        "claim_snippet": "Procédé d'entraînement d'un réseau de neurones sur corpus parallèles.",
        "cpc_codes": ["G06N3/04"],
        "filing_date": "2019-07-22",
        "assignee": "LangueTech SA",
    },
    {
        "patent_id": "10000003",
        "title": "Reinforcement learning for robotic grasping",
        "abstract": "An agent learns grasp policies via simulated self-play.",
        "claim_snippet": "A robotic controller trained with policy gradient reinforcement learning.",
        "cpc_codes": ["G06N3/006"],
        "filing_date": "2020-11-01",
        "assignee": "RoboLearn Inc",
    },
    {
        "patent_id": "10000004",
        "title": "Détection d'anomalies par autoencodeur",
        "abstract": "Un autoencodeur détecte des défauts industriels sur chaîne de production.",
        "claim_snippet": "Un procédé de détection utilisant un réseau encodeur-décodeur.",
        "cpc_codes": ["G06N3/08"],
        "filing_date": "2021-05-10",
        "assignee": "IndustrieVision",
    },
    {
        "patent_id": "10000005",
        "title": "Federated learning on edge devices",
        "abstract": "Distributed model updates without centralizing private user data.",
        "claim_snippet": "A federation server aggregates gradient updates from edge clients.",
        "cpc_codes": ["G06N3/098"],
        "filing_date": "2022-01-30",
        "assignee": "EdgeML LLC",
    },
]


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


VECTOR_DIM = 384


def _pseudo_vector(text: str, dim: int = VECTOR_DIM) -> list[float]:
    """Deterministic fake embedding (no model download) for offline smoke tests."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def _embed_text(record: dict[str, object]) -> str:
    parts = [
        str(record.get("title") or "").strip(),
        str(record.get("abstract") or "").strip(),
        str(record.get("claim_snippet") or "").strip(),
    ]
    return "\n\n".join(p for p in parts if p)


def create_test_index(
    output_path: Path | None = None,
    *,
    force: bool = False,
    offline: bool = False,
    settings: Settings | None = None,
) -> Path:
    """
    Create a small searchable LanceDB index using fastembed (flat layout).

    Layout matches ingest/build_index.py: db at ``data/``, table at ``data/patents.lance/``.
    """
    settings = settings or get_settings()
    output_path = (output_path or settings.lance_path).resolve()
    db_dir = output_path.parent
    table_name = output_path.stem
    flat_table_dir = db_dir / f"{table_name}.lance"

    if output_path.exists():
        if force:
            logger.info("Removing existing index at %s", output_path)
            shutil.rmtree(output_path)
        elif not _is_lance_table_dir(flat_table_dir):
            raise RuntimeError(
                f"Index exists at {output_path} but not in flat layout "
                f"(nested or corrupt). Pass --force to replace it."
            )

    if _is_lance_table_dir(flat_table_dir) and not force:
        logger.info(
            "Flat index already exists at %s — skipping (use --force to replace)",
            flat_table_dir,
        )
        return flat_table_dir

    db_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_dir))

    texts = [_embed_text(p) for p in DUMMY_PATENTS]
    if offline:
        logger.warning(
            "OFFLINE MODE: pseudo-embeddings only. Streamlit search uses real "
            "fastembed vectors — results will NOT be semantically meaningful."
        )
        vectors = [_pseudo_vector(text) for text in texts]
        vector_dim = VECTOR_DIM
    else:
        logger.info("Embedding %s dummy patents with %s", len(texts), settings.fastembed_model)
        embedder = get_fastembed_model(settings.fastembed_model)
        vectors = [vec.tolist() for vec in embedder.embed(texts)]
        vector_dim = len(vectors[0])

    rows: list[dict[str, object]] = []
    for patent, text, vector in zip(DUMMY_PATENTS, texts, vectors, strict=True):
        rows.append(
            {
                "patent_id": patent["patent_id"],
                "title": patent["title"],
                "abstract": patent["abstract"],
                "claim_snippet": patent["claim_snippet"],
                "cpc_codes": patent["cpc_codes"],
                "filing_date": patent["filing_date"],
                "assignee": patent["assignee"],
                "text": text,
                "vector": vector,
            }
        )

    db.create_table(table_name, data=rows, schema=_schema(vector_dim), mode="overwrite")

    if offline:
        (output_path / OFFLINE_INDEX_MARKER).write_text("offline\n", encoding="utf-8")

    logger.info("Test index ready at %s (%s rows)", flat_table_dir, len(rows))
    return flat_table_dir


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Create minimal PatentHub test index")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Lance path (default: PATENTHUB_LANCE_PATH / ./data/patents.lance)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing index directory before creating",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip fastembed model download; use deterministic pseudo-vectors",
    )
    args = parser.parse_args(argv)

    if args.offline:
        print(
            "WARNING: --offline uses pseudo-embeddings. Search in Streamlit still "
            "loads the real fastembed model — results are smoke-test only."
        )

    try:
        path = create_test_index(args.output, force=args.force, offline=args.offline)
    except Exception as exc:
        logger.error("Failed to create test index: %s", exc)
        return 1

    print(f"OK: test index at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())