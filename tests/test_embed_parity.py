"""Optional golden-vector parity test: sentence-transformers vs fastembed."""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("fastembed")

import numpy as np


@pytest.mark.slow
def test_offline_and_runtime_embedders_produce_similar_vectors() -> None:
    """ST (offline ingest) and fastembed (runtime) should be close for same model."""
    from fastembed import TextEmbedding
    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    text = "réseaux de neurones pour diagnostic médical"

    st_vec = SentenceTransformer(model_name).encode(text, normalize_embeddings=True)
    fe_vec = list(TextEmbedding(model_name=model_name).embed([text]))[0]

    st_arr = np.asarray(st_vec)
    fe_arr = np.asarray(fe_vec)
    cosine = np.dot(st_arr, fe_arr) / (np.linalg.norm(st_arr) * np.linalg.norm(fe_arr))
    assert cosine > 0.95, f"Embedding parity too low: cosine={cosine:.4f}"