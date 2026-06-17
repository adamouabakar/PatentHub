"""PatentHub Streamlit search UI — mobile-first, French primary."""

from __future__ import annotations

import httpx
import streamlit as st

from app.validation import EMPTY_QUERY_MESSAGE, INDEX_DOWNLOAD_MESSAGE, validate_search_input
from config import get_settings
from search.query import IndexDownloadError, _get_fastembed_model, load_index, search_patents

st.set_page_config(
    page_title="PatentHub — Recherche Brevets IA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MOBILE_CSS = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 720px; }
    h1 { font-size: 1.6rem !important; line-height: 1.3; }
    .patent-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
        background: #fafafa;
    }
    .disclaimer { font-size: 0.85rem; color: #666; margin-top: 2rem; }
    @media (max-width: 640px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 1.35rem !important; }
    }
</style>
"""
st.markdown(MOBILE_CSS, unsafe_allow_html=True)


@st.cache_resource
def _cached_load_index():
    return load_index(get_settings())


@st.cache_resource
def _cached_embed_model():
    settings = get_settings()
    return _get_fastembed_model(settings.fastembed_model)


def _google_patents_url(patent_id: str) -> str:
    return f"https://patents.google.com/patent/US{patent_id}"


def main() -> None:
    st.title("🔍 PatentHub")
    st.caption(
        "Recherche sémantique de brevets USPTO (CPC G06N — IA) / "
        "Semantic search for AI patents (G06N)"
    )

    query = st.text_input(
        "Votre recherche / Your query",
        placeholder="Ex : réseaux de neurones pour diagnostic médical",
        label_visibility="collapsed",
    )

    settings = get_settings()
    search_clicked = st.button("Rechercher", type="primary", use_container_width=True)

    validation_msg = validate_search_input(query, search_clicked)
    if validation_msg:
        st.warning(validation_msg)
    elif search_clicked and query.strip():
        with st.spinner("Recherche en cours…"):
            try:
                table = _cached_load_index()
                embed_model = _cached_embed_model()
                results = search_patents(
                    query.strip(),
                    top_k=settings.top_k,
                    table=table,
                    settings=settings,
                    embed_model=embed_model,
                )
            except FileNotFoundError as exc:
                st.error(
                    "Index non disponible. Configurez PATENTHUB_RELEASE_URL ou "
                    "construisez l'index localement.\n\n"
                    f"{exc}"
                )
                return
            except (IndexDownloadError, httpx.HTTPError):
                st.error(INDEX_DOWNLOAD_MESSAGE)
                return
            except Exception as exc:
                st.error(f"Erreur de recherche : {exc}")
                return

        if not results:
            st.info("Aucun résultat trouvé.")
            return

        st.subheader(f"Top {len(results)} résultats")
        for i, hit in enumerate(results, 1):
            patent_id = hit.get("patent_id", "")
            title = hit.get("title") or "(sans titre)"
            score = hit.get("score")
            assignee = hit.get("assignee") or "—"
            filing = hit.get("filing_date") or "—"
            abstract = hit.get("abstract") or ""
            snippet = abstract[:280] + ("…" if len(abstract) > 280 else "")
            gp_link = _google_patents_url(patent_id)

            score_txt = f" · score {score:.3f}" if score is not None else ""
            st.markdown(
                f'<div class="patent-card">'
                f"<strong>{i}. {title}</strong><br/>"
                f"<small>US{patent_id} · {assignee} · {filing}{score_txt}</small><br/>"
                f"<p style='margin:0.5rem 0 0.25rem'>{snippet}</p>"
                f'<a href="{gp_link}" target="_blank">Voir sur Google Patents →</a>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        '<p class="disclaimer">'
        "⚠️ Outil de recherche, pas un conseil juridique. "
        "Search tool only — not legal advice."
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()