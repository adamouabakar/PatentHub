import streamlit as st
import pandas as pd
import lancedb
import os

st.set_page_config(page_title="PatentHub", page_icon="🔍")
st.title("🔎 PatentHub")
st.markdown("**Recherche de brevets IA (G06N)**")

# Chargement de l'index
@st.cache_resource
def load_index():
    try:
        db = lancedb.connect("data/patents.lance")
        if "patents" in db.table_names():
            return db.open_table("patents")
    except:
        pass
    return None

table = load_index()

query = st.text_input("Recherche brevets", placeholder="réseaux de neurones, machine learning...")

if st.button("Rechercher") and query:
    if table is None:
        st.info("Mode test - Résultats simulés")
        results = [
            {"title": "Réseaux de neurones pour diagnostic médical", "abstract": "Utilisation de deep learning pour analyser des images médicales", "score": "0.95"},
            {"title": "Machine Learning appliqué à la blockchain", "abstract": "Amélioration de la sécurité des transactions avec IA", "score": "0.88"},
            {"title": "Satellite IA pour observation terrestre", "abstract": "Utilisation de l'intelligence artificielle pour l'astrophysique", "score": "0.75"},
        ]
    else:
        # Recherche par mots-clés (simple et fiable)
        results_df = table.search(query, query_type="fts").limit(10).to_pandas()
        results = results_df.to_dict('records')
    
    for r in results:
        st.subheader(r.get("title", "Brevet"))
        st.write(r.get("abstract", ""))
        st.caption(f"Score: {r.get('score', 'N/A')}")
        st.divider()

st.caption("PatentHub MVP - Recherche par mots-clés (version stable)")