import streamlit as st

st.set_page_config(page_title="PatentHub", page_icon="🔍", layout="centered")

st.title("🔎 PatentHub")
st.markdown("**Indexation intelligente de brevets IA • Blockchain • Astrophysique / Space Tech**")

query = st.text_input("Recherche brevets", placeholder="réseaux de neurones...")

col1, col2 = st.columns(2)
with col1:
    domains = st.multiselect("Domaine", ["IA", "Blockchain", "Space Tech"], default=["IA", "Blockchain", "Space Tech"])

with col2:
    num_results = st.slider("Nombre de résultats", min_value=1, max_value=50, value=10)

if st.button("🔍 Rechercher") and query:
    st.success(f"Résultats pour : **{query}** ({num_results} affichés)")
    
    count = 0
    if "IA" in domains and count < num_results:
        st.subheader("1. Réseaux de neurones pour diagnostic médical")
        st.write("Utilisation de deep learning pour analyser des images médicales.")
        st.caption("Score : 0.95 | Année : 2023")
        count += 1
    
    if "Blockchain" in domains and count < num_results:
        st.subheader("2. Blockchain Based Secure Federated Learning")
        st.write("Méthode sécurisée de machine learning distribué.")
        st.caption("Score : 0.88 | Année : 2024")
        count += 1
    
    if "Space Tech" in domains and count < num_results:
        st.subheader("3. AI Powered Satellite Image Processing")
        st.write("Analyse d'images satellitaires par intelligence artificielle.")
        st.caption("Score : 0.82 | Année : 2025")
        count += 1

st.caption("PatentHub MVP — Filtres avancés | Abubakr Adamou")