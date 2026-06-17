# PatentHub v0 — Recherche Brevets IA (G06N)

**FR** · Hub open-source de recherche sémantique de brevets USPTO (CPC G06N — intelligence artificielle).  
**EN** · Open-source semantic search for USPTO AI patents (CPC G06N).

> ⚠️ Outil de recherche, pas un conseil juridique / Search tool only — not legal advice.

## Fonctionnalités / Features

- Requête en langage naturel (FR/EN) → top-10 brevets G06N
- Index vectoriel LanceDB (~5 s sur mobile)
- Embeddings multilingues HF (`paraphrase-multilingual-MiniLM-L12-v2`, fastembed ONNX en runtime)
- Zéro budget : Streamlit Cloud + GitHub Releases

## Architecture

```
ingest/   → PatentsView API → sentence-transformers → LanceDB
search/   → fastembed ONNX → LanceDB vector search
app/      → Streamlit UI (mobile-first)
```

L'index LanceDB est publié en **GitHub Release** (pas dans git).

## Installation locale

```bash
git clone <repo-url>
cd PatentHub
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Offline ingest only (not Streamlit Cloud):
pip install -e ".[ingest]"
```

### Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `PATENTHUB_RELEASE_URL` | — | URL de l'archive index (GitHub Release) |
| `PATENTHUB_LANCE_PATH` | `./data/patents.lance` | Chemin local LanceDB |
| `PATENTHUB_TOP_K` | `10` | Nombre de résultats |
| `PATENTSVIEW_API_KEY` | — | Clé API PatentsView (optionnelle) |

### Construire l'index (offline)

```bash
pip install -e ".[ingest]"
export PATENTSVIEW_API_KEY=your_key   # optional
python -m ingest.build_index --limit 100 --reset-checkpoint
```

### Lancer l'app Streamlit

```bash
# Option A: index local
export PATENTHUB_LANCE_PATH=./data/patents.lance
streamlit run app/streamlit_app.py

# Option B: télécharger depuis Release
export PATENTHUB_RELEASE_URL=https://github.com/ORG/REPO/releases/download/index-latest/patents-index.tar.gz
streamlit run app/streamlit_app.py
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Le test `test_ingest_resume.py` vérifie la reprise après échec simulé (checkpoint).

## Déploiement Streamlit Cloud

1. Pousser le repo sur GitHub
2. [share.streamlit.io](https://share.streamlit.io) → New app
3. Main file: `app/streamlit_app.py`
4. Secrets (Settings → Secrets):

```toml
PATENTHUB_RELEASE_URL = "https://github.com/ORG/REPO/releases/download/index-latest/patents-index.tar.gz"
```

5. Deploy — l'index est téléchargé au premier chargement (`@st.cache_resource`).

## Publier l'index (GitHub Release)

```bash
# Manuel
./scripts/publish_index.sh index-latest ./data/patents.lance

# Ou via GitHub Actions → "Publish Index" workflow
```

## Structure

```
PatentHub/
├── config.py
├── ingest/{fetch,embed,build_index}.py
├── search/{models,query}.py
├── app/streamlit_app.py
├── tests/
├── scripts/publish_index.sh
└── .github/workflows/{ci,publish-index}.yml
```

## Licence

Open-source — contributions bienvenues.