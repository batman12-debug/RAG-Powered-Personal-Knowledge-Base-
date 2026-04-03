# RAG Powered Personal Knowledge Base

A small **retrieval-augmented generation (RAG)** app: index your own PDFs and text files, store embeddings in **Chroma**, and chat with a **local LLM** via **Ollama** — all through a **Streamlit** UI.

## Features

- **Ingest** PDFs, `.txt`, and `.md` from a folder you choose
- **Chunking** with overlap and stable chunk IDs (skip re-embedding duplicates)
- **Local embeddings** with [Sentence Transformers](https://www.sbert.net/) (default: `all-MiniLM-L6-v2`)
- **Vector search** with Chroma (persisted on disk)
- **Q&A** with LangChain + **ChatOllama**; answers can show **source excerpts**
- **UI** tuned for a soft, animated experience (with `prefers-reduced-motion` support)

## Prerequisites

- **Python** 3.11 or 3.12 recommended (3.14 may hit missing wheels for some dependencies)
- **[Ollama](https://ollama.com/)** installed and running, with a chat model pulled, e.g.  
  `ollama pull llama3.2`

## Setup

1. **Clone or copy** this project and open a terminal in the project root.

2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment file** — copy or edit `.env` in the project root (see [Configuration](#configuration)). All keys have sensible defaults except that you must have a working Ollama model for chat.

5. **Optional: data folder** — create `./data` (or any path you will use in the UI) and add documents:

   ```bash
   mkdir -p data
   ```

## Configuration

Variables are read from `.env` (and defaults in `src/config.py`):

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `OLLAMA_BASE_URL` | Ollama API base URL | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | Model name (must be pulled in Ollama) | `llama3.2` |
| `OLLAMA_NUM_PREDICT` | Max tokens for the LLM reply | `2048` |
| `CHROMA_PERSIST_PATH` | Chroma persistence directory | `./chroma_db` |
| `EMBEDDING_MODEL` | Sentence Transformers model id | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Chunk character length | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K_RESULTS` | Chunks retrieved per question | `4` |

## Run the app

From the **project root** (so `src` imports resolve):

```bash
PYTHONPATH=. streamlit run ui/app.py
```

Open the URL Streamlit prints (usually **http://localhost:8501**). Prefer a normal browser; some IDE embedded previews do not render Streamlit reliably.

### Workflow

1. In the sidebar, set **Data directory** (e.g. `./data`).
2. Click **Index** to load, chunk, and embed documents into Chroma.
3. Use the chat box to ask questions; expand **Sources** on assistant messages when available.

## Project layout

```
├── README.md
├── requirements.txt
├── .env                    # local config (not committed if you use .gitignore)
├── .streamlit/
│   └── config.toml         # theme + file watcher (reduces noisy logs)
├── data/                   # put your documents here (or use another folder)
├── src/
│   ├── config.py           # settings from .env
│   ├── ingest.py           # load & chunk documents
│   ├── retriever.py        # embeddings, Chroma, retriever
│   └── chain.py            # RAG prompt + Ollama chat
└── ui/
    └── app.py              # Streamlit UI
```

## Troubleshooting

- **`ModuleNotFoundError` for `src`** — run with `PYTHONPATH=. ` from the project root, or `cd` to the root first.
- **First run is slow** — the embedding model may download (~90MB). Wait for the loading spinner to finish.
- **Ollama errors** — ensure `ollama serve` is running and `ollama list` includes `OLLAMA_MODEL`.
- **Streamlit + `transformers` log spam** (`torchvision`, etc.) — this project sets `fileWatcherType = "none"` in `.streamlit/config.toml` so the dev server does not crawl large optional `transformers` submodules. You lose auto-reload on save; restart Streamlit after code changes.
- **`st.chat_message` avatar errors** — custom `avatar=` values must be image URLs or paths, not emoji. This app uses default avatars.

## Tech stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/) + [langchain-ollama](https://pypi.org/project/langchain-ollama/)
- [Chroma](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration

## License

No license file is included in this template; add one if you distribute the project.
