from __future__ import annotations

import hashlib
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .config import settings

_embeddings: HuggingFaceEmbeddings | None = None
_vector_store: Chroma | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create embedding model once per process (first call can download weights)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store() -> Chroma:
    """Load or create persisted ChromaDB store (one instance per process)."""
    global _vector_store
    if _vector_store is None:
        persist_dir = Path(settings.CHROMA_PERSIST_PATH)
        persist_dir.mkdir(parents=True, exist_ok=True)
        _vector_store = Chroma(
            collection_name="knowledge_base",
            embedding_function=get_embeddings(),
            persist_directory=str(persist_dir),
        )
    return _vector_store


def _chunk_stable_id(chunk: Document) -> str:
    """Create a deterministic ID for a chunk to avoid re-embedding."""
    source = str(chunk.metadata.get("source", "unknown_source"))
    chunk_id = chunk.metadata.get("chunk_id")

    if chunk_id is not None:
        base = f"{source}::chunk_id::{chunk_id}"
    else:
        # Fallback: hash content (stable) if chunk_id isn't present.
        digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        base = f"{source}::content_sha256::{digest}"

    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _existing_ids(store: Chroma) -> set[str]:
    """Fetch already-indexed IDs from the underlying Chroma collection."""
    result = store._collection.get()  # noqa: SLF001 (intentional: Chroma internals)
    ids = result.get("ids") or []
    return set(map(str, ids))


def ingest_to_store(chunks: list[Document]) -> Chroma:
    """Embed chunks and upsert into ChromaDB (skip already-indexed chunks)."""
    store = get_vector_store()

    existing = _existing_ids(store)

    new_docs: list[Document] = []
    new_ids: list[str] = []
    for chunk in chunks:
        cid = _chunk_stable_id(chunk)
        if cid in existing:
            continue
        new_docs.append(chunk)
        new_ids.append(cid)

    if new_docs:
        store.add_documents(new_docs, ids=new_ids)
        print(f"✅ Added {len(new_docs)} new chunks")
    else:
        print("ℹ️ All documents already indexed")

    return store


def get_retriever(store: Chroma):
    """MMR retriever — balances relevance + diversity."""
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.TOP_K_RESULTS,
            "fetch_k": 20,
            "lambda_mult": 0.7,  # 1=max relevance, 0=max diversity
        },
    )

