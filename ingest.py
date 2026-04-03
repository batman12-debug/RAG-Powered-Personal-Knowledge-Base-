from __future__ import annotations

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import settings


def load_documents(data_dir: str) -> list[Document]:
    """Load PDFs and .txt/.md files from a directory."""
    loaders = [
        DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader),
    ]

    docs: list[Document] = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            # Keep ingestion running even if one loader fails.
            print(f"Loader error ({type(loader).__name__}): {e}")

    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split docs into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,  # tracks source position
    )

    chunks = splitter.split_documents(docs)

    # Enrich metadata for citation display / stable IDs.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["char_count"] = len(chunk.page_content)

    return chunks

