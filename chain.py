from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .config import settings

SYSTEM_PROMPT = """You are a knowledgeable assistant for a personal knowledge base.
Answer questions using ONLY the provided context. Be precise and cite sources.

Rules:
- If the answer isn't in the context, say "I don't have information on that."
- Always mention which document(s) you drew from.
- Format answers clearly with bullet points when listing multiple items.
- Keep answers concise unless asked for detail.

Context:
{context}"""


def _format_context(docs: list[Document]) -> str:
    """Format retrieved docs into a compact context block for the LLM."""
    parts: list[str] = []
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        page_str = str(page) if page is not None else "?"
        excerpt = doc.page_content
        parts.append(f"[{i}] {src} (page {page_str})\n{excerpt}")
    return "\n\n".join(parts)


@dataclass
class _InvokeResult:
    answer: str
    source_documents: list[Document]

    def as_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "source_documents": self.source_documents}


class SimpleRAGChain:
    """Minimal conversational RAG chain compatible with your installed langchain."""

    def __init__(self, *, retriever, llm: BaseChatModel):
        self.retriever = retriever
        self.llm = llm
        self.chat_history: list[tuple[str, str]] = []  # (user, assistant)

        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # Support multiple retriever interfaces.
        if hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.get_relevant_documents(query)
        # Runnable retrievers often accept {"query": ...}
        try:
            docs = self.retriever.invoke({"query": query})
        except Exception:
            docs = self.retriever.invoke(query)
        return list(docs)

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs.get("question", ""))

        docs = self._get_relevant_documents(question)
        context = _format_context(docs)

        # Provide a lightweight chat history to keep answers consistent.
        if self.chat_history:
            history_lines = []
            for u, a in self.chat_history[-5:]:
                history_lines.append(f"User: {u}\nAssistant: {a}")
            question_with_history = f"{question}\n\nChat history:\n" + "\n\n".join(history_lines)
        else:
            question_with_history = question

        messages = self._prompt.format_messages(context=context, question=question_with_history)
        response = self.llm.invoke(messages)

        # Chat models usually return an AIMessage with `.content`.
        answer = getattr(response, "content", None) or str(response)

        self.chat_history.append((question, answer))

        return _InvokeResult(answer=answer, source_documents=docs).as_dict()


def build_rag_chain(retriever) -> SimpleRAGChain:
    llm = ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=settings.OLLAMA_NUM_PREDICT,
    )
    return SimpleRAGChain(retriever=retriever, llm=llm)


def ask(chain: SimpleRAGChain, question: str) -> dict:
    """Query the chain, return answer + cited sources."""
    result = chain.invoke({"question": question})

    # Deduplicate source documents for display.
    seen: set[str] = set()
    sources: list[dict[str, str]] = []
    for doc in result.get("source_documents", []):
        src = str(doc.metadata.get("source", "Unknown"))
        page = str(doc.metadata.get("page", "?"))
        key = f"{src}:{page}"
        if key in seen:
            continue
        excerpt = doc.page_content[:200]
        sources.append({"file": src, "page": page, "excerpt": excerpt})
        seen.add(key)

    return {"answer": result["answer"], "sources": sources}
