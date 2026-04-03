from __future__ import annotations

import streamlit as st

from src.chain import ask, build_rag_chain
from src.config import settings
from src.ingest import chunk_documents, load_documents
from src.retriever import get_retriever, get_vector_store, ingest_to_store


def _inject_ui_styles() -> None:
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {
        --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
        --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
        --dur-press: 140ms;
        --dur-ui: 220ms;
        --dur-hero: 280ms;
        --c-ink: #3F3A45;
        --c-ink-soft: #6B6575;
        --c-lilac: #B8A4D9;
        --c-lilac-deep: #8E7AB5;
        --c-blush: #E8D4E0;
        --c-mint: #B8D9C9;
        --c-cream: #FDF9FB;
        --glass: rgba(255, 255, 255, 0.72);
        --glass-border: rgba(184, 164, 217, 0.22);
    }
    html, body, [class*="css"] {
        font-family: 'Outfit', ui-sans-serif, system-ui, sans-serif !important;
    }
    .stApp {
        position: relative;
        background: var(--c-cream) !important;
        background-attachment: fixed !important;
    }
    /* Ambient layer — slow transform-only drift (decorative) */
    .stApp::before {
        content: "";
        position: fixed;
        inset: -15%;
        z-index: 0;
        pointer-events: none;
        background:
            radial-gradient(ellipse 55% 45% at 15% 25%, rgba(232, 212, 224, 0.55) 0%, transparent 52%),
            radial-gradient(ellipse 50% 42% at 88% 18%, rgba(200, 184, 230, 0.4) 0%, transparent 48%),
            radial-gradient(ellipse 45% 50% at 75% 88%, rgba(184, 217, 201, 0.38) 0%, transparent 50%),
            linear-gradient(168deg, #FDF9FB 0%, #F8F2FA 38%, #F2F8F5 100%);
        animation: ambientDrift 32s var(--ease-in-out) infinite alternate;
        will-change: transform;
    }
    @keyframes ambientDrift {
        from { transform: translate(0, 0) scale(1); }
        to { transform: translate(1.5%, -1%) scale(1.03); }
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stSidebar"] {
        position: relative;
        z-index: 1;
    }
    [data-testid="stHeader"] {
        background: rgba(253, 249, 251, 0.82) !important;
        backdrop-filter: blur(14px) saturate(1.2);
        -webkit-backdrop-filter: blur(14px) saturate(1.2);
        border-bottom: 1px solid rgba(184, 164, 217, 0.14);
        transition: background-color 200ms var(--ease-out);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(175deg,
            rgba(253, 249, 251, 0.95) 0%,
            rgba(245, 238, 250, 0.92) 45%,
            rgba(238, 248, 243, 0.88) 100%) !important;
        border-right: 1px solid rgba(184, 164, 217, 0.16);
        box-shadow: 4px 0 40px rgba(110, 90, 140, 0.04);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--c-lilac-deep) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    .sidebar-brand {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(110deg, var(--c-lilac-deep) 0%, #A894C8 50%, #6A9B86 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }
    .sidebar-card {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        margin: 0.75rem 0 1rem 0;
        backdrop-filter: blur(8px);
        transition: border-color 200ms var(--ease-out), box-shadow 200ms var(--ease-out);
    }
    @media (hover: hover) and (pointer: fine) {
        .sidebar-card:hover {
            border-color: rgba(184, 164, 217, 0.32);
            box-shadow: 0 8px 28px rgba(110, 90, 140, 0.07);
        }
    }
    /* Hero + feature column */
    .hero-shell {
        animation: shellIn var(--dur-hero) var(--ease-out) both;
    }
    @keyframes shellIn {
        from { opacity: 0; transform: translateY(10px) scale(0.985); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    .hero-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(145deg,
            rgba(255,255,255,0.88) 0%,
            rgba(250, 244, 252, 0.82) 40%,
            rgba(242, 250, 246, 0.78) 100%);
        border: 1px solid rgba(184, 164, 217, 0.2);
        border-radius: 24px;
        padding: 1.65rem 1.85rem 1.5rem 1.85rem;
        margin-bottom: 0.5rem;
        box-shadow:
            0 4px 24px rgba(110, 90, 140, 0.06),
            0 12px 48px rgba(110, 90, 140, 0.05),
            inset 0 1px 0 rgba(255,255,255,0.85);
        transform-origin: 50% 0%;
    }
    .hero-card::after {
        content: "";
        position: absolute;
        top: -40%;
        right: -20%;
        width: 55%;
        height: 80%;
        background: radial-gradient(circle, rgba(184, 164, 217, 0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-kicker {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--c-lilac-deep);
        opacity: 0.85;
        margin: 0 0 0.5rem 0;
        animation: staggerIn var(--dur-hero) var(--ease-out) both;
        animation-delay: 45ms;
    }
    .hero-title {
        font-size: clamp(1.45rem, 2.5vw, 1.85rem);
        font-weight: 700;
        letter-spacing: -0.035em;
        line-height: 1.2;
        margin: 0 0 0.55rem 0;
        background: linear-gradient(115deg, #6B5A82 0%, var(--c-lilac-deep) 35%, #7A9E8C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: staggerIn var(--dur-hero) var(--ease-out) both;
        animation-delay: 85ms;
    }
    .hero-sub {
        color: var(--c-ink-soft);
        font-size: 0.97rem;
        font-weight: 400;
        line-height: 1.55;
        margin: 0;
        max-width: 36rem;
        animation: staggerIn var(--dur-hero) var(--ease-out) both;
        animation-delay: 125ms;
    }
    @keyframes staggerIn {
        from { opacity: 0; transform: translateY(8px) scale(0.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 1.1rem;
        animation: staggerIn var(--dur-hero) var(--ease-out) both;
        animation-delay: 165ms;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.74rem;
        font-weight: 500;
        padding: 0.38rem 0.8rem;
        border-radius: 999px;
        background: rgba(184, 164, 217, 0.16);
        color: var(--c-lilac-deep);
        border: 1px solid rgba(184, 164, 217, 0.28);
    }
    .pill-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--c-mint);
        box-shadow: 0 0 0 3px rgba(184, 217, 201, 0.35);
        animation: livePulse 3s linear infinite;
    }
    @keyframes livePulse {
        0%, 100% { opacity: 0.65; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.12); }
    }
    /* Feature stack (right column) */
    .feature-stack {
        display: flex;
        flex-direction: column;
        gap: 0.65rem;
        padding-top: 0.25rem;
    }
    .feature-tile {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.85rem 1rem;
        border-radius: 16px;
        background: var(--glass);
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(10px);
        transition: transform 200ms var(--ease-out), border-color 200ms var(--ease-out), box-shadow 200ms var(--ease-out);
        animation: tileIn var(--dur-ui) var(--ease-out) both;
    }
    .feature-tile:nth-child(1) { animation-delay: 100ms; }
    .feature-tile:nth-child(2) { animation-delay: 155ms; }
    .feature-tile:nth-child(3) { animation-delay: 210ms; }
    @keyframes tileIn {
        from { opacity: 0; transform: translateX(10px) scale(0.99); }
        to { opacity: 1; transform: translateX(0) scale(1); }
    }
    @media (hover: hover) and (pointer: fine) {
        .feature-tile:hover {
            transform: translateY(-2px);
            border-color: rgba(184, 164, 217, 0.35);
            box-shadow: 0 10px 32px rgba(110, 90, 140, 0.08);
        }
    }
    .feature-icon {
        font-size: 1.25rem;
        line-height: 1;
        opacity: 0.92;
    }
    .feature-tile h4 {
        margin: 0;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--c-ink);
        letter-spacing: -0.02em;
    }
    .feature-tile p {
        margin: 0.2rem 0 0 0;
        font-size: 0.78rem;
        color: var(--c-ink-soft);
        line-height: 1.45;
    }
    .section-chat-label {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 1.35rem 0 0.65rem 0;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--c-lilac-deep);
        opacity: 0.88;
    }
    .section-chat-label::after {
        content: "";
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(184,164,217,0.35), transparent);
    }
    @keyframes msgEnter {
        from { opacity: 0; transform: translateY(6px) scale(0.985); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    [data-testid="stChatMessage"] {
        animation: msgEnter var(--dur-ui) var(--ease-out) both;
        transform-origin: 50% 0%;
        background: rgba(255, 255, 255, 0.58) !important;
        border: 1px solid rgba(184, 164, 217, 0.14) !important;
        border-radius: 18px !important;
        transition: border-color 180ms var(--ease-out), box-shadow 180ms var(--ease-out);
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stChatMessage"]:focus-within {
        border-color: rgba(184, 164, 217, 0.28) !important;
        box-shadow: 0 6px 28px rgba(110, 90, 140, 0.07);
    }
    .stButton > button {
        border-radius: 14px !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
        border: none !important;
        transition:
            transform var(--dur-press) var(--ease-out),
            box-shadow 200ms var(--ease-out) !important;
        box-shadow: 0 3px 14px rgba(184, 164, 217, 0.28) !important;
        transform-origin: center;
    }
    @media (hover: hover) and (pointer: fine) {
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 22px rgba(184, 164, 217, 0.32) !important;
        }
    }
    .stButton > button:active {
        transform: scale(0.97);
        transition-duration: 100ms;
    }
    @media (hover: hover) and (pointer: fine) {
        .stButton > button:active {
            transform: scale(0.97) translateY(0);
        }
    }
    [data-testid="stExpander"] {
        border: 1px solid rgba(184, 164, 217, 0.16) !important;
        border-radius: 14px !important;
        overflow: hidden;
        background: rgba(255,255,255,0.45) !important;
        transition: border-color 200ms var(--ease-out), background-color 200ms var(--ease-out);
    }
    @media (hover: hover) and (pointer: fine) {
        [data-testid="stExpander"]:hover {
            border-color: rgba(184, 164, 217, 0.28) !important;
        }
    }
    [data-testid="stChatInput"] > div {
        border-radius: 18px !important;
    }
    div[data-testid="stChatInput"] textarea {
        border-radius: 16px !important;
        transition: box-shadow 200ms var(--ease-out), border-color 200ms var(--ease-out);
    }
    div[data-testid="stChatInput"] textarea:focus {
        box-shadow: 0 0 0 3px rgba(184, 164, 217, 0.22) !important;
    }
    .stTextInput input {
        border-radius: 12px !important;
        transition: box-shadow 200ms var(--ease-out), border-color 200ms var(--ease-out);
    }
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(184,164,217,0.22), transparent);
        margin: 1rem 0;
    }
    @media (prefers-reduced-motion: reduce) {
        :root {
            --dur-press: 0ms;
            --dur-ui: 1ms;
            --dur-hero: 1ms;
        }
        .stApp::before {
            animation: none !important;
        }
        .hero-shell, .hero-kicker, .hero-title, .hero-sub, .pill-row,
        .feature-tile, [data-testid="stChatMessage"] {
            animation: none !important;
        }
        .pill-dot { animation: none !important; opacity: 0.9; }
        .stButton > button,
        .stButton > button:hover,
        .stButton > button:active {
            transition: none !important;
            transform: none !important;
        }
        .feature-tile:hover { transform: none !important; }
    }
</style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Personal Knowledge Base",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
_inject_ui_styles()

with st.sidebar:
    st.markdown('<p class="sidebar-brand">Knowledge base</p>', unsafe_allow_html=True)
    st.caption("Index documents, then chat — answers stay grounded in your files.")
    st.markdown(
        """
<div class="sidebar-card">
    <span style="font-size:0.85rem;font-weight:600;color:#6B5A82;">Quick tips</span>
    <p style="margin:0.5rem 0 0 0;font-size:0.8rem;color:#6B6575;line-height:1.5;">
        Put PDFs, .txt, or .md in your data folder, run <b>Index</b>, then ask questions here.
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("##### Indexing")
    data_dir = st.text_input("Data directory", value="./data", help="Folder with PDF, .txt, or .md files")

    col_a, col_b = st.columns(2)
    with col_a:
        index_clicked = st.button("Index", use_container_width=True, type="primary")
    with col_b:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.pop("messages", None)
            st.rerun()

    if index_clicked:
        with st.spinner("Reading & chunking…"):
            docs = load_documents(data_dir)
            chunks = chunk_documents(docs)
        with st.spinner("Embedding into Chroma…"):
            store = ingest_to_store(chunks)
        st.session_state.pop("chain", None)
        st.session_state.pop("retriever", None)
        st.success(f"Indexed **{len(chunks)}** chunks · `{store._collection.name}`")
        st.balloons()

    st.divider()
    st.caption(f"**Ollama model** · `{settings.OLLAMA_MODEL}`")
    st.caption(f"**Endpoint** · `{settings.OLLAMA_BASE_URL}`")


def get_or_create_chain():
    if "chain" in st.session_state:
        return st.session_state.chain
    with st.spinner("Loading embeddings & vector store (first run may take a minute)…"):
        store = get_vector_store()
        st.session_state.retriever = get_retriever(store)
        st.session_state.chain = build_rag_chain(st.session_state.retriever)
    return st.session_state.chain


chain = get_or_create_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Hero + features ---
hc, fc = st.columns([1.25, 1], gap="large")
with hc:
    st.markdown(
        """
<div class="hero-shell">
  <div class="hero-card">
    <p class="hero-kicker">Private RAG</p>
    <p class="hero-title">Ask your library anything</p>
    <p class="hero-sub">Natural-language Q&amp;A over <strong>your</strong> indexed documents. Replies use retrieved context only — calm, local, and under your control.</p>
    <div class="pill-row">
      <span class="pill"><span class="pill-dot"></span> Live stack</span>
      <span class="pill">Chroma</span>
      <span class="pill">Ollama</span>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
with fc:
    st.markdown(
        """
<div class="feature-stack">
  <div class="feature-tile">
    <span class="feature-icon">📚</span>
    <div><h4>Index once</h4><p>Chunk &amp; embed PDFs and text into a persistent vector store.</p></div>
  </div>
  <div class="feature-tile">
    <span class="feature-icon">🔮</span>
    <div><h4>Grounded answers</h4><p>Retrieval + local LLM — citations in each reply when sources exist.</p></div>
  </div>
  <div class="feature-tile">
    <span class="feature-icon">🌸</span>
    <div><h4>Soft &amp; fast UI</h4><p>Motion stays snappy; turn on reduced motion in OS if you prefer less animation.</p></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-chat-label">Conversation</div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for s in msg["sources"]:
                    st.caption(f"{s['file']} · page {s['page']}")
                    st.code(s["excerpt"], language=None)

if prompt := st.chat_input("Message your knowledge base…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = ask(chain, prompt)
        st.markdown(result["answer"])
        sources = result.get("sources", [])
        if sources:
            with st.expander("Sources", expanded=False):
                for s in sources:
                    st.caption(f"{s['file']} · page {s['page']}")
                    st.code(s["excerpt"], language=None)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": sources,
        }
    )
