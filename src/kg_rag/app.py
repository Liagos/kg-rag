import streamlit as st
from kg_rag.rag.qa import ask

st.set_page_config(
    page_title="Jira Ticket Assistant",
    page_icon="🎫",
    layout="wide",
)

st.title("🎫 Jira Ticket Assistant")
st.caption("Ask questions about your support tickets using ChromaDB, Neo4j, or both.")

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.selectbox(
        "Retrieval mode",
        options=["hybrid", "chroma", "neo4j"],
        index=0,
        help="chroma = semantic search | neo4j = graph filters | hybrid = both"
    )

    model = st.selectbox(
        "LLM model",
        options=["gpt-4.1", "claude-sonnet-4-6"],
        index=0,
    )

    st.divider()
    st.markdown("**Mode guide**")
    st.markdown("🔵 **chroma** — semantic similarity, great for vague queries")
    st.markdown("🟢 **neo4j** — exact filters, great for structured queries")
    st.markdown("🟣 **hybrid** — both, best for complex queries")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about your tickets..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching tickets..."):
            answer, meta = ask(                    # ← unpack tuple
                question=prompt,
                chat_history=st.session_state.chat_history,
                mode=mode,
                model=model,
            )

        st.markdown(answer)

        # source indicators — only show in hybrid mode
        if mode == "hybrid":
            col1, col2 = st.columns(2)
            col1.metric(
                "Semantic + BM25",
                "✅ results" if meta["chroma_hits"] > 0 else "⚠️ empty"
            )
            col2.metric(
                "Neo4j graph",
                "✅ results" if meta["neo4j_hits"] > 0 else "⚠️ empty"
            )

        # actionable hints when a source is empty
        if mode == "hybrid":
            if meta["chroma_hits"] == 0:
                st.info("💡 Semantic search found no results — try rephrasing your question.")
            if meta["neo4j_hits"] == 0:
                st.info("💡 Graph search found no results — try adding a ticket ID, product name, or region.")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append({
        "question": prompt,
        "answer":   answer,
    })