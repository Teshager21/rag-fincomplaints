import sys
import os

# import time
import streamlit as st

# Import RAG pipeline
from rag.query_rag_pipeline import answer_question, load_vector_store

# Add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)


# Cache vector store
@st.cache_resource(show_spinner="Loading vector store...")
def get_vector_store():
    return load_vector_store(
        "vector_store/faiss_index", "sentence-transformers/all-MiniLM-L6-v2"
    )


vs = get_vector_store()

st.set_page_config(
    page_title="Financial Complaints RAG Chat", page_icon="💬", layout="wide"
)

st.title("💬 Financial Complaints RAG Chatbot")

st.markdown(
    """
Ask any question about financial complaints.
The AI will answer based on real customer complaints.
Below each answer, you'll see the retrieved excerpts used as context.
"""
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Your question:", value="", key="user_input")

col1, col2 = st.columns([1, 5])

with col1:
    ask_button = st.button("Ask", type="primary")

with col2:
    clear_button = st.button("Clear Chat")

if clear_button:
    st.session_state.chat_history = []
    st.rerun()

if ask_button and user_input.strip() != "":
    answer, sources = answer_question(user_input, vs, top_k=5)
    st.session_state.chat_history.append(
        {"question": user_input, "answer": answer, "sources": sources}
    )

# --- Styling ---
chat_container = st.container()

with chat_container:
    st.markdown(
        """
        <style>
        .chat-message {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            margin-bottom: 10px;
            font-size: 16px;
            line-height: 1.4;
        }
        .user-message {
            background-color: var(--primary-color);
            color: var(--text-color);
            margin-left: auto;
            margin-right: 10px;
            text-align: right;
        }
        .ai-message {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            margin-left: 10px;
            margin-right: auto;
            text-align: left;
        }
        .sources {
            font-size: 14px;
            color: var(--text-color);
            margin-left: 10px;
            margin-right: 10px;
            margin-top: -8px;
            margin-bottom: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for msg in st.session_state.chat_history:
        st.markdown(
            f'<div class="chat-message user-message">🧑‍💻 {msg["question"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="chat-message ai-message">🤖 {msg["answer"]}</div>',
            unsafe_allow_html=True,
        )
        sources_md = "<br>".join(
            [f"- {src.page_content[:300]}..." for src in msg["sources"]]
        )
        st.markdown(
            f'<div class="sources"><b>🔎 Sources used:</b><br>{sources_md}</div>',
            unsafe_allow_html=True,
        )
