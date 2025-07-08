import sys
import os
import streamlit as st

# import time
from rag.query_rag_pipeline import answer_question, load_vector_store

# Add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)


# Load vectorstore once
vs = load_vector_store(
    "vector_store/faiss_index", "sentence-transformers/all-MiniLM-L6-v2"
)

# Page config
st.set_page_config(
    page_title="Financial Complaints RAG Chat", page_icon="💬", layout="wide"
)

# CSS for ChatGPT style UI
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    .main > div {
        height: 100vh;
        display: flex;
        flex-direction: column;
        background-color: var(--background-color);
        color: var(--text-color);
    }

    #chat-container {
        flex-grow: 1;
        overflow-y: auto;
        padding: 16px;
    }

    .chat-message {
        max-width: 75%;
        padding: 12px 16px;
        border-radius: 18px;
        margin-bottom: 10px;
        font-size: 16px;
        line-height: 1.4;
        white-space: pre-wrap;
        background-color: var(--bubble-color);
        color: var(--text-color);
    }

    .user-message {
        margin-left: auto;
        margin-right: 10px;
        text-align: right;
    }

    .ai-message {
        margin-left: 10px;
        margin-right: auto;
        text-align: left;
    }

    .sources {
        font-size: 14px;
        color: var(--text-secondary-color);
        margin-left: 10px;
        margin-right: 10px;
        margin-top: -8px;
        margin-bottom: 15px;
    }

    #input-container {
        padding: 16px;
        border-top: 1px solid var(--border-color);
        background-color: var(--background-color);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        position: sticky;
        bottom: 0;
    }

    .chatgpt-input-box {
        flex-grow: 1;
        display: flex;
        align-items: center;
        background-color: var(--input-bg-color);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 8px 12px;
        color: var(--text-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .chatgpt-input-box input {
        border: none;
        background: transparent;
        flex-grow: 1;
        font-size: 16px;
        color: var(--text-color);
        outline: none;
    }

    .chatgpt-input-box input::placeholder {
        color: var(--text-secondary-color);
        opacity: 0.6;
    }

    .chatgpt-input-box button {
        background: none;
        border: none;
        color: var(--text-secondary-color);
        cursor: pointer;
        font-size: 20px;
        padding: 4px 8px;
        transition: color 0.2s ease;
    }

    .chatgpt-input-box button:hover {
        color: var(--focus-color);
    }

    :root {
        --background-color: #ffffff;
        --text-color: #333333;
        --text-secondary-color: #666666;
        --bubble-color: #f5f5f5;
        --input-bg-color: #f0f0f0;
        --border-color: #d0d0d0;
        --focus-color: #10a37f;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #202123;
            --text-color: #f7f7f8;
            --text-secondary-color: #999999;
            --bubble-color: #343541;
            --input-bg-color: #40414f;
            --border-color: #3e3f4b;
            --focus-color: #10a37f;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("💬 Financial Complaints RAG Chatbot")

# Chat container with fixed height and scrolling
chat_container = st.container()

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Input and buttons in a fixed container at the bottom
def render_input():
    with st.form(key="input_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your question here:", key="input-text", label_visibility="collapsed"
        )
        submit = st.form_submit_button("Send")
        return user_input, submit


# Render chat messages inside scrollable container
with chat_container:
    st.markdown('<div id="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        st.markdown(
            f'<div class="chat-message user-message">🧑‍💻 {message["question"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="chat-message ai-message">🤖 {message["answer"]}</div>',
            unsafe_allow_html=True,
        )
        # Show sources truncated
        sources_md = "<br>".join(
            [
                f"- {src.page_content[:300].replace(chr(10), ' ')}..."  # chr(10) is \n
                for src in message["sources"]
            ]
        )
        st.markdown(
            f'<div class="sources"><b>🔎 Sources used:</b><br>{sources_md}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# Render input form fixed at bottom
user_input, submitted = render_input()

# Process input
if submitted and user_input.strip() != "":
    answer, sources = answer_question(user_input, vs, top_k=5)
    st.session_state.chat_history.append(
        {
            "question": user_input,
            "answer": answer,
            "sources": sources,
        }
    )
    # Rerun to update chat
    st.rerun()
