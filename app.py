import streamlit as st
import time
import sys
import os

# Import your RAG pipeline functions
# ----------------------------------------------------
from rag.query_rag_pipeline import answer_question, load_vectorstore

# ----------------------------------------------------

# Add the "src" directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Load vector store once at startup
vs = load_vectorstore()

st.set_page_config(
    page_title="Financial Complaints RAG Chat", page_icon="💬", layout="wide"
)

st.title("💬 Financial Complaints RAG Chatbot")

st.markdown(
    """
Ask any question related to financial complaints.
The AI will answer based on real consumer complaint data.
Below each answer, you'll see the retrieved text excerpts
used as context for transparency.
"""
)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Text input
user_input = st.text_input("Type your question here:", value="", key="user_input")

col1, col2 = st.columns([1, 5])

with col1:
    ask_button = st.button("Ask", type="primary")

with col2:
    clear_button = st.button("Clear Chat")

# Clear chat history if requested
if clear_button:
    st.session_state.chat_history = []
    st.experimental_rerun()

# Process user query
if ask_button and user_input.strip() != "":
    # Run RAG pipeline
    answer, sources = answer_question(user_input, vs, top_k=5)

    # Store in chat history
    st.session_state.chat_history.append(
        {"question": user_input, "answer": answer, "sources": sources}
    )

# Display conversation
for message in st.session_state.chat_history:
    st.markdown(f"**🧑‍💻 You:** {message['question']}")

    # Simulate streaming effect
    placeholder = st.empty()
    partial = ""
    for char in message["answer"]:
        partial += char
        placeholder.markdown(f"**🤖 AI:** {partial}")
        time.sleep(0.01)

    st.markdown("<br>", unsafe_allow_html=True)

    # Show retrieved context chunks
    st.markdown("**🔎 Sources Used:**")
    for i, chunk in enumerate(message["sources"], start=1):
        st.info(f"[{i}] {chunk}")

    st.markdown("---")
