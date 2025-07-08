#!/usr/bin/env python3

import gradio as gr
from src.rag.query_rag_pipeline import load_vector_store, answer_question

# Load vector store once at startup
vs = load_vector_store(
    vector_store_dir="vector_store/faiss_index",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
)


def chat_fn(message, history):
    answer, sources = answer_question(message, vs, top_k=5)

    # Prepare sources text
    sources_text = "\n".join(
        [f"[{i+1}] {chunk.page_content[:300]}..." for i, chunk in enumerate(sources)]
    )

    final_reply = f"{answer}\n\n**Sources:**\n{sources_text}"

    return final_reply


if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat_fn,
        title="💬 Financial Complaints RAG Chatbot",
        description="Ask any question about financial complaints. "
        "The chatbot responds based on customer complaint data and "
        "shows source excerpts.",
        theme=gr.themes.Default(),
    ).launch()
