#!/usr/bin/env python3
# src/rag/query_rag_pipeline.py

import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline


VECTOR_STORE_DIR = "vector_store/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints. Use the following
retrieved complaint excerpts to formulate your answer. If the context doesn't
contain the answer, say you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""


def load_vectorstore():
    """Load FAISS vector store from disk."""
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        VECTOR_STORE_DIR, embeddings_model, allow_dangerous_deserialization=True
    )
    return vector_store


def retrieve_chunks(vector_store, query, top_k=TOP_K):
    """Embed question and retrieve top-k similar chunks."""
    results = vector_store.similarity_search(query, k=top_k)
    return results


def build_prompt(context_chunks, question):
    """Construct prompt for LLM."""
    context_texts = [doc.page_content for doc in context_chunks]
    context_str = "\n---\n".join(context_texts)
    prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)
    return prompt


def generate_answer(prompt):
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=512,
        temperature=0.2,
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    output_parser = StrOutputParser()
    chain = llm | output_parser
    answer = chain.invoke(prompt)
    return answer


def answer_question(question, vector_store, top_k=TOP_K):
    """End-to-end RAG process for a single user question."""
    chunks = retrieve_chunks(vector_store, question, top_k)
    prompt = build_prompt(chunks, question)
    answer = generate_answer(prompt)
    return answer, chunks


if __name__ == "__main__":
    vs = load_vectorstore()
    question = "How often do people mention fraud in credit cards?"
    answer, sources = answer_question(question, vs, top_k=TOP_K)

    print("\n=== GENERATED ANSWER ===")
    print(answer)

    print("\n=== RETRIEVED CONTEXT CHUNKS ===")
    for i, doc in enumerate(sources, 1):
        print(f"[{i}] {doc.page_content[:300]} ...\n")
