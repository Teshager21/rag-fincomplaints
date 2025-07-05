#!/usr/bin/env python3
# src/embeddings/create_vector_store.py

"""
Vector Store Creation Script for RAG Complaint Analysis

- Loads cleaned complaint narratives
- Splits text into chunks for efficient embeddings
- Embeds each chunk using SentenceTransformers
- Stores embeddings + metadata in a FAISS vector database
"""

import os
import pandas as pd
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# --------------------------------------------
# CONFIG
# --------------------------------------------

INPUT_CSV_PATH = "data/interim/filtered_complaints.csv"
VECTOR_STORE_DIR = "vector_store/faiss_index"

CHUNK_SIZE = 300  # Experimented and chosen for short narratives
CHUNK_OVERLAP = 50  # Small overlap preserves context between chunks

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------
# FUNCTIONS
# --------------------------------------------


def load_cleaned_data(path):
    print(f"Loading cleaned complaints data from {path} ...")
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows.")
    return df


def chunk_texts(df, chunk_size=300, chunk_overlap=50):
    """Split narratives into overlapping chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    documents = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking narratives"):
        text = str(row["Cleaned Narrative"])
        chunks = splitter.split_text(text)

        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "complaint_id": int(row["Complaint ID"]),
                    "product": row["Product"],
                    "original_narrative": text,
                },
            )
            documents.append(doc)

    print(f"Created {len(documents)} text chunks.")
    return documents


def embed_and_store(docs, embedding_model_name, save_dir):
    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print("Embedding chunks and building FAISS index...")
    vector_store = FAISS.from_documents(docs, embedding=embeddings_model)

    os.makedirs(save_dir, exist_ok=True)
    vector_store.save_local(save_dir)
    print(f"Vector store saved to: {save_dir}")


# --------------------------------------------
# MAIN
# --------------------------------------------

if __name__ == "__main__":
    df_cleaned = load_cleaned_data(INPUT_CSV_PATH)
    documents = chunk_texts(
        df_cleaned, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    embed_and_store(documents, EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR)

    print("✅ Vector store creation complete.")
