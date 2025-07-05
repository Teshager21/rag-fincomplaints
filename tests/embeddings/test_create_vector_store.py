# tests/embeddings/test_create_vector_store.py

# import os
# import shutil
import pandas as pd

import pytest

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

# Import functions directly
from src.embeddings import create_vector_store as cvs


@pytest.fixture
def dummy_dataframe():
    """Create a minimal test DataFrame"""
    data = {
        "Complaint ID": [1, 2],
        "Product": ["Credit card", "Buy Now, Pay Later (BNPL)"],
        "Cleaned Narrative": [
            "This is a test complaint about credit cards. It has some details.",
            "Another test complaint for BNPL. Many users are unhappy with fees.",
        ],
    }
    return pd.DataFrame(data)


def test_chunk_texts(dummy_dataframe):
    chunks = cvs.chunk_texts(dummy_dataframe, chunk_size=20, chunk_overlap=5)

    # Ensure chunks were created
    assert len(chunks) > 0
    assert isinstance(chunks[0], Document)

    # Check metadata correctness
    assert "complaint_id" in chunks[0].metadata
    assert "product" in chunks[0].metadata
    assert "original_narrative" in chunks[0].metadata

    # Check chunk content is string
    assert isinstance(chunks[0].page_content, str)


def test_embed_and_store(tmp_path, dummy_dataframe):
    # Create chunks
    chunks = cvs.chunk_texts(dummy_dataframe, chunk_size=20, chunk_overlap=5)

    # Define a temp directory
    save_dir = tmp_path / "vector_store"

    # Run embedding + store
    cvs.embed_and_store(
        docs=chunks,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        save_dir=str(save_dir),
    )

    # Check files exist
    index_path = save_dir / "index.faiss"
    store_path = save_dir / "index.pkl"

    assert index_path.exists(), "FAISS index file not created."
    assert store_path.exists(), "Metadata pickle file not created."

    # Attempt to load vector store
    embeddings_model = cvs.HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        folder_path=str(save_dir),
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True,
    )

    # Try searching the vector store
    results = vector_store.similarity_search("credit card", k=1)
    assert len(results) >= 1
    assert isinstance(results[0].page_content, str)
