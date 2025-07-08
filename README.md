# 💬 Financial Complaints RAG Chatbot

**Financial Complaints RAG Chatbot** is an AI-powered assistant that answers questions about financial customer complaints, based on real complaint narratives.

- Retrieves relevant complaint excerpts from a FAISS vector store
- Uses sentence-transformers embeddings
- Generates answers with an LLM (Flan-T5 Small)
- Displays sources for transparency
- Runs via either Streamlit or Gradio interfaces

---

## 🚀 Features

✅ Retrieval-Augmented Generation (RAG) pipeline
✅ Fast similarity search with FAISS
✅ Supports CPU and GPU execution
✅ Works in light or dark mode
✅ Looks and feels like ChatGPT
✅ Easy deployment via:
- Streamlit UI
- Gradio ChatInterface

---

## 🛠️ Installation

1. **Clone the repo**

```bash
git clone https://github.com/YOUR_USERNAME/rag-fincomplaints.git
cd rag-fincomplaints
```

2. **Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

> 💡 If running on GPU, ensure your PyTorch and CUDA versions are compatible.

---

## 🏗️ Vector Store

Your vector store lives under:

```
vector_store/faiss_index
```

- Ensure this folder exists and contains the saved FAISS index.
- If you need to build your index, modify `query_rag_pipeline.py` to index your complaint data.

---

## 💻 Running the App

### Option 1 — Streamlit

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the provided URL (e.g. http://localhost:8501).

---

### Option 2 — Gradio

Start the Gradio app:

```bash
python3 app_gradio.py
```

Then visit [http://localhost:7860](http://localhost:7860)

> You’ll see a UI like this:
>
> *(Replace this with your real screenshot filename in your repo!)*

---

## 🤖 Example Gradio Chatbot Usage

```python
import requests

# Example local Gradio call:
url = "http://127.0.0.1:7860/run/predict"
payload = {
  "data": ["What issues occur with BNPL products?", []]
}

r = requests.post(url, json=payload)
print(r.json())
```

---

## 🧩 Project Structure

```
├── app.py                  # Streamlit app
├── app_gradio.py           # Gradio app
├── src/
│   └── rag/
│       └── query_rag_pipeline.py
├── vector_store/
│    └── faiss_index/       # Saved FAISS index
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

- Python 3.12+
- LangChain
- sentence-transformers
- HuggingFace Transformers
- FAISS
- Streamlit
- Gradio

---

## ⚠️ Troubleshooting

### CUDA errors

If you see errors like:

```
RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

- Ensure your GPU drivers and CUDA versions match PyTorch
- Run in CPU mode by forcing:

```python
device = -1
```

in `transformers.pipeline(...)`

Or set:

```python
model_kwargs={"device": "cpu"}
```

when loading HuggingFace embeddings.

---

## 📝 License

MIT License

---

## 🙌 Credits

This chatbot was developed as part of a financial complaints analysis project for 10 Academy.

---

## ⭐️ Future Improvements

- Switch to newer `langchain-huggingface` import
- Add streaming token-by-token output
- UI polish to perfectly mimic ChatGPT (light & dark themes)
- Cloud deployment (e.g. Hugging Face Spaces)
