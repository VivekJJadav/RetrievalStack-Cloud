# 📚 RL Research Paper Assistant

A **Retrieval-Augmented Generation (RAG)** system that lets you ask questions about Reinforcement Learning research papers and get grounded, cited answers — powered by a local LLM running entirely on your machine.

## ✨ Features

- **PDF Ingestion** — Extracts and chunks text from RL papers with smart paragraph merging, reference filtering, and PDF artifact cleanup
- **Semantic Search** — FAISS vector index with sentence-transformer embeddings for fast retrieval
- **IDF-Weighted Reranking** — Two-stage retrieval: FAISS top-k → keyword reranking with stopword removal, Porter stemming, and rare-term boosting
- **Local LLM Generation** — TinyLlama 1.1B (GGUF Q4) via `llama-cpp-python` with Metal GPU acceleration on Apple Silicon
- **REST API** — FastAPI server with `/ask` and `/health` endpoints
- **Dockerized** — Ready to containerize for deployment

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────┐     ┌───────────┐     ┌───────────┐
│ FastAPI   │────▶│ Retriever │────▶│ Generator │
│ (api.py)  │     │           │     │           │
└──────────┘     │ FAISS     │     │ TinyLlama │
                 │ + Rerank  │     │ (GGUF)    │
                 └───────────┘     └───────────┘
                       │                 │
                       ▼                 ▼
                 ┌───────────┐     ┌──────────┐
                 │ Embeddings│     │  Answer   │
                 │ Index     │     │  + Cites  │
                 └───────────┘     └──────────┘
```

## 📁 Project Structure

```
RL Research Paper Assistant/
├── data/papers/           # PDF research papers (19 RL papers)
├── models/
│   ├── tinyllama.gguf     # TinyLlama 1.1B Q4 model (~608MB)
│   ├── faiss_index.bin    # FAISS vector index
│   └── chunk_metadata.pkl # Chunk text + source metadata
├── src/
│   ├── ingest.py          # PDF → chunks → embeddings → FAISS index
│   ├── retriever.py       # Semantic search + IDF reranking
│   ├── generator.py       # LLM prompt building + generation
│   ├── utils.py           # Shared utilities (tokenization, stemming, logging)
│   ├── api.py             # FastAPI REST endpoints
│   └── test.py            # Quick test script
├── requirements.txt
├── dockerfile
├── .dockerignore
└── .gitignore
```

## 🚀 Setup

### Prerequisites
- Python 3.10+
- ~1GB free disk space (for model + index)

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download LLM Model
```bash
curl -L -o models/tinyllama.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
```

### 4. Add Research Papers
Place your PDF papers in `data/papers/`.

### 5. Build the Index
```bash
cd src
python ingest.py
```

### 6. Run the API
```bash
cd src
uvicorn api:app --reload
```

## 📡 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Proximal Policy Optimization?"}'
```

**Response:**
```json
{
  "answer": "Proximal Policy Optimization (PPO) is a family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a 'surrogate' objective function...",
  "latency_seconds": 6.374
}
```

## 🐳 Docker

```bash
docker build -t rl-rag .
docker run -p 8000:8000 rl-rag
```

> **Note:** Metal GPU acceleration is not available inside Docker (Linux VM). The LLM will run CPU-only, which is slower but functional.

## 🔧 Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Embeddings | `all-MiniLM-L6-v2` | Fast, lightweight, good quality |
| Vector DB | FAISS (IndexFlatL2) | Simple, no server needed |
| LLM | TinyLlama 1.1B Q4 | Runs locally, no API keys |
| Chunking | Paragraph-merge + sliding window | Semantic coherence vs. fixed-size |
| Reranking | IDF-weighted keyword + vector similarity | Better precision than vector-only |

## 📄 Included Papers

The system comes pre-configured with 19 foundational RL papers including:
- **PPO** — Proximal Policy Optimization (Schulman et al., 2017)
- **TRPO** — Trust Region Policy Optimization (Schulman et al., 2015)
- **DDPG** — Deep Deterministic Policy Gradient (Lillicrap et al., 2015)
- **A3C** — Asynchronous Advantage Actor-Critic (Mnih et al., 2016)
- **SAC** — Soft Actor-Critic (Haarnoja et al., 2018)
- **AlphaGo** — Mastering Go with Neural Networks (Silver et al., 2016)
- **DQN** — Playing Atari with Deep RL (Mnih et al., 2013)
- And more...

## 📝 License

This project is for educational and portfolio purposes.
