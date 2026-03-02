from fastapi import FastAPI
from pydantic import BaseModel
from retriever import Retriever
from generator import Generator
import time

app = FastAPI()

# Lazy-load: models load on first request, not at startup
# This prevents Render's port scan timeout on free tier
_retriever = None
_generator = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_generator():
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QueryRequest):
    start = time.perf_counter()

    retriever = get_retriever()
    generator = get_generator()

    chunks = retriever.search(request.query)
    answer = generator.generate(request.query, chunks)

    latency = time.perf_counter() - start

    return {
        "answer": answer,
        "latency_seconds": round(latency, 3)
    }