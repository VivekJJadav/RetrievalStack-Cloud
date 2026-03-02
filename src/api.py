from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI()

# Fully lazy-load: don't even import retriever/generator at module level.
# Importing them triggers sentence_transformers → torch loading, which is
# too slow on Render free tier (0.1 CPU) and causes port scan timeout.
_retriever = None
_generator = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from retriever import Retriever
        _retriever = Retriever()
    return _retriever


def get_generator():
    global _generator
    if _generator is None:
        from generator import Generator
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