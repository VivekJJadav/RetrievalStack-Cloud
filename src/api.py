from fastapi import FastAPI
from pydantic import BaseModel
from retriever import Retriever
from generator import Generator
import time

app = FastAPI()

retriever = Retriever()
generator = Generator()


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QueryRequest):
    start = time.perf_counter()

    chunks = retriever.search(request.query)
    answer = generator.generate(request.query, chunks)

    latency = time.perf_counter() - start

    return {
        "answer": answer,
        "latency_seconds": round(latency, 3)
    }