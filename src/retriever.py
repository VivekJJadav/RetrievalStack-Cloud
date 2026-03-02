import faiss
import pickle
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import tokenize, get_logger, log_timing


INDEX_PATH = "../models/faiss_index.bin"
META_PATH = "../models/chunk_metadata.pkl"

logger = get_logger("retriever")


class Retriever:
    def __init__(self):
        logger.info("Loading FAISS index...")
        self.index = faiss.read_index(INDEX_PATH)

        logger.info("Loading metadata...")
        with open(META_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Pre-compute document frequencies for IDF weighting
        with log_timing(logger, "Term statistics"):
            self.num_docs = len(self.metadata)
            self.doc_freq = {}
            for entry in self.metadata:
                tokens = tokenize(entry["text"])
                for tok in tokens:
                    self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1

        logger.info(f"Ready — {self.num_docs} chunks indexed")

    def _idf(self, term):
        """Inverse document frequency — rare terms get higher weight."""
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(self.num_docs / (1 + df))

    def rerank(self, query, candidates):
        query_tokens = tokenize(query)

        # Pre-compute IDF weights for query terms
        idf_weights = {t: self._idf(t) for t in query_tokens}
        total_idf = sum(idf_weights.values()) or 1.0

        scored = []

        for chunk, dist in candidates:
            chunk_tokens = tokenize(chunk["text"])

            # Weighted keyword score: rare terms count more
            keyword_score = sum(
                idf_weights[t] for t in query_tokens if t in chunk_tokens
            ) / total_idf

            similarity = 1 / (1 + dist)

            score = 0.6 * similarity + 0.4 * keyword_score

            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [chunk for score, chunk in scored]

    def search(self, query, top_k=15, final_k=5):
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            candidates.append((self.metadata[idx], dist))

        reranked = self.rerank(query, candidates)

        return reranked[:final_k]