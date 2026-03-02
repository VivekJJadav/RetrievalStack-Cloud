from llama_cpp import Llama
from utils import pack_context, get_logger


MODEL_PATH = "../models/tinyllama.gguf"

logger = get_logger("generator")


class Generator:
    def __init__(self):
        logger.info("Loading LLM...")
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=6,          # adjust if needed
            n_gpu_layers=20       # Metal acceleration on M1
        )
        logger.info("LLM ready")

    def build_prompt(self, query, retrieved_chunks, max_context_tokens=1600):
        context_blocks, num_packed = pack_context(
            retrieved_chunks, max_tokens=max_context_tokens
        )
        logger.info(f"Packed {num_packed}/{len(retrieved_chunks)} chunks into prompt")

        prompt = f"""
You are a research assistant.

Answer the question using ONLY the provided context.
If the answer is not present in the context, say:
"The answer is not available in the provided documents."

Cite chunk numbers in your answer.

Context:
{context_blocks}

Question:
{query}

Answer:
"""
        return prompt.strip()

    def generate(self, query, retrieved_chunks):
        prompt = self.build_prompt(query, retrieved_chunks)

        output = self.llm(
            prompt,
            max_tokens=300,
            temperature=0.1,
            stop=["Question:"]
        )

        return output["choices"][0]["text"].strip()