import os
from openai import OpenAI
from utils import pack_context, get_logger


HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")

logger = get_logger("generator")


SYSTEM_PROMPT = """You are a research assistant.

Answer the question using ONLY the provided context.
If the answer is not present in the context, say:
"The answer is not available in the provided documents."

Cite chunk numbers in your answer."""


class Generator:
    def __init__(self):
        logger.info(f"Initializing HF Inference client (model: {MODEL})")
        self.client = OpenAI(
            api_key=HF_TOKEN,
            base_url="https://router.huggingface.co/v1",
        )
        logger.info("HF Inference client ready")

    def generate(self, query, retrieved_chunks, max_context_tokens=1600):
        context_blocks, num_packed = pack_context(
            retrieved_chunks, max_tokens=max_context_tokens
        )
        logger.info(f"Packed {num_packed}/{len(retrieved_chunks)} chunks into prompt")

        user_message = f"""Context:
{context_blocks}

Question:
{query}"""

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            temperature=0.1,
        )

        return response.choices[0].message.content.strip()