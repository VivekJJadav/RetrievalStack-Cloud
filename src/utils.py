"""
utils.py — Shared utilities for the RL Research Paper Assistant.

Provides:
  - Token estimation
  - Text tokenization (stopword filtering, stemming)
  - Context packing for LLM prompts
  - Logging helpers
"""

import re
import math
import logging
import sys
from datetime import datetime


# ──────────────────────────────────────────────
# Logging Helpers
# ──────────────────────────────────────────────

def get_logger(name, level=logging.INFO):
    """Create a formatted logger for a module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_timing(logger, label):
    """Simple context manager to log elapsed time for a block."""
    class _Timer:
        def __enter__(self):
            self.start = datetime.now()
            return self
        def __exit__(self, *args):
            elapsed = (datetime.now() - self.start).total_seconds()
            logger.info(f"{label} completed in {elapsed:.2f}s")
    return _Timer()


# ──────────────────────────────────────────────
# Token Estimation
# ──────────────────────────────────────────────

def estimate_tokens(text):
    """Rough token count approximation (~1.3 tokens per whitespace word)."""
    return int(len(text.split()) * 1.3)


# ──────────────────────────────────────────────
# Stopword Filtering & Stemming
# ──────────────────────────────────────────────

STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about",
    "up", "down", "it", "its", "he", "she", "they", "them", "we", "you",
    "i", "me", "my", "your", "his", "her", "our", "their", "this", "that",
    "these", "those", "what", "which", "who", "whom"
})


def porter_stem(word):
    """Minimal Porter stemmer for common English suffixes."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("tion"):
        return word[:-4]
    if word.endswith("ment") and len(word) > 5:
        return word[:-4]
    if word.endswith("ness") and len(word) > 5:
        return word[:-4]
    if word.endswith("ous") and len(word) > 4:
        return word[:-3]
    if word.endswith("ive") and len(word) > 4:
        return word[:-3]
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("er") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def tokenize(text):
    """Lowercase, extract words, remove stopwords, and stem.

    Returns a set of stemmed tokens for efficient lookup.
    """
    words = re.findall(r'[a-z][a-z0-9]+', text.lower())
    return set(porter_stem(w) for w in words if w not in STOPWORDS)


def filter_stopwords(text):
    """Remove stopwords from text, returning cleaned string."""
    words = text.lower().split()
    return " ".join(w for w in words if w not in STOPWORDS)


# ──────────────────────────────────────────────
# Context Packing
# ──────────────────────────────────────────────

def pack_context(chunks, max_tokens=1600):
    """Pack retrieved chunks into a context string within a token budget.

    Args:
        chunks: list of dicts with 'source' and 'text' keys.
        max_tokens: maximum estimated tokens allowed.

    Returns:
        context_str: formatted context string.
        num_packed: number of chunks that fit.
    """
    context_blocks = ""
    used_tokens = 0
    num_packed = 0

    for i, chunk in enumerate(chunks):
        block = f"\n[Chunk {i+1} | Source: {chunk['source']}]\n{chunk['text']}\n"
        block_tokens = estimate_tokens(block)

        if used_tokens + block_tokens > max_tokens:
            break

        context_blocks += block
        used_tokens += block_tokens
        num_packed += 1

    return context_blocks, num_packed
