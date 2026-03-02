import os
import re
import fitz  # PyMuPDF
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


DATA_DIR = "../data/papers"
INDEX_PATH = "../models/faiss_index.bin"
META_PATH = "../models/chunk_metadata.pkl"

CHUNK_SIZE = 800   # characters
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 100  # filter out tiny fragments


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text




# Section headings that mark the start of references
_REF_HEADINGS = re.compile(
    r'^(references|bibliography|works cited|literature cited)\s*$',
    re.IGNORECASE
)


def _strip_references_section(text):
    """Remove everything after the References/Bibliography heading."""
    lines = text.split('\n')
    cut_index = None
    # Search from the end (references are usually at the bottom)
    for i in range(len(lines) - 1, -1, -1):
        if _REF_HEADINGS.match(lines[i].strip()):
            cut_index = i
            break
    if cut_index is not None:
        text = '\n'.join(lines[:cut_index])
    return text


def _is_reference_line(text):
    """Check if text looks like a bibliography/reference entry."""
    text = text.strip()
    # [1] Author..., [19] M. Hessel...
    if re.match(r'^\[\d+\]', text):
        return True
    # Numbered refs like "1. Author, ..." or "23. Author, ..."
    if re.match(r'^\d{1,3}\.\s+[A-Z]', text):
        return True
    return False


def _clean_pdf_text(text):
    """Clean up common PDF extraction artifacts."""
    # Strip references/bibliography section first
    text = _strip_references_section(text)
    # Merge hyphenated line breaks (e.g., "rein-\nforcement" -> "reinforcement")
    text = re.sub(r'-\n', '', text)
    # Replace single newlines (within a paragraph) with spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    return text


def chunk_text(text, max_length=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
               min_length=MIN_CHUNK_LENGTH):
    """Split text into semantic chunks by merging small paragraphs."""
    text = _clean_pdf_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Filter out reference entries and very short noise
    filtered = []
    for para in paragraphs:
        if _is_reference_line(para):
            continue
        if len(para) < 30:  # skip tiny fragments like page numbers, headers
            continue
        filtered.append(para)

    # Merge small paragraphs into larger chunks up to max_length
    chunks = []
    current = ""

    for para in filtered:
        # If adding this paragraph stays under limit, merge
        if len(current) + len(para) + 1 <= max_length:
            current = (current + "\n\n" + para).strip()
        else:
            # Save current chunk if big enough
            if len(current) >= min_length:
                chunks.append(current)

            # If the paragraph itself is too long, split with sliding window
            if len(para) > max_length:
                start = 0
                while start < len(para):
                    end = start + max_length
                    piece = para[start:end]
                    if len(piece) >= min_length:
                        chunks.append(piece)
                    start = end - overlap
                current = ""
            else:
                current = para

    # Don't forget the last chunk
    if len(current) >= min_length:
        chunks.append(current)

    return chunks


def main():
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = []
    metadata = []

    print("Reading PDFs...")
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, filename)
            text = extract_text_from_pdf(pdf_path)

            chunks = chunk_text(text)

            for chunk in chunks:
                all_chunks.append(chunk)
                metadata.append({
                    "source": filename,
                    "text": chunk
                })

    print(f"Total chunks: {len(all_chunks)}")

    print("Generating embeddings...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("Saving index...")
    os.makedirs("../models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("Done!")


if __name__ == "__main__":
    main()