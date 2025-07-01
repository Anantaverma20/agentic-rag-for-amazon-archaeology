import fitz  # PyMuPDF
import os
import re
import pickle
import numpy as np
import faiss

from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ---------- Step 1: Extract Text from All PDFs ----------
def extract_text_from_pdfs(pdf_paths):
    all_chunks = []
    all_sources = []

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator=".")

    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        # Clean text
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        chunks = splitter.split_text(cleaned_text)
        all_chunks.extend(chunks)
        all_sources.extend([os.path.basename(pdf_path)] * len(chunks))

    return all_chunks, all_sources

# ---------- Step 2: Generate Embeddings ----------
def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# ---------- Step 3: Store in FAISS + Save ----------
def store_to_faiss(chunks, sources, embeddings, faiss_path="text_index.faiss", meta_path="text_metadata.pkl"):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, faiss_path)

    metadata = {"texts": chunks, "sources": sources}
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Saved FAISS index to: {faiss_path}")
    print(f"✅ Saved metadata to: {meta_path}")

# ---------- Main ----------
if __name__ == "__main__":
    pdf_files = [
        "Data/textual_data/Exploration_of_the_Valley_of_the_Amazon.pdf",
        "Data/textual_data/book.pdf",
        "Data/textual_data/narrativeoftrave00walluoft.pdf",
        "Data/textual_data/public-gdcmassbookdig-voyageupriverama00edwa_0-voyageupriverama00edwa_0.pdf"
    ]

    print("📘 Extracting text...")
    chunks, sources = extract_text_from_pdfs(pdf_files)

    print("🔎 Embedding text...")
    embeddings = embed_chunks(chunks)

    print("💾 Saving to FAISS...")
    store_to_faiss(chunks, sources, embeddings)

    # Save text chunks for downstream use
    with open("text_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("✅ Saved text chunks to: text_chunks.pkl")
