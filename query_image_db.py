import faiss
import pickle
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from textblob import TextBlob

# ---------- CONFIG ----------
index_path = "clip_tile_index.faiss"
meta_path = "clip_tile_metadata.pkl"
tile_folder = "tiles"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load CLIP Model ----------
print(f"🔁 Loading CLIP on {device}...")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# ---------- Load Vector DB ----------
index = faiss.read_index(index_path)
with open(meta_path, "rb") as f:
    filenames = pickle.load(f)

# ---------- Get and Correct Query ----------
original_query = input("🔎 Enter your visual search query: ")
corrected_query = str(TextBlob(original_query).correct())
print(f"✅ Spell-corrected query: '{corrected_query}'")

# ---------- Encode and Search ----------
inputs = processor(text=corrected_query, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    query_embedding = model.get_text_features(**inputs).cpu().numpy()

k = 5  # number of top results
_, indices = index.search(query_embedding, k)

# ---------- Show Results ----------
print(f"\n🎯 Top {k} image tiles for: '{corrected_query}'\n")
for i in indices[0]:
    fname = filenames[i]
    print(f"🖼️ {fname}")
    image_path = f"{tile_folder}/{fname}"
    
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(fname)
    plt.axis("off")
    plt.show()
