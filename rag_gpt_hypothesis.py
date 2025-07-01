import faiss
import pickle
import torch
import openai
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from dotenv import load_dotenv
import os

# ---------- CONFIG ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = openai.OpenAI(api_key=OPENAI_API_KEY)

device = "cuda" if torch.cuda.is_available() else "cpu"
k = 3  # top-k image and text results

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS + metadata
clip_index = faiss.read_index("clip_tile_index.faiss")
text_index = faiss.read_index("text_index.faiss")
with open("clip_tile_metadata.pkl", "rb") as f:
    tile_filenames = pickle.load(f)
with open("text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)

def generate_hypothesis(user_query):
    # Spell-correct the query
    corrected_query = str(TextBlob(user_query).correct())

    # ----- Get CLIP image search embedding -----
    inputs = clip_processor(text=corrected_query, return_tensors="pt").to(device)
    with torch.no_grad():
        image_query_vector = clip_model.get_text_features(**inputs).cpu().numpy().astype("float32")

    # ----- Get SentenceTransformer embedding for text search -----
    text_query_vector = text_model.encode([corrected_query]).astype("float32")

    # ----- Perform FAISS search -----
    _, image_indices = clip_index.search(image_query_vector, k)
    _, text_indices = text_index.search(text_query_vector, k)

    top_tiles = [tile_filenames[i] for i in image_indices[0]]
    top_texts = [text_chunks[i] for i in text_indices[0] if i < len(text_chunks)]

    # ----- Build GPT prompt -----
    image_section = "\n".join([f"{i+1}. {fname}" for i, fname in enumerate(top_tiles)])
    text_section = "\n".join([f"{i+1}. \"{txt.strip()[:400]}...\"" for i, txt in enumerate(top_texts)])

    prompt = f"""
You are an AI archaeological assistant.
You are given:
- Satellite elevation image tiles (filenames)
- Historical text excerpts from Amazon expedition logs

Use the information to:
- Hypothesize whether any image tiles show signs of ancient human settlements (e.g., geometric mounds, clearings, roads)
- Justify your reasoning using the elevation patterns AND the historical descriptions

---
User Query: "{corrected_query}"

🖼️ Image Tiles:
{image_section}

📚 Historical Text Chunks:
{text_section}

---
Please provide:
- A short hypothesis about potential ancient occupation
- Justification combining image and text information
"""

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in archaeological analysis and Amazon rainforest studies."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content, top_tiles, top_texts

# if __name__ == "__main__":
#     user_query = input("Enter your archaeological query: ")
#     hypothesis, top_tiles, top_texts = generate_hypothesis(user_query)
#     print("\n--- GPT Hypothesis ---\n")
#     print(hypothesis)
#     print("\nTop Image Tiles:")
#     for fname in top_tiles:
#         print(f"  - {fname}")
#     print("\nTop Text Chunks:")
#     for i, txt in enumerate(top_texts):
#         print(f"  {i+1}. {txt[:300]}...")
