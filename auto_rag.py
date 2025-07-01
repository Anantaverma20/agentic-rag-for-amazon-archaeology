import faiss
import pickle
import torch
import openai
import folium
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from dotenv import load_dotenv
import os
from IPython.display import display, Image as IPImage

# ---------- SETUP ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
device = "cuda" if torch.cuda.is_available() else "cpu"
k = 3  # top-k results

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
with open("tile_bounds.pkl", "rb") as f:
    tile_bounds = pickle.load(f)  # dict of {tile_filename: [south, west, north, east]}

# ---------- FUNCTION: Generate GPT hypothesis and map ----------
def run_query_and_generate_map(user_query):
    corrected_query = str(TextBlob(user_query).correct())

    # Embed query for CLIP (image) and SentenceTransformer (text)
    clip_input = clip_processor(text=corrected_query, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_query_vec = clip_model.get_text_features(**clip_input).cpu().numpy().astype("float32")
    text_query_vec = text_model.encode([corrected_query]).astype("float32")

    # FAISS search
    _, img_idx = clip_index.search(clip_query_vec, k)
    _, txt_idx = text_index.search(text_query_vec, k)

    top_tiles = [tile_filenames[i] for i in img_idx[0]]
    top_texts = [text_chunks[i] for i in txt_idx[0] if i < len(text_chunks)]

    # Build GPT prompt
    img_section = "\n".join([f"{i+1}. {f}" for i, f in enumerate(top_tiles)])
    txt_section = "\n".join([f"{i+1}. \"{txt.strip()[:400]}...\"" for i, txt in enumerate(top_texts)])
    prompt = f"""

    
You are an AI archaeological assistant.
User query: "{corrected_query}"

🖼️ Image Tiles:
{img_section}

📚 Text Chunks:
{txt_section}

Use this to:
- Hypothesize potential ancient human activity
- Justify using both image patterns and historical text
"""

    # Call GPT
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in archaeological analysis and Amazon rainforest studies."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    hypothesis = response.choices[0].message.content

     # Create interactive Folium map
    m = folium.Map(location=[-3.0, -54.9], zoom_start=8, tiles="CartoDB positron")
    missing = []

    for tile in top_tiles:
        bounds = tile_bounds.get(tile)
        if bounds:
            south, west, north, east = bounds
            popup_text = f"<b>{tile}</b><br><b>Hypothesis:</b><br>{hypothesis}"
            folium.Rectangle(
                bounds=[[south, west], [north, east]],
                color="blue",
                fill=True,
                fill_opacity=0.3,
                popup=folium.Popup(popup_text, max_width=400)
            ).add_to(m)
        else:
            missing.append(tile)

    if missing:
        print(f"⚠️ WARNING: No bounds found for these tiles: {missing}")

    os.makedirs("outputs", exist_ok=True)
    map_filename = f"outputs/map_{user_query.replace(' ', '_')}.html"
    m.save(map_filename)
    print(f"✅ Map saved to: {map_filename}")

    # Display top image tiles inline
    print("\n🖼️ Displaying Top Image Tiles:")
    for tile in top_tiles:
        tile_path = os.path.join("tiles", tile)
        if os.path.exists(tile_path):
            display(IPImage(filename=tile_path))
        else:
            print(f"⚠️ Tile not found: {tile}")

    return hypothesis

if __name__ == "__main__":
    print("🟢 Running auto_rag.py")
    result = run_query_and_generate_map("rectangular earthworks near Madeira river")
    print("\n✅ Final Hypothesis:\n")
    print(result)
