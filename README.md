# Agentic RAG for Amazon Archaeology

This project presents an **Agentic Multimodal Retrieval-Augmented Generation (RAG)** system designed to explore and hypothesize about potential archaeological sites in the Amazon basin. It intelligently combines **historical texts**, **satellite imagery**, and **geospatial analysis** to identify and interpret anthropogenic features like geoglyphs, mound structures, and pre-Columbian settlements.

---

## Project Overview

The system uses:

* **CLIP-based image tile retrieval** from deforestation raster data
* **FAISS-powered text search** from scanned 19th-century Amazon expedition books
* **GPT-4-based hypothesis agent** for interpreting historical and spatial data
* **Folium-based interactive maps** with optional shapefile overlays
* A **Geographic Expansion Agent** that proposes alternative regions of interest based on learned patterns

---

## Repository Contents

| File                                       | Description                                                                  |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| `agentic-rag-for-amazon-archaeology.ipynb` | Main Kaggle notebook orchestrating the full multimodal RAG pipeline          |
| `auto_rag.py`                              | Driver script for executing the full agentic RAG workflow                    |
| `batch_process_tifs.py`                    | Preprocessing utility for converting and tiling large `.tif` satellite files |
| `generate_gpt_prompt.py`                   | Builds structured prompts for hypothesis generation using LLMs               |
| `generate_tile_bounds.py`                  | Extracts and saves lat/lon bounds of image tiles                             |
| `query_image_db.py`                        | Queries image tile database using CLIP embeddings                            |
| `rag_gpt_hypothesis.py`                    | Main function for generating hypotheses from retrieved data                  |
| `shapefile_overlay.png`                    | Example visualization showing shapefile overlays on satellite data           |
| `text_data.py`                             | Manages loading and parsing of scanned historical texts                      |
| `text_chunks.pkl`                          | Precomputed semantic text chunks from historical books                       |
| `text_index.faiss`                         | FAISS index of historical text embeddings                                    |
| `text_metadata.pkl`                        | Metadata for the text chunks (e.g., page number, book title)                 |
| `tile_bounds.pkl`                          | Bounding box coordinates for image tiles                                     |

---

## Data Sources

* **Textual Data**: 4 historical books from the 19th century including:

  * *Exploration of the Valley of the Amazon* (Herndon & Gibbon)
  * *Narrative of Travels on the Amazon and Rio Negro* (Wallace)
  * *A Voyage up the River Amazon* (Edwards)
* **Satellite Data**: Processed tiles from Brazil's PRODES 2023 deforestation dataset
* **Shapefiles**: DETER alerts from Brazil’s INPE for recent forest changes

---

## Features

* Agentic workflow with multi-agent prompt orchestration
* Clickable map thumbnails with hypothesis-linked tiles
* Option to overlay shapefiles (LIDAR, fire scars, alerts)
* Evaluation via ROUGE to validate generated hypotheses

---

## How to Run

1. Set up your OpenAI API Key (`OPENAI_API_KEY`)
2. Ensure your tile/image/text FAISS indices are stored in:

   ```
   /kaggle/input/
   ├── tiles/
   ├── embeddings/
   ├── textual-data/
   └── tile-bounds/
   ```
3. Run the main notebook or `auto_rag.py` in a Kaggle environment

---

## Evaluation

Includes a ROUGE-based comparison between generated hypotheses and source documents to assess relevance and fidelity.

---

## Acknowledgments

* Built for the **OpenAI to Z Challenge** on Kaggle
* Inspired by historical explorations and recent advances in geospatial AI
