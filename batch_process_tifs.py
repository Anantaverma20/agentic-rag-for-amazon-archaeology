import os
import rasterio
import numpy as np
from rasterio.windows import Window
from PIL import Image

# GPU acceleration imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 GPU acceleration enabled with CuPy")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  CuPy not available, using CPU. Install with: pip install cupy-cuda11x")

# ---------- CONFIG ----------
input_folder = "Data/satellite_elevation/"
output_folder = "tiles"
tile_size = 512  # pixels

os.makedirs(output_folder, exist_ok=True)

# ---------- GPU UTILITY FUNCTIONS ----------
def to_gpu(array):
    """Move numpy array to GPU if available"""
    if GPU_AVAILABLE:
        return cp.asarray(array)
    return array

def to_cpu(array):
    """Move array back to CPU"""
    if GPU_AVAILABLE and hasattr(array, 'get'):
        return cp.asnumpy(array)
    return array

def gpu_normalize(array):
    """Normalize array using GPU if available"""
    if GPU_AVAILABLE:
        array_gpu = cp.asarray(array)
        min_val = cp.min(array_gpu)
        max_val = cp.max(array_gpu)
        normalized = (array_gpu - min_val) / (max_val - min_val + 1e-5)
        return cp.asnumpy(normalized)
    else:
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val + 1e-5)

# ---------- TILE PROCESSING ----------
def process_tif(file_path, output_base):
    print(f"📂 Processing {os.path.basename(file_path)}")

    with rasterio.open(file_path) as src:
        width = src.width
        height = src.height
        # Remove the line that loads entire image into memory
        # band = src.read(1)  # This was causing the 22.7 GB memory error

        print(f"📏 Image size: {width} x {height} pixels")
        print(f"💾 Estimated memory needed: {width * height / 1e9:.2f} GB")

        count = 0
        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                window = Window(col, row, tile_size, tile_size)

                try:
                    # Read only the tile window - much more memory efficient
                    tile = src.read(1, window=window)

                    # Skip if mostly empty or zeroed
                    if np.all(tile == 0):
                        continue

                    # GPU-accelerated normalization
                    norm_tile = gpu_normalize(tile)
                    
                    # Convert to uint8 for image saving
                    img_array = (norm_tile * 255).astype(np.uint8)
                    img = Image.fromarray(img_array)

                    tile_filename = f"{output_folder}/{output_base}_tile_{count}.png"
                    img.save(tile_filename)
                    count += 1
                    
                    # Progress indicator for large files
                    if count % 100 == 0:
                        print(f"  📊 Processed {count} tiles...")
                        
                except Exception as e:
                    print(f"❌ Skipped tile at {col}, {row}: {e}")

        print(f"✅ Saved {count} tiles for {output_base}")


# ---------- MAIN LOOP ----------
print(f"🔧 Processing mode: {'GPU' if GPU_AVAILABLE else 'CPU'}")
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        full_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        process_tif(full_path, base_name)
