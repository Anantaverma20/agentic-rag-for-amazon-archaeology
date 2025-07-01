import os
import rasterio
from rasterio.windows import Window
import pickle
from multiprocessing import Pool, cpu_count

# ---------- CONFIG ----------
tif_folder = "Data/satellite_elevation"  # folder with original .tif files
tile_dir = "tiles"  # folder with actual .png tiles
output_bounds_file = "tile_bounds.pkl"
tile_size = 512

# Map from base filename to its raster path
tif_map = {os.path.splitext(f)[0]: os.path.join(tif_folder, f)
           for f in os.listdir(tif_folder) if f.endswith(".tif")}

# Load all tile names
tile_names = [f for f in os.listdir(tile_dir) if f.endswith(".png")]

def compute_tile_bound(tile_name):
    try:
        parts = tile_name.replace(".png", "").split("_tile_")
        base_name, index = parts[0], int(parts[1])

        tif_path = tif_map.get(base_name)
        if not tif_path:
            return None

        with rasterio.open(tif_path) as src:
            width_in_tiles = src.width // tile_size
            row = index // width_in_tiles
            col = index % width_in_tiles

            window = Window(col * tile_size, row * tile_size, tile_size, tile_size)
            bounds = rasterio.windows.bounds(window, src.transform)
            west, south, east, north = bounds
            return (tile_name, [south, west, north, east])

    except Exception as e:
        print(f"❌ Failed: {tile_name} — {e}")
        return None

# ---------- MULTIPROCESSING ----------
if __name__ == "__main__":
    print(f"🚀 Using {cpu_count()} CPU cores for parallel processing...")
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compute_tile_bound, tile_names)

    tile_bounds = {name: bounds for name, bounds in results if name and bounds}

    with open(output_bounds_file, "wb") as f:
        pickle.dump(tile_bounds, f)

    print(f"✅ Recreated tile_bounds.pkl with {len(tile_bounds)} entries.")


