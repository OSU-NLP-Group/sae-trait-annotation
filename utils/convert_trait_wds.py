import os
import json
import webdataset as wds
from PIL import Image

# --- Config ---
json_path    = "/path/to/morph_trait_annot_bioscan_80k_kaggle/trait_dataset.json"

img_dir      = "/"         # folder with img001.jpg, img002.jpg, …
shard_pattern = "clip_data/traits-%06d.tar"   # output shards, e.g. traits-000000.tar, traits-000001.tar
max_size = 1e8

# --- Load JSON ---
with open(json_path, "r") as f:
    data = json.load(f)

# --- Write shards ---
writer = wds.ShardWriter(shard_pattern, maxsize=max_size)

for idx, entry in enumerate(data):
    img_path = os.path.join(img_dir, entry["image"])

    img = Image.open(img_path)

    # encode text
    # txt_bytes = entry["text"].encode("utf-8")
    txt = entry["text"]

    sample = {
        "__key__": f"sample{idx:08d}",  # unique key per sample
        "jpg": img,         # image bytes stored under .jpg suffix
        "txt": txt,         # text bytes under .txt suffix
    }
    writer.write(sample)

writer.close()
print("Finished writing WebDataset shards.")
