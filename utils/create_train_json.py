import json
import pandas as pd
from tqdm import tqdm

csv_file = "/path/to/morph_trait_annot_bioscan_80k_kaggle/dataset.csv"

df = pd.read_csv(csv_file)

samples = []
for _, row in tqdm(df.iterrows()):
    sample = {
        "image": "/path/to/morph_trait_annot_bioscan_80k_kaggle/" + row["image"],
        "text": "A photo of " + row["species"] + " with " + row["trait_description"]
    }
    samples.append(sample)

# Save to JSON file
output_json_file = "/path/to/morph_trait_annot_bioscan_80k_kaggle/trait_dataset.json"

json.dump(samples, open(output_json_file, "w"), indent=4)
