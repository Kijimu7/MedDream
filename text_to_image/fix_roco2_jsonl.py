# fix_roco2_jsonl.py
import json
from pathlib import Path

# 1) Path to your old JSONL inside train_images/
old = Path("/home/jovyan/datafabric/ROCOv2_Anatomical_Prompts/train_images/roco2_train_for_diffusers.jsonl")

# 2) Path for the new, corrected JSONL in the same train_images/ folder
new = Path("/home/jovyan/datafabric/ROCOv2_Anatomical_Prompts/train_images/train_data_for_diffusers.jsonl")

with old.open("r") as fin, new.open("w") as fout:
    for line in fin:
        rec = json.loads(line)
        # rec["text"] is the caption
        # rec["image"] is something like "roco2/train_images/ROCOv2_2023_train_000001.jpg"
        caption   = rec["text"]
        full_path = rec["image"]
        filename  = Path(full_path).name  # "ROCOv2_2023_train_000001.jpg"
        fout.write(
            json.dumps({"image": filename, "text": caption}, ensure_ascii=False) + "\n"
        )

print(f"Wrote corrected JSONL to: {new}")
