import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """PyTorch Dataset for BLIP fine-tuning on Artpedia captions.

    Reads a .jsonl manifest produced by prepare_data.py and applies
    LAVIS vis/txt processors at __getitem__ time (no model loaded here).
    """

    def __init__(self, manifest_path, vis_processor, txt_processor, base_dir, split="train"):
        self.vis_processor = vis_processor
        self.txt_processor = txt_processor
        # base_dir is the data output-dir; relative image_paths are resolved against it.
        self.base_dir = Path(base_dir)

        # utf-8-sig strips a leading BOM written by Windows/PowerShell tools.
        manifest_path = Path(manifest_path)
        self.records = []
        bad_lines = 0
        with open(manifest_path, encoding="utf-8-sig") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    snippet = line[:60] + ("…" if len(line) > 60 else "")
                    print(f"  [WARN] manifest line {lineno} skipped ({e}): {snippet!r}")
                    bad_lines += 1
        if bad_lines:
            print(f"  {bad_lines} malformed line(s) skipped while loading manifest.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        # Resolve relative path (forward-slash, cross-platform) against base_dir.
        img_path = self.base_dir / Path(record["image_path"])

        # Load image; give a clear error if the file is missing or corrupt.
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            raise FileNotFoundError(f"Cannot open image at '{img_path}': {e}") from e

        # Apply LAVIS processors — vis returns a tensor, txt returns a string.
        image_tensor = self.vis_processor(image)
        caption = self.txt_processor(record["caption"])

        return {"image": image_tensor, "text_input": caption}


if __name__ == "__main__":
    from lavis.models import load_model_and_preprocess

    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/train.jsonl"
    base_dir      = sys.argv[2] if len(sys.argv) > 2 else "data"

    # Load processors only (is_eval=False gives training-time augmentations).
    # We discard the model itself — this demo only exercises the Dataset.
    _, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=False
    )

    dataset = CaptionDataset(
        manifest_path,
        vis_processor=vis_processors["train"],
        txt_processor=txt_processors["train"],
        base_dir=base_dir,
    )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"image tensor shape : {sample['image'].shape}")
    print(f"text_input         : {sample['text_input']}")
