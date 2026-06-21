import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

ARTPEDIA_JSON = Path(__file__).parent.parent / "dataset" / "artpedia" / "artpedia.json"


def load_image(image_path):
    """Return a PIL image from a local path or an artpedia URL fallback."""
    if image_path:
        p = Path(image_path)
        if not p.is_file():
            print(f"Error: file not found: {image_path}")
            sys.exit(1)
        return Image.open(p).convert("RGB"), image_path

    # No path given — pick the first entry from artpedia.json
    with open(ARTPEDIA_JSON) as f:
        data = json.load(f)
    first = next(iter(data.values()))
    url = first["img_url"]
    title = first["title"]
    print(f"No path given. Using artpedia image: {title}\n  URL: {url}")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    return Image.open(io.BytesIO(raw)).convert("RGB"), url


def main():
    parser = argparse.ArgumentParser(description="Sanity check: caption an image with BLIP")
    parser.add_argument("image_path", nargs="?", help="Path to image (omit to use first artpedia entry)")
    args = parser.parse_args()

    raw_image, source = load_image(args.image_path)

    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load BLIP captioning model and its paired preprocessing transforms
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )

    # Preprocess and caption
    image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption = model.generate({"image": image_tensor})[0]
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
