import sys
from pathlib import Path

import torch
from lavis.models import load_model_and_preprocess


class Captioner:
    """Thin wrapper around a LAVIS captioning model."""

    def __init__(self, model_name="blip_caption", model_type="base_coco", device=None,
                 checkpoint_path=None):
        # Auto-detect device when not specified.
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load model and its paired visual preprocessor via LAVIS.
        self.model, vis_processors, _ = load_model_and_preprocess(
            name=model_name,
            model_type=model_type,
            is_eval=True,
            device=device,
        )
        # Only the "eval" preprocessor is needed for inference.
        self.vis_processor = vis_processors["eval"]

        # Optionally load a fine-tuned state_dict checkpoint.
        if checkpoint_path is not None:
            raw = torch.load(checkpoint_path, map_location=device)
            # Unwrap if saved as {"model": ..., } or {"state_dict": ...}.
            if isinstance(raw, dict) and not any(
                k.startswith(("visual_encoder", "text_decoder"))
                for k in list(raw.keys())[:5]
            ):
                state_dict = raw.get("model") or raw.get("state_dict") or raw
            else:
                state_dict = raw
            result = self.model.load_state_dict(state_dict, strict=False)
            missing  = len(result.missing_keys)
            unexpected = len(result.unexpected_keys)
            print(f"Loaded checkpoint: {checkpoint_path}")
            print(f"  missing keys: {missing}  unexpected keys: {unexpected}")

        self.model.eval()

    def caption_image(self, pil_image, num_captions=1):
        """Caption a PIL RGB image and return a list of caption strings.

        Args:
            pil_image: PIL.Image in RGB mode.
            num_captions: how many independent captions to generate.

        Returns:
            List of strings, length == num_captions.
        """
        # Preprocess: resize/normalise the image and add the batch dimension.
        image_tensor = self.vis_processor(pil_image).unsqueeze(0).to(self.device)

        # LAVIS generate() accepts num_captions directly.
        captions = self.model.generate(
            {"image": image_tensor},
            num_captions=num_captions,
        )
        return captions  # already a list of strings


if __name__ == "__main__":
    # Allow running as "python src/captioner.py" from the repo root.
    sys.path.insert(0, str(Path(__file__).parent))
    from artpedia_dataset import ArtpediaDataset

    json_path = Path(__file__).parent.parent / "dataset" / "artpedia" / "artpedia.json"

    # Load test split and grab the first record.
    dataset = ArtpediaDataset(json_path, split="test")
    record = dataset[0]

    print(f"Title : {record['title']}")
    print("Ground-truth visual sentences:")
    for s in record["visual_sentences"]:
        print(f"  - {s}")

    # Download the image on demand, then caption it.
    print("\nDownloading image and generating caption...")
    image = dataset.load_image(0)

    captioner = Captioner()
    captions = captioner.caption_image(image, num_captions=1)
    print(f"\nGenerated caption: {captions[0]}")
