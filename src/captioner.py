import re
import sys
from pathlib import Path

import torch
from lavis.models import load_model_and_preprocess


# Matches a trailing dangling connector or bare punctuation that signals an
# incomplete clause, e.g. " and", ", with", "; or", " the", ",".
_DANGLING_RE = re.compile(
    r'[\s,;]*\s+(?:and|with|but|or|the|a|an|in|of|to|for|on|at|by|from)\s*$',
    re.IGNORECASE,
)


def _clean_caption(text: str) -> str:
    """Trim dangling sentence fragments produced by hard token-length cutoff.

    Strategy (conservative — good captions are untouched):
    1. Collapse whitespace; capitalise first letter.
    2. If already ends in . ! ? → return as-is.
    3. If a sentence-ending punctuation exists earlier in the text → cut there.
    4. Otherwise strip any trailing dangling connector/comma and append a period.
    """
    text = re.sub(r'  +', ' ', text.strip())
    if not text:
        return text

    text = text[0].upper() + text[1:]

    if text[-1] in '.!?':
        return text

    # Cut back to the last complete sentence when one exists.
    last_terminal = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_terminal > 0:
        return text[:last_terminal + 1]

    # No terminal punctuation — strip dangling tail and close with a period.
    text = _DANGLING_RE.sub('', text).rstrip(' ,;')
    if text:
        text += '.'
    return text


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

    def caption_image(
        self,
        pil_image,
        num_captions=1,
        max_length=40,
        min_length=10,
        num_beams=1,
        repetition_penalty=1.2,
    ):
        """Caption a PIL RGB image and return a list of caption strings.

        Args:
            pil_image: PIL.Image in RGB mode.
            num_captions: how many independent captions to generate.
            max_length: maximum token length of each caption.
            min_length: minimum token length (prevents empty/trivial output).
            num_beams: 1 = greedy (default, low VRAM); >1 = beam search.
            repetition_penalty: >1.0 penalises repeated tokens; 1.5 strongly
                suppresses looping phrases without distorting fluency.

        Returns:
            List of strings, length == num_captions.
        """
        # Preprocess: resize/normalise the image and add the batch dimension.
        image_tensor = self.vis_processor(pil_image).unsqueeze(0).to(self.device)

        captions = self.model.generate(
            {"image": image_tensor},
            use_nucleus_sampling=False,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            num_captions=num_captions,
        )
        return [_clean_caption(c) for c in captions]


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
