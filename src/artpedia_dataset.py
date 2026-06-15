import io
import json
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

# Wikimedia blocks requests without a User-Agent; mimic a real browser.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


class ArtpediaDataset:
    """Lazy-loading dataset wrapper for Artpedia.

    Images are NOT loaded at init time; call load_image(index) on demand.
    """

    def __init__(self, json_path, split=None):
        """Load records from json_path, optionally filtering by split."""
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        # The JSON is a dict keyed by artwork ID; we only need the values.
        records = list(raw.values())

        # Keep only the requested split when specified.
        if split is not None:
            records = [r for r in records if r.get("split") == split]

        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        """Return the raw record dict for the given index."""
        return self.records[index]

    def load_image(self, index):
        """Download and return the image at records[index] as a PIL RGB image.

        Raises RuntimeError with a clear message if the download fails.
        """
        record = self.records[index]
        url = record["img_url"]

        req = urllib.request.Request(url, headers=_HEADERS)
        try:
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Failed to download image (HTTP {e.code}): {url}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to reach URL ({e.reason}): {url}"
            ) from e

        return Image.open(io.BytesIO(data)).convert("RGB")


if __name__ == "__main__":
    json_path = Path(__file__).parent.parent / "dataset" / "artpedia" / "artpedia.json"

    # Load only test-split records.
    dataset = ArtpediaDataset(json_path, split="test")
    print(f"Test split: {len(dataset)} records")

    # Inspect the first record.
    first = dataset[0]
    print(f"First title : {first['title']}")
    print(f"Visual sentences: {len(first['visual_sentences'])}")

    # Download the first image on demand.
    print("Downloading first image...")
    img = dataset.load_image(0)
    print(f"Image size: {img.width} x {img.height}")
