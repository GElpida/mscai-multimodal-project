import argparse
import io
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

# Allow "python src/prepare_data.py" from the repo root.
sys.path.insert(0, str(Path(__file__).parent))
from artpedia_dataset import ArtpediaDataset

ARTPEDIA_JSON = Path(__file__).parent.parent / "dataset" / "artpedia" / "artpedia.json"
SLEEP_BETWEEN_DOWNLOADS = 0.5  # seconds between successful downloads

# Retry settings for image downloads.
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds; doubles on each retry (2, 4, 8, 16, ...)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def download_with_retry(url):
    """Download url and return a PIL RGB Image.

    Retries on HTTP 429, 5xx, and connection/timeout errors with exponential
    backoff.  HTTP 404 is treated as permanent and raises immediately.
    Raises RuntimeError after MAX_RETRIES failed attempts.
    """
    delay = RETRY_BASE_DELAY
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            return Image.open(io.BytesIO(data)).convert("RGB")

        except urllib.error.HTTPError as e:
            # 404 is permanent — no point retrying.
            if e.code == 404:
                raise RuntimeError(f"HTTP 404 (permanent): {url}") from e

            if e.code == 429 or (500 <= e.code < 600):
                last_exc = e
                if attempt == MAX_RETRIES:
                    break
                # Respect Retry-After header when the server sends one.
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay + random.random()
                print(f"    [retry {attempt}/{MAX_RETRIES}] HTTP {e.code} — waiting {wait:.1f}s")
                time.sleep(wait)
                delay *= 2
            else:
                raise RuntimeError(f"HTTP {e.code}: {url}") from e

        except (urllib.error.URLError, OSError) as e:
            last_exc = e
            if attempt == MAX_RETRIES:
                break
            wait = delay + random.random()
            print(f"    [retry {attempt}/{MAX_RETRIES}] connection error — waiting {wait:.1f}s")
            time.sleep(wait)
            delay *= 2

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {url}") from last_exc


def parse_args():
    p = argparse.ArgumentParser(
        description="Download/cache Artpedia images and build a training manifest."
    )
    p.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="Dataset split to process (default: train)",
    )
    p.add_argument(
        "--output-dir", default="data",
        help="Root output directory; pass a Drive path in Colab (default: data)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    split = args.split

    # Create output directories if they don't exist yet.
    img_dir = output_dir / "images" / split
    processed_dir = output_dir / "processed"
    img_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = processed_dir / f"{split}.jsonl"

    # Load only the records for the requested split.
    dataset = ArtpediaDataset(ARTPEDIA_JSON, split=split)
    print(f"Loaded {len(dataset)} records for split='{split}'")
    print(f"Images   → {img_dir.resolve()}")
    print(f"Manifest → {manifest_path.resolve()}\n")

    succeeded = 0  # freshly downloaded
    reused = 0     # already on disk
    skipped = 0    # empty captions or download failures

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for i in range(len(dataset)):
            record = dataset[i]

            # Skip records with no visual description (nothing to supervise on).
            if not record.get("visual_sentences"):
                skipped += 1
                continue

            # Build a single caption by joining all visual sentences.
            caption = " ".join(record["visual_sentences"])
            img_path = img_dir / f"{i}.jpg"

            if img_path.exists():
                # Image already cached — no download needed.
                reused += 1
            else:
                # Download with retry/backoff; uses the project User-Agent above.
                try:
                    image = download_with_retry(record["img_url"])
                    image.save(img_path, format="JPEG")
                    succeeded += 1
                except RuntimeError as e:
                    print(f"  [WARN] index {i} — {e}")
                    skipped += 1
                    continue
                # Pause between actual downloads to avoid hammering Wikimedia.
                time.sleep(SLEEP_BETWEEN_DOWNLOADS)

            # Write one JSON line to the manifest (absolute path avoids ambiguity).
            entry = {
                "image_path": str(img_path.resolve()),
                "caption": caption,
                "title": record["title"],
            }
            manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Print progress every 50 written records.
            written = succeeded + reused
            if written % 50 == 0 and written > 0:
                print(f"  {written} written | {skipped} skipped so far...")

    total = len(dataset)
    print(f"\nDone. {total} records processed.")
    print(f"  Downloaded : {succeeded}")
    print(f"  Reused     : {reused}")
    print(f"  Skipped    : {skipped}")


if __name__ == "__main__":
    main()
