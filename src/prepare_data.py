import argparse
import io
import json
import random
import re
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

# Retry settings for image downloads.
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds; doubles on each retry (2, 4, 8, 16, ...)
MAX_WAIT = 30.0         # seconds; stop the run if any retry wait would exceed this

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

THUMB_WIDTH = 500  # default pixel width for Wikimedia thumbnail downloads

# Wikimedia only generates thumbnails at these standard widths; other values return HTTP 400.
ALLOWED_THUMB_WIDTHS = [20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840]

# Matches upload.wikimedia.org URLs of the form:
#   /wikipedia/<project>/<X>/<XX>/<filename>
# capturing the base-up-to-project, the hash+filename path, and the filename alone.
_WIKI_THUMB_RE = re.compile(
    r"^(https://upload\.wikimedia\.org/wikipedia/[^/]+)"
    r"(/[0-9a-f]/[0-9a-f]{2}/(.+))$"
)


def wikimedia_thumb_url(url, width=THUMB_WIDTH):
    """Convert a Wikimedia original URL to its cached thumbnail URL.

    Inserts /thumb after the project name and appends /<width>px-<filename>.
    SVG originals get a .png suffix on the thumbnail filename (Wikimedia renders
    SVGs to PNG for thumbnails). Non-matching URLs are returned unchanged.

    Example:
      https://upload.wikimedia.org/wikipedia/commons/8/8c/The_Fighting_Temeraire.jpg
      -> https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/The_Fighting_Temeraire.jpg/800px-The_Fighting_Temeraire.jpg
    """
    m = _WIKI_THUMB_RE.match(url)
    if not m:
        return url
    base, hash_path, filename = m.group(1), m.group(2), m.group(3)
    # Wikimedia renders SVG thumbnails as PNG
    thumb_filename = filename + ".png" if filename.lower().endswith(".svg") else filename
    return f"{base}/thumb{hash_path}/{width}px-{thumb_filename}"


class RateLimitAbort(Exception):
    """Raised when a retry wait would exceed MAX_WAIT seconds."""


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
                if wait > MAX_WAIT:
                    raise RateLimitAbort(wait)
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
            if wait > MAX_WAIT:
                raise RateLimitAbort(wait)
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
    p.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to wait between successful downloads (default: 2.0)",
    )
    p.add_argument(
        "--thumb-width", type=int, default=THUMB_WIDTH,
        help=f"Pixel width for Wikimedia thumbnail downloads (default: {THUMB_WIDTH})",
    )
    p.add_argument(
        "--rebuild-manifest", action="store_true", default=False,
        help="Overwrite any existing manifest and rewrite all entries with relative paths. "
             "Images already on disk are reused; only missing ones are re-downloaded.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Snap thumb_width up to the nearest allowed standard Wikimedia size.
    if args.thumb_width not in ALLOWED_THUMB_WIDTHS:
        snapped = next(
            (w for w in ALLOWED_THUMB_WIDTHS if w >= args.thumb_width),
            ALLOWED_THUMB_WIDTHS[-1],
        )
        print(f"Width {args.thumb_width} is not a standard Wikimedia size; using {snapped} instead.")
        args.thumb_width = snapped

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

    # In rebuild mode we ignore any existing manifest and rewrite from scratch.
    # Otherwise read it to find already-recorded indices so reruns don't duplicate lines.
    already_recorded = set()
    if args.rebuild_manifest:
        print("--rebuild-manifest set: overwriting existing manifest.\n")
        manifest_open_mode = "w"
    else:
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        # Index is encoded in the image filename (e.g. "42.jpg").
                        stem = Path(entry["image_path"]).stem
                        already_recorded.add(int(stem))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
            print(f"Resuming: {len(already_recorded)} records already in manifest.\n")
        manifest_open_mode = "a"

    succeeded = 0  # freshly downloaded this run
    reused = 0     # already on disk (skipped download but still in manifest)
    skipped = 0    # empty captions or download failures

    with open(manifest_path, manifest_open_mode, encoding="utf-8") as manifest:
        try:
            for i in range(len(dataset)):
                record = dataset[i]

                # Skip records with no visual description (nothing to supervise on).
                if not record.get("visual_sentences"):
                    skipped += 1
                    continue

                # Build a single caption by joining all visual sentences.
                caption = " ".join(record["visual_sentences"])
                img_path = img_dir / f"{i}.jpg"

                # Already downloaded and recorded in a previous run — skip entirely.
                if i in already_recorded:
                    reused += 1
                    continue

                if img_path.exists():
                    # Image on disk but not yet in manifest — record it now.
                    reused += 1
                else:
                    # Prefer the cached thumbnail (smaller, less rate-limited).
                    # Fall back to the original URL once if the thumbnail is 404.
                    original_url = record["img_url"]
                    thumb_url = wikimedia_thumb_url(original_url, args.thumb_width)
                    try:
                        try:
                            image = download_with_retry(thumb_url)
                        except RuntimeError as e:
                            if thumb_url != original_url and "404" in str(e):
                                print(f"    [fallback] thumbnail 404 — retrying original URL")
                                image = download_with_retry(original_url)
                            else:
                                raise
                        image.save(img_path, format="JPEG")
                        succeeded += 1
                    except RuntimeError as e:
                        print(f"  [WARN] index {i} — {e}")
                        skipped += 1
                        continue
                    # Pause between actual downloads to avoid hammering Wikimedia.
                    time.sleep(args.delay)

                # Store a relative path with forward slashes so the manifest is
                # portable across machines (Colab, Windows, etc.).
                rel_path = f"images/{split}/{i}.jpg"
                entry = {
                    "image_path": rel_path,
                    "caption": caption,
                    "title": record["title"],
                }
                manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")

                # Print progress every 50 written records.
                written = succeeded + reused
                if written % 50 == 0 and written > 0:
                    print(f"  {written} written | {skipped} skipped so far...")

        except RateLimitAbort as e:
            manifest.flush()
            print(
                f"\nWikimedia is rate-limiting us (Retry-After={e.args[0]:.0f}s). "
                "Stopping for now — rerun later to resume from where we left off."
            )
            print(f"  Downloaded so far: {succeeded}")
            sys.exit(1)

    total = len(dataset)
    print(f"\nDone. {total} records processed.")
    print(f"  Downloaded : {succeeded}")
    print(f"  Reused     : {reused}")
    print(f"  Skipped    : {skipped}")


if __name__ == "__main__":
    main()
