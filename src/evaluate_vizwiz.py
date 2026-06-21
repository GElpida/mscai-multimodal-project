"""Zero-shot evaluation of BLIP captioning on VizWiz-Caps.

VizWiz-Caps contains photos taken by blind people — a real-world domain-robustness
test for a model fine-tuned on art-style captions (Artpedia).

Two loading modes:
  --vizwiz-dir PATH   Load images and captions from a local directory:
                        PATH/val/          ← JPEG images
                        PATH/val.json      ← official annotation file
  (no flag)           Stream from HuggingFace (lmms-lab/VizWiz-Caps).
                      Requires network access; may hit rate limits.
"""

import argparse
import datetime
import json
import os as _os
import ssl
import sys
from pathlib import Path

import certifi as _certifi
import torch

# ── Windows SSL / cert-store fixes ───────────────────────────────────────────
#
# The Windows certificate store has a corrupt entry ([ASN1: NOT_ENOUGH_DATA]).
# Three complementary patches are applied so every HTTP client in this process
# (aiohttp, requests, urllib3, huggingface_hub) bypasses the system store and
# uses certifi's CA bundle instead.
#
# Patch 1 — ssl.create_default_context:
#   Most libraries (aiohttp, urllib3 ≤1.x, httpx) call this function to build
#   their SSL context.  Our replacement skips load_default_certs() (which reads
#   the Windows store) and loads certifi directly.
_orig_create_default_context = ssl.create_default_context

def _certifi_create_default_context(purpose=ssl.Purpose.SERVER_AUTH,
                                    *, cafile=None, capath=None, cadata=None):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = True
    if cafile is None and capath is None and cadata is None:
        ctx.load_verify_locations(cafile=_certifi.where())
    else:
        ctx.load_verify_locations(cafile=cafile, capath=capath, cadata=cadata)
    return ctx

ssl.create_default_context = _certifi_create_default_context

# Patch 2 — ssl.SSLContext._load_windows_store_certs:
#   aiohttp/connector.py calls ssl.create_default_context() at import time, but
#   some other paths call _load_windows_store_certs directly.  Skip malformed certs.
def _patched_load_windows_store_certs(self, storename, purpose):
    try:
        for cert, encoding, trust in ssl.enum_certificates(storename):
            if encoding == "x509_asn" and (trust is True or purpose.oid in trust):
                try:
                    self.load_verify_locations(cadata=cert)
                except ssl.SSLError:
                    pass
    except PermissionError:
        pass

ssl.SSLContext._load_windows_store_certs = _patched_load_windows_store_certs

# Patch 3 — environment variables:
#   urllib3 2.x and requests read these to locate the CA bundle, bypassing
#   whatever ssl.create_default_context returns.
_os.environ["REQUESTS_CA_BUNDLE"] = _certifi.where()
_os.environ["SSL_CERT_FILE"]      = _certifi.where()
_os.environ["CURL_CA_BUNDLE"]     = _certifi.where()

# ── pyarrow compatibility shim ────────────────────────────────────────────────
# pyarrow>=14.0 removed PyExtensionType.  datasets==2.14.6 uses it at class
# definition time in features.py; provide a shim so the import succeeds.
import pyarrow as pa
if not hasattr(pa, "PyExtensionType"):
    class _PyExtensionTypeShim(pa.ExtensionType):
        def __init__(self, storage_type):
            pa.ExtensionType.__init__(self, storage_type, type(self).__name__)

        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

    pa.PyExtensionType = _PyExtensionTypeShim

# Allow "python src/evaluate_vizwiz.py" from the repo root.
sys.path.insert(0, str(Path(__file__).parent))

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Defensive field helpers
# ---------------------------------------------------------------------------

# Candidate image-field names, tried in order.  VizWiz-Caps typically uses
# "image", but the HuggingFace schema can vary between dataset versions.
_IMAGE_KEYS = ["image", "img", "pil_image", "pixel_values"]

# Candidate captions-field names.  VizWiz-Caps typically ships ~5 captions
# per image; the field may be a list of strings or a list of dicts.
_CAPTION_KEYS = ["captions", "caption", "annotations", "references", "text"]


def get_image(sample: dict):
    """Return the PIL image from a sample, trying known field names in order."""
    for key in _IMAGE_KEYS:
        val = sample.get(key)
        if val is not None:
            return val
    print(f"[ERROR] No image field found. Sample keys: {list(sample.keys())}")
    sys.exit(1)


def get_references(sample: dict) -> list:
    """Return a list of reference caption strings from a sample."""
    for key in _CAPTION_KEYS:
        val = sample.get(key)
        if val is None:
            continue
        # List of plain strings — most common case.
        if isinstance(val, list) and val:
            if isinstance(val[0], str):
                return val
            # List of dicts e.g. [{"caption": "..."}, ...]
            if isinstance(val[0], dict):
                texts = [v.get("caption") or v.get("text", "") for v in val]
                return [t for t in texts if t]
        if isinstance(val, str):
            return [val]
    print(f"[ERROR] No captions field found. Sample keys: {list(sample.keys())}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_model(model, vis_processor, eval_data, device, desc="Generating captions"):
    """Generate one caption per sample; return {'Bleu_4': float, 'CIDEr': float}."""
    gts, res = {}, {}
    model.eval()

    with torch.no_grad():
        for img_id, (image, refs) in enumerate(
            tqdm(eval_data, desc=desc, total=len(eval_data), unit="img")
        ):
            image_tensor = vis_processor(image).unsqueeze(0).to(device)
            caption      = model.generate(
                {"image": image_tensor},
                use_nucleus_sampling=False,
                num_beams=3,
                max_length=40,
                min_length=10,
                repetition_penalty=1.2,
            )[0]
            gts[img_id]  = refs
            res[img_id]  = [caption]

    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    cider_score, _ = Cider().compute_score(gts, res)
    return {"Bleu_4": bleu_scores[3], "CIDEr": cider_score}


def print_results(label, metrics):
    print(f"\n  {label}")
    for k, v in metrics.items():
        print(f"    {k:10s}: {v:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Zero-shot BLIP eval on VizWiz-Caps via HuggingFace streaming."
    )
    p.add_argument("--checkpoint",        default=None,
                   help="Fine-tuned state_dict .pth (optional; skipped if omitted)")
    p.add_argument("--limit",             type=int, default=300,
                   help="Number of streamed samples to evaluate (default: 300)")
    p.add_argument("--split",             default="val")
    p.add_argument("--results-json",      default="vizwiz_eval_results.json")
    p.add_argument("--use-mlflow",        action="store_true", default=False)
    p.add_argument("--mlflow-experiment", default="blip-vizwiz")
    p.add_argument("--run-name",          default=None)
    p.add_argument("--output-dir",        default="outputs",
                   help="Root outputs directory; a timestamped vizwiz_<...>/ subfolder is created")
    p.add_argument("--vizwiz-dir",         default=None,
                   help="Path to local VizWiz directory containing "
                        "<split>/ images folder and <split>.json annotation file. "
                        "When provided, HuggingFace streaming is skipped entirely.")
    p.add_argument("--hf-timeout",        type=int, default=120,
                   help="HuggingFace download timeout in seconds (default: 120); "
                        "increase on slow connections to avoid ReadTimeoutError")
    return p.parse_args()


def main():
    args   = parse_args()

    # Set BEFORE any import that might pull in mlflow (e.g. LAVIS → omegaconf → mlflow).
    _os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Per-run timestamped folder.
    ts_str  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output_dir) / f"vizwiz_{ts_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {run_dir}")

    # ---- Build eval_data: local files OR HuggingFace streaming ---------------
    eval_data = []

    if args.vizwiz_dir:
        # ---- Local mode: load from downloaded annotation JSON + image folder --
        from collections import defaultdict
        from PIL import Image as _PILImage

        vizwiz_dir = Path(args.vizwiz_dir)
        ann_path   = vizwiz_dir / f"{args.split}.json"
        img_dir    = vizwiz_dir / args.split

        if not ann_path.exists():
            print(f"[ERROR] Annotation file not found: {ann_path}", flush=True)
            sys.exit(1)
        if not img_dir.is_dir():
            print(f"[ERROR] Images folder not found: {img_dir}", flush=True)
            sys.exit(1)

        print(f"Loading annotations from {ann_path} ...", flush=True)
        with open(ann_path, encoding="utf-8") as _f:
            _ann = json.load(_f)

        # Group captions by image_id; drop rejected and precanned entries
        # (precanned = preset "unanswerable" response, not a real description).
        _id_to_caps = defaultdict(list)
        for _a in _ann["annotations"]:
            if _a.get("is_rejected") or _a.get("is_precanned"):
                continue
            _cap = _a.get("caption", "").strip()
            if _cap:
                _id_to_caps[_a["image_id"]].append(_cap)

        print(
            f"  {len(_ann['images'])} images in JSON, "
            f"{len(_ann['annotations'])} annotations "
            f"→ {sum(len(v) for v in _id_to_caps.values())} usable captions",
            flush=True,
        )

        _missing = 0
        with tqdm(total=args.limit or len(_ann["images"]),
                  desc="Loading VizWiz images", unit="img") as _pbar:
            for _img_info in _ann["images"]:
                _img_id = _img_info["id"]
                _refs   = _id_to_caps.get(_img_id)
                if not _refs:
                    continue
                _img_path = img_dir / _img_info["file_name"]
                if not _img_path.exists():
                    _missing += 1
                    continue
                try:
                    _image = _PILImage.open(_img_path).convert("RGB")
                except Exception as _exc:
                    print(f"  [WARN] Cannot open {_img_path.name}: {_exc}")
                    continue
                eval_data.append((_image, _refs))
                _pbar.update(1)
                if args.limit and len(eval_data) >= args.limit:
                    break

        if _missing:
            print(f"  [INFO] {_missing} images listed in JSON but not found on disk (skipped).", flush=True)

    else:
        # ---- HuggingFace streaming mode --------------------------------------
        _os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.hf_timeout)

        import requests as _req
        _hf_api = "https://huggingface.co/api/datasets/lmms-lab/VizWiz-Caps"
        try:
            _r = _req.head(_hf_api, timeout=15)
        except Exception as _e:
            print(
                f"\n[ERROR] Cannot reach HuggingFace Hub API: {_e}\n\n"
                "Tip: pass --vizwiz-dir to use a locally downloaded dataset.\n",
                flush=True,
            )
            sys.exit(1)
        print(f"HuggingFace reachable (HTTP {_r.status_code}).", flush=True)

        print("Connecting to HuggingFace and streaming VizWiz val split ...", flush=True)
        from datasets import load_dataset
        import time as _time
        _MAX_ATTEMPTS = 7
        _BACKOFF_BASE  = 10
        _stream = None
        for _attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                _stream = load_dataset("lmms-lab/VizWiz-Caps", split=args.split, streaming=True)
                break
            except Exception as _e:
                if _attempt == _MAX_ATTEMPTS:
                    raise
                _wait = _BACKOFF_BASE * (2 ** (_attempt - 1))
                print(
                    f"  [WARN] load_dataset attempt {_attempt}/{_MAX_ATTEMPTS - 1} failed: {_e}"
                    f"  — retrying in {_wait} s...",
                    flush=True,
                )
                _time.sleep(_wait)

        _first = next(iter(_stream))
        print("\n--- First sample schema ---")
        for _k, _v in _first.items():
            _shape = getattr(_v, "size", None) or (len(_v) if hasattr(_v, "__len__") else "?")
            print(f"  {_k!r:30s} type={type(_v).__name__}, shape/len={_shape}")
        print("---------------------------\n")

        import itertools
        with tqdm(total=args.limit, desc="Collecting VizWiz samples", unit="img") as _pbar:
            for _sample in itertools.chain([_first], _stream):
                try:
                    _image = get_image(_sample).convert("RGB")
                    _refs  = get_references(_sample)
                    if not _refs:
                        continue
                except SystemExit:
                    raise
                except Exception as _exc:
                    print(f"  [WARN] Skipping malformed sample: {_exc}")
                    continue
                eval_data.append((_image, _refs))
                _pbar.update(1)
                if len(eval_data) >= args.limit:
                    break

    print(f"Collected {len(eval_data)} valid samples.", flush=True)
    if not eval_data:
        print("No valid samples — check your --vizwiz-dir path or HuggingFace connection.")
        sys.exit(1)

    # ---- Load LAVIS (deferred so field-discovery errors surface first) -----
    from lavis.models import load_model_and_preprocess

    # ---- Pretrained --------------------------------------------------------
    print("\nLoading pretrained blip_caption/base_coco ...", flush=True)
    model_pre, vis_pre, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )
    vis_proc = vis_pre["eval"]

    print("Evaluating pretrained model ...", flush=True)
    pre_metrics = eval_model(model_pre, vis_proc, eval_data, device,
                             desc="Generating captions [pretrained]")
    print_results("PRETRAINED (VizWiz zero-shot)", pre_metrics)

    results = {"pretrained": pre_metrics}

    # ---- Fine-tuned (optional) ---------------------------------------------
    ft_metrics = None
    if args.checkpoint:
        # Free pretrained model before loading fine-tuned to avoid dual-model VRAM pressure.
        del model_pre, vis_proc
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"\nLoading fine-tuned checkpoint: {args.checkpoint}", flush=True)
        model_ft, vis_ft, _ = load_model_and_preprocess(
            name="blip_caption", model_type="base_coco", is_eval=True, device=device
        )
        raw = torch.load(args.checkpoint, map_location=device, weights_only=True)
        # Unwrap if saved as {"model": ...} or {"state_dict": ...}.
        if isinstance(raw, dict) and not any(
            k.startswith(("visual_encoder", "text_decoder")) for k in list(raw.keys())[:5]
        ):
            state_dict = raw.get("model") or raw.get("state_dict") or raw
        else:
            state_dict = raw
        result = model_ft.load_state_dict(state_dict, strict=False)
        print(
            f"  missing keys: {len(result.missing_keys)}  "
            f"unexpected: {len(result.unexpected_keys)}"
        )
        model_ft.eval()

        print("Evaluating fine-tuned model ...", flush=True)
        ft_metrics = eval_model(model_ft, vis_ft["eval"], eval_data, device,
                                desc="Generating captions [fine-tuned]")
        print_results("FINE-TUNED (VizWiz zero-shot)", ft_metrics)
        results["finetuned"] = ft_metrics

    # ---- Side-by-side summary ----------------------------------------------
    if ft_metrics:
        print(f"\n  {'Metric':<12} {'Pretrained':>12} {'Fine-tuned':>12}")
        print("  " + "-" * 38)
        for k in pre_metrics:
            print(f"  {k:<12} {pre_metrics[k]:>12.4f} {ft_metrics[k]:>12.4f}")

    # ---- Optional MLflow ---------------------------------------------------
    if args.use_mlflow:
        import mlflow
        db = Path(args.output_dir).resolve() / "mlruns.db"
        mlflow.set_tracking_uri("sqlite:///" + str(db).replace("\\", "/"))
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=args.run_name):
            mlflow.log_params({
                "checkpoint": args.checkpoint or "pretrained_only",
                "limit":      args.limit,
                "split":      args.split,
            })
            for k, v in pre_metrics.items():
                mlflow.log_metric(f"pretrained_{k}", v)
            if ft_metrics:
                for k, v in ft_metrics.items():
                    mlflow.log_metric(f"finetuned_{k}", v)

    # ---- Save results ------------------------------------------------------
    out = run_dir / Path(args.results_json).name
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}", flush=True)


if __name__ == "__main__":
    main()
