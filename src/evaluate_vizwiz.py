"""Zero-shot evaluation of BLIP captioning on VizWiz-Caps (streaming).

VizWiz-Caps contains photos taken by blind people — a real-world domain-robustness
test for a model fine-tuned on art-style captions (Artpedia).

Streaming mode: images are fetched on demand from HuggingFace; the full val split
(~7.5 GB) is never written to disk.  Only --limit samples are consumed.
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
            caption      = model.generate({"image": image_tensor})[0]
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

    # Set HuggingFace download timeout BEFORE any HF/datasets import so that
    # slow connections don't hit a premature ReadTimeoutError mid-stream.
    _os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.hf_timeout)

    # ---- Connectivity check — test the HF Hub API, not just the homepage -----
    import requests as _req
    _hf_api = "https://huggingface.co/api/datasets/lmms-lab/VizWiz-Caps"
    try:
        _r = _req.head(_hf_api, timeout=15)
        # 200 or 401/403 both mean we reached the server — only network errors abort.
    except Exception as _e:
        print(
            f"\n[ERROR] Cannot reach HuggingFace Hub API: {_e}\n\n"
            "This is a network-level failure (not an SSL or code issue).\n"
            "Possible fixes:\n"
            "  1. Check your internet connection.\n"
            "  2. If behind a proxy:\n"
            "       $env:HTTPS_PROXY='http://proxy-host:port'  (PowerShell)\n"
            "       set HTTPS_PROXY=http://proxy-host:port     (cmd)\n"
            "  3. If on a corporate/university network, ask IT to whitelist huggingface.co.\n"
            "  4. Try opening https://huggingface.co in a browser to confirm general access.\n",
            flush=True,
        )
        sys.exit(1)
    print(f"HuggingFace reachable (HTTP {_r.status_code}).", flush=True)

    # ---- Stream dataset (no full download) ---------------------------------
    print(
        "Connecting to HuggingFace and streaming VizWiz val split "
        "(this can take a while on a slow connection)...",
        flush=True,
    )
    from datasets import load_dataset   # lazy import; not needed for non-streaming usage
    import time as _time
    stream = None
    for _attempt in range(1, 4):
        try:
            stream = load_dataset("lmms-lab/VizWiz-Caps", split=args.split, streaming=True)
            break
        except Exception as _e:
            if _attempt == 3:
                raise
            print(f"  [WARN] load_dataset attempt {_attempt} failed: {_e}  — retrying in 5 s...", flush=True)
            _time.sleep(5)

    # Defensive field discovery: inspect the first sample and print its schema
    # so that any schema change is immediately visible instead of silently broken.
    first = next(iter(stream))
    print("\n--- First sample schema ---")
    for k, v in first.items():
        shape = getattr(v, "size", None) or (len(v) if hasattr(v, "__len__") else "?")
        print(f"  {k!r:30s} type={type(v).__name__}, shape/len={shape}")
    print("---------------------------\n")

    import itertools
    eval_data = []
    # Chain first back so schema-peek doesn't lose a sample.
    with tqdm(total=args.limit, desc="Collecting VizWiz samples", unit="img") as pbar:
        for sample in itertools.chain([first], stream):
            try:
                image = get_image(sample).convert("RGB")
                refs  = get_references(sample)
                if not refs:
                    continue
            except SystemExit:
                raise
            except Exception as exc:
                print(f"  [WARN] Skipping malformed sample: {exc}")
                continue

            eval_data.append((image, refs))
            pbar.update(1)
            if len(eval_data) >= args.limit:
                break

    print(f"Collected {len(eval_data)} valid samples.", flush=True)
    if not eval_data:
        print("No valid samples — check the schema output above and adjust field helpers.")
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
