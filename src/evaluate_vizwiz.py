"""Zero-shot evaluation of BLIP captioning on VizWiz-Caps (streaming).

VizWiz-Caps contains photos taken by blind people — a real-world domain-robustness
test for a model fine-tuned on art-style captions (Artpedia).

Streaming mode: images are fetched on demand from HuggingFace; the full val split
(~7.5 GB) is never written to disk.  Only --limit samples are consumed.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Allow "python src/evaluate_vizwiz.py" from the repo root.
sys.path.insert(0, str(Path(__file__).parent))

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider


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

def eval_model(model, vis_processor, eval_data, device):
    """Generate one caption per sample; return {'Bleu_4': float, 'CIDEr': float}."""
    gts, res = {}, {}
    model.eval()

    with torch.no_grad():
        for img_id, (image, refs) in enumerate(eval_data):
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
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Stream dataset (no full download) ---------------------------------
    print(
        f"Streaming VizWiz-Caps split='{args.split}' — "
        f"collecting up to {args.limit} samples ..."
    )
    from datasets import load_dataset   # lazy import; not needed for non-streaming usage
    stream = load_dataset("lmms-lab/VizWiz-Caps", split=args.split, streaming=True)

    # Defensive field discovery: inspect the first sample and print its schema
    # so that any schema change is immediately visible instead of silently broken.
    first = next(iter(stream))
    print("\n--- First sample schema ---")
    for k, v in first.items():
        shape = getattr(v, "size", None) or (len(v) if hasattr(v, "__len__") else "?")
        print(f"  {k!r:30s} type={type(v).__name__}, shape/len={shape}")
    print("---------------------------\n")

    eval_data = []
    for sample in stream:
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
        if len(eval_data) >= args.limit:
            break

    print(f"Collected {len(eval_data)} valid samples.")
    if not eval_data:
        print("No valid samples — check the schema output above and adjust field helpers.")
        sys.exit(1)

    # ---- Load LAVIS (deferred so field-discovery errors surface first) -----
    from lavis.models import load_model_and_preprocess

    # ---- Pretrained --------------------------------------------------------
    print("\nLoading pretrained blip_caption/base_coco ...")
    model_pre, vis_pre, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )
    vis_proc = vis_pre["eval"]

    print("Evaluating pretrained model ...")
    pre_metrics = eval_model(model_pre, vis_proc, eval_data, device)
    print_results("PRETRAINED (VizWiz zero-shot)", pre_metrics)

    results = {"pretrained": pre_metrics}

    # ---- Fine-tuned (optional) ---------------------------------------------
    ft_metrics = None
    if args.checkpoint:
        print(f"\nLoading fine-tuned checkpoint: {args.checkpoint}")
        model_ft, vis_ft, _ = load_model_and_preprocess(
            name="blip_caption", model_type="base_coco", is_eval=True, device=device
        )
        raw = torch.load(args.checkpoint, map_location=device)
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

        print("Evaluating fine-tuned model ...")
        ft_metrics = eval_model(model_ft, vis_ft["eval"], eval_data, device)
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
    out = Path(args.results_json)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
