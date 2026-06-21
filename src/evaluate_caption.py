"""Evaluate BLIP captioning on Artpedia test split using BLEU-4 and CIDEr.
Compares pretrained base model vs an optional fine-tuned checkpoint.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from tqdm.auto import tqdm

# Allow "python src/evaluate_caption.py" from the repo root.
sys.path.insert(0, str(Path(__file__).parent))
from artpedia_dataset import ArtpediaDataset

ARTPEDIA_DEFAULT = str(
    Path(__file__).parent.parent / "dataset" / "artpedia" / "artpedia.json"
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate BLIP captioning on Artpedia.")
    p.add_argument("--manifest",     required=True, help="Path to test .jsonl manifest")
    p.add_argument("--base-dir",     required=True, help="Base dir for resolving relative image paths")
    p.add_argument("--artpedia",     default=ARTPEDIA_DEFAULT, help="Path to artpedia.json")
    p.add_argument("--checkpoint",   default=None,   help="Fine-tuned state_dict .pth (optional)")
    p.add_argument("--split",        default="test")
    p.add_argument("--limit",        type=int, default=0,
                   help="Evaluate only first N images (0 = all)")
    p.add_argument("--results-json", default="eval_results.json",
                   help="Where to write metric results (default: eval_results.json)")
    p.add_argument("--use-mlflow",        action="store_true", default=False,
                   help="Enable MLflow experiment tracking")
    p.add_argument("--mlflow-experiment", default="blip-artpedia",
                   help="MLflow experiment name (default: blip-artpedia)")
    p.add_argument("--run-name",          default=None,
                   help="MLflow run name (optional)")
    p.add_argument("--output-dir",        default="outputs",
                   help="Root outputs directory; a timestamped eval_<...>/ subfolder is created")
    return p.parse_args()


def load_eval_data(args):
    """Read manifest + artpedia to build a list of (img_path, refs) pairs."""
    artpedia_ds = ArtpediaDataset(args.artpedia, split=args.split)
    base_dir    = Path(args.base_dir)

    eval_data = []
    total     = args.limit if args.limit > 0 else None
    with tqdm(desc="Loading eval images", total=total, unit="img") as pbar:
        with open(args.manifest, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Derive the split-local index from the image filename stem (e.g. "42.jpg" → 42).
                try:
                    idx = int(Path(entry["image_path"]).stem)
                except (ValueError, KeyError):
                    print(f"  [WARN] Cannot parse index from '{entry.get('image_path')}' — skipping.")
                    continue

                if idx >= len(artpedia_ds):
                    print(f"  [WARN] Index {idx} out of range ({len(artpedia_ds)}) — skipping.")
                    continue

                refs = artpedia_ds[idx].get("visual_sentences", [])
                if not refs:
                    continue

                img_path = base_dir / Path(entry["image_path"])
                if not img_path.exists():
                    print(f"  [WARN] Image not found: {img_path} — skipping.")
                    continue

                eval_data.append((img_path, refs))
                pbar.update(1)
                if args.limit > 0 and len(eval_data) >= args.limit:
                    break

    return eval_data


def eval_model(model, vis_processor, eval_data, device, n_samples=5,
               desc="Generating captions"):
    """Generate captions; return ({"Bleu_4": float, "CIDEr": float}, samples).

    samples — list of (img_id, filename, generated_caption, first_reference)
    for the first n_samples images.
    """
    gts, res, samples = {}, {}, []
    model.eval()

    with torch.no_grad():
        for img_id, (img_path, refs) in enumerate(
            tqdm(eval_data, desc=desc, total=len(eval_data), unit="img")
        ):
            image        = Image.open(img_path).convert("RGB")
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
            if len(samples) < n_samples:
                samples.append((img_id, img_path.name, caption, refs[0]))

    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    cider_score, _ = Cider().compute_score(gts, res)

    return {"Bleu_4": bleu_scores[3], "CIDEr": cider_score}, samples


def print_results(label, metrics):
    print(f"\n  {label}")
    for k, v in metrics.items():
        print(f"    {k:10s}: {v:.4f}")


def main():
    args   = parse_args()

    # Set BEFORE any import that might pull in mlflow (e.g. LAVIS → omegaconf → mlflow).
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Per-run timestamped folder.
    ts_str  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output_dir) / f"eval_{ts_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {run_dir}")

    print("Building evaluation set from manifest...", flush=True)
    eval_data = load_eval_data(args)
    print(f"Evaluation set: {len(eval_data)} images", flush=True)
    if not eval_data:
        print("No images to evaluate — check manifest and base-dir.")
        sys.exit(1)

    from lavis.models import load_model_and_preprocess

    # --- Pretrained model ---
    print("\nLoading pretrained blip_caption/base_coco...", flush=True)
    model_pre, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )
    vis_proc_eval = vis_processors["eval"]

    print("Evaluating pretrained model...", flush=True)
    pre_metrics, pre_samples = eval_model(model_pre, vis_proc_eval, eval_data, device,
                                          desc="Generating captions [pretrained]")
    print_results("PRETRAINED", pre_metrics)

    results = {"pretrained": pre_metrics}

    # --- Fine-tuned checkpoint (optional) ---
    ft_metrics = None
    if args.checkpoint:
        # Free pretrained model before loading fine-tuned to avoid dual-model VRAM pressure.
        del model_pre, vis_proc_eval
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"\nLoading fine-tuned checkpoint: {args.checkpoint}...", flush=True)
        model_ft, vis_processors_ft, _ = load_model_and_preprocess(
            name="blip_caption", model_type="base_coco", is_eval=True, device=device
        )
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        result = model_ft.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"  missing keys: {len(result.missing_keys)}  unexpected: {len(result.unexpected_keys)}")
        model_ft.eval()

        print("Evaluating fine-tuned model...", flush=True)
        ft_metrics, _ = eval_model(model_ft, vis_processors_ft["eval"], eval_data, device,
                                   desc="Generating captions [fine-tuned]")
        print_results("FINE-TUNED", ft_metrics)
        results["finetuned"] = ft_metrics

    # --- Side-by-side summary ---
    if ft_metrics:
        print(f"\n  {'Metric':<12} {'Pretrained':>12} {'Fine-tuned':>12}")
        print("  " + "-" * 38)
        for k in pre_metrics:
            print(f"  {k:<12} {pre_metrics[k]:>12.4f} {ft_metrics[k]:>12.4f}")

    # --- Save results inside run folder ---
    out_path = run_dir / Path(args.results_json).name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}", flush=True)

    # --- Qualitative sample (pretrained captions) ---
    sample_path = run_dir / "captions_sample.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("img_id\timage_file\tgenerated_caption\tfirst_reference\n")
        for row in pre_samples:
            f.write("\t".join(str(x) for x in row) + "\n")
    print(f"Caption sample  → {sample_path}")

    # --- Optional MLflow logging (lazy import) ---
    if args.use_mlflow:
        import mlflow
        db = Path(args.output_dir).resolve() / "mlruns.db"
        mlflow.set_tracking_uri("sqlite:///" + str(db).replace("\\", "/"))
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=args.run_name):
            mlflow.log_params({
                "checkpoint":  args.checkpoint or "pretrained_only",
                "split":       args.split,
                "limit":       args.limit,
            })
            for k, v in pre_metrics.items():
                mlflow.log_metric(f"pretrained_{k}", v)
            if ft_metrics:
                for k, v in ft_metrics.items():
                    mlflow.log_metric(f"finetuned_{k}", v)


if __name__ == "__main__":
    main()
