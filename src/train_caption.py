import argparse
import datetime
import json
import sys
import time
import warnings
from pathlib import Path

# Suppress FutureWarnings from fairscale's internal AMP wrappers (third-party,
# not fixable from our side; harmless on PyTorch 2.x).
warnings.filterwarnings("ignore", category=FutureWarning, module="fairscale")

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Allow "python src/train_caption.py" from the repo root.
sys.path.insert(0, str(Path(__file__).parent))
from caption_dataset import CaptionDataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000


def collate_fn(batch):
    """Stack image tensors; collect captions as a plain list of strings."""
    images = torch.stack([item["image"] for item in batch])
    captions = [item["text_input"] for item in batch]
    return {"image": images, "text_input": captions}


# ── Local defaults — edit these to run without any CLI flags ──────────────────
MANIFEST     = r"G:\My Drive\Github\mscai-multimodal-project\data\processed\train.jsonl"
VAL_MANIFEST = None
BASE_DIR     = r"G:\My Drive\Github\mscai-multimodal-project\data"
OUTPUT_DIR   = "outputs"
EPOCHS       = 100
PATIENCE     = 10      # stop if monitored loss does not improve for this many epochs
MIN_DELTA    = 0.0     # minimum improvement to reset patience counter
BATCH_SIZE   = 8
LR           = 1e-5
WEIGHT_DECAY = 0.0
UNFREEZE_N   = 4       # number of decoder transformer layers to keep trainable
FP16         = False   # set False if running on CPU
NUM_WORKERS  = 0       # keep 0 on Windows
SEED         = None    # set an int for reproducibility
USE_MLFLOW   = True
MLFLOW_EXP   = "blip-artpedia"
RUN_NAME     = None    # optional string label for the MLflow run
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune BLIP captioning on Artpedia.")
    p.add_argument("--manifest",     default=MANIFEST,
                   help="Path to train .jsonl manifest")
    p.add_argument("--val-manifest", default=VAL_MANIFEST,
                   help="Path to validation .jsonl manifest (optional; enables val-loss tracking)")
    p.add_argument("--base-dir",     default=BASE_DIR,
                   help="Base dir for resolving relative image paths")
    p.add_argument("--output-dir",   default=OUTPUT_DIR,
                   help="Root checkpoint directory; a timestamped subfolder is created per run")
    p.add_argument("--epochs",       type=int,   default=EPOCHS)
    p.add_argument("--patience",     type=int,   default=PATIENCE,
                   help="Early-stopping patience: stop after this many epochs with no improvement")
    p.add_argument("--min-delta",    type=float, default=MIN_DELTA,
                   help="Minimum loss decrease to count as an improvement (default: 0.0)")
    p.add_argument("--batch-size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                   help="AdamW weight decay (default: 0.0)")
    p.add_argument("--seed",         type=int,   default=SEED,
                   help="Random seed for reproducibility (optional)")
    p.add_argument("--freeze-vision", action=argparse.BooleanOptionalAction, default=True,
                   help="Freeze BLIP vision encoder (default: True)")
    p.add_argument("--fp16",          action=argparse.BooleanOptionalAction, default=FP16,
                   help="fp16 mixed precision on CUDA (use --no-fp16 to disable)")
    p.add_argument("--num-workers",   type=int, default=NUM_WORKERS,
                   help="DataLoader worker processes (default: 0, Windows-safe)")
    p.add_argument("--unfreeze-last-n", type=int, default=UNFREEZE_N,
                   help="Train only the last N decoder transformer layers + cls head")
    p.add_argument("--use-mlflow",        action="store_true", default=USE_MLFLOW,
                   help="Enable MLflow experiment tracking")
    p.add_argument("--mlflow-experiment", default=MLFLOW_EXP,
                   help="MLflow experiment name")
    p.add_argument("--run-name",          default=RUN_NAME,
                   help="MLflow run name (optional)")
    return p.parse_args()


def freeze_vision_encoder(model):
    """Freeze the vision encoder; detect its attribute name robustly."""
    encoder = getattr(model, "visual_encoder", None)
    if encoder is None:
        candidates = [(n, m) for n, m in model.named_children()
                      if "visual" in n.lower() or "vision" in n.lower()]
        if not candidates:
            print("  Top-level module names:", [n for n, _ in model.named_children()])
            raise RuntimeError("Cannot identify vision encoder — see names above.")
        name, encoder = candidates[0]
        print(f"  Vision encoder found as model.{name}")
    for param in encoder.parameters():
        param.requires_grad = False
    frozen    = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Vision encoder frozen ({frozen:,} params). Trainable: {trainable:,} params.")


def freeze_decoder_partial(model, n):
    """Freeze early decoder layers; keep the last n layers and the cls head trainable.

    Freezes: text_decoder.bert.embeddings + transformer layers [0 .. total-n-1].
    Trains:  transformer layers [total-n .. total-1] + text_decoder.cls.
    Returns (frozen_layer_indices, trainable_layer_indices).
    """
    decoder = model.text_decoder

    # Step 1 — freeze embeddings.
    try:
        for p in decoder.bert.embeddings.parameters():
            p.requires_grad = False
    except AttributeError:
        print("  [WARN] text_decoder.bert.embeddings not found — skipping.")

    # Step 2 — locate the transformer layer ModuleList dynamically.
    try:
        layer_list = decoder.bert.encoder.layer
    except AttributeError:
        print("  text_decoder children:", [nm for nm, _ in decoder.named_children()])
        print("ERROR: text_decoder.bert.encoder.layer not found — see structure above.")
        sys.exit(1)

    total       = len(layer_list)
    freeze_upto = max(0, total - n)

    for i, layer in enumerate(layer_list):
        for p in layer.parameters():
            p.requires_grad = (i >= freeze_upto)

    frozen_layers    = list(range(freeze_upto))
    trainable_layers = list(range(freeze_upto, total))

    if n >= total:
        print(f"  --unfreeze-last-n {n} >= {total} total layers: full decoder trainable.")
    else:
        print(f"  Decoder layers frozen   : {frozen_layers}")
        print(f"  Decoder layers trainable: {trainable_layers}")
    print("  text_decoder.cls: trainable")
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params — total: {total_p:,}  trainable: {train_p:,}  frozen: {total_p - train_p:,}")

    return frozen_layers, trainable_layers


def run_validation(model, val_loader, device, use_amp):
    """Evaluate model over val_loader; return average loss without updating weights."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images   = batch["image"].to(device)
            captions = batch["text_input"]
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out  = model({"image": images, "text_input": captions})
                loss = out["loss"]
            total_loss += loss.item()
    model.train()
    return total_loss / len(val_loader)


def write_reports(run_dir, run_info):
    """Write report.json and report.md into run_dir."""
    # ── JSON ──────────────────────────────────────────────────────────────────
    with open(run_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    # ── Markdown ──────────────────────────────────────────────────────────────
    hp    = run_info["hyperparameters"]
    ds    = run_info["dataset"]
    mi    = run_info["model"]
    summ  = run_info["summary"]
    ts    = run_info["start_time"]

    def _dur(secs):
        if secs is None:
            return "—"
        return f"{secs / 3600:.2f} h" if secs >= 3600 else f"{secs / 60:.1f} min"

    lines = [
        f"# Training Run — {ts}",
        "",
        "## Hyperparameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| epochs | {hp['epochs']} |",
        f"| batch_size | {hp['batch_size']} |",
        f"| lr | {hp['lr']} |",
        f"| weight_decay | {hp['weight_decay']} |",
        f"| unfreeze_last_n | {hp['unfreeze_last_n']} |",
        f"| freeze_vision | {hp['freeze_vision']} |",
        f"| fp16 | {hp['fp16']} |",
        f"| patience | {hp['patience']} |",
        f"| min_delta | {hp['min_delta']} |",
        f"| seed | {hp['seed']} |",
        f"| num_workers | {hp['num_workers']} |",
        "",
        "## Dataset",
        "",
        "| Split | Size | Manifest |",
        "|-------|------|----------|",
        f"| train | {ds['train_size']} | `{ds['train_manifest']}` |",
    ]
    if ds["val_manifest"]:
        lines.append(f"| val | {ds['val_size']} | `{ds['val_manifest']}` |")

    lines += [
        "",
        "## Model",
        "",
        "| Stat | Value |",
        "|------|-------|",
        f"| total_params | {mi['total_params']:,} |",
        f"| trainable_params | {mi['trainable_params']:,} |",
        f"| frozen_params | {mi['frozen_params']:,} |",
        f"| trainable_decoder_layers | {mi['trainable_decoder_layers']} |",
        f"| cls_trainable | {mi['cls_trainable']} |",
        "",
        "## Training Log",
        "",
        "| epoch | train_loss | val_loss | seconds |",
        "|-------|-----------|----------|---------|",
    ]
    for rec in run_info["epochs"]:
        val_str = f"{rec['val_loss']:.4f}" if rec["val_loss"] is not None else "—"
        lines.append(
            f"| {rec['epoch']} | {rec['train_loss']:.4f} | {val_str} | {rec['duration_s']:.1f} |"
        )

    # Summary section.
    best_loss_label, best_loss_val = (
        ("best val loss",   summ["best_val_loss"])
        if summ["best_val_loss"] is not None
        else ("best train loss", summ["best_train_loss"])
    )
    best_loss_str = f"{best_loss_val:.4f}" if best_loss_val is not None else "—"
    final_str     = f"{summ['final_train_loss']:.4f}" if summ["final_train_loss"] is not None else "—"
    avg_str       = f"{summ['avg_train_loss']:.4f}"   if summ["avg_train_loss"]   is not None else "—"
    es_str        = (
        f"yes — epoch {summ['early_stopped_epoch']}" if summ["early_stopped"] else "no"
    )

    lines += [
        "",
        "## Summary",
        "",
        f"- **Total duration:** {_dur(summ['total_duration_s'])}",
        f"- **Best epoch:** {summ['best_epoch']}",
        f"- **{best_loss_label.capitalize()}:** {best_loss_str}",
        f"- **Final train loss:** {final_str}",
        f"- **Avg train loss:** {avg_str}",
        f"- **Early stopping:** {es_str}",
        f"- **Start:** {run_info['start_time']}",
        f"- **End:** {run_info['end_time'] or '—'}",
        "",
    ]

    with open(run_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Reports written → {run_dir / 'report.json'}")
    print(f"             → {run_dir / 'report.md'}")


def main():
    args   = parse_args()

    # Set BEFORE any import that might pull in mlflow (e.g. LAVIS → omegaconf → mlflow).
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Optional reproducibility seed.
    if args.seed is not None:
        import random
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seed: {args.seed}")

    # ── Per-run timestamped folder ─────────────────────────────────────────────
    start_dt  = datetime.datetime.now()
    ts_str    = start_dt.strftime("%Y-%m-%d_%H-%M-%S")
    start_ts  = start_dt.isoformat(timespec="seconds")
    run_dir   = Path(args.output_dir) / f"train_{ts_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {run_dir}")

    # Load BLIP model and LAVIS processors.
    from lavis.models import load_model_and_preprocess
    print("Loading BLIP model...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=False, device=device
    )

    # ── Train dataset ──────────────────────────────────────────────────────────
    dataset = CaptionDataset(
        args.manifest,
        vis_processor=vis_processors["train"],
        txt_processor=txt_processors["train"],
        base_dir=args.base_dir,
    )
    print(f"Train dataset: {len(dataset)} records")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ── Optional validation dataset ───────────────────────────────────────────
    val_dataset = val_loader = None
    if args.val_manifest:
        val_dataset = CaptionDataset(
            args.val_manifest,
            vis_processor=vis_processors["eval"],
            txt_processor=txt_processors["eval"],
            base_dir=args.base_dir,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        print(f"Val dataset:   {len(val_dataset)} records")

    # ── Freeze strategy ───────────────────────────────────────────────────────
    if args.freeze_vision:
        freeze_vision_encoder(model)

    frozen_layers, trainable_layers = freeze_decoder_partial(model, args.unfreeze_last_n)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_amp = bool(args.fp16 and device.type == "cuda")
    scaler  = torch.amp.GradScaler(device.type, enabled=use_amp)

    total_p    = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Initialise run record ─────────────────────────────────────────────────
    run_info = {
        "start_time": start_ts,
        "end_time":   None,
        "run_folder": str(run_dir),
        "hyperparameters": {
            "epochs":          args.epochs,
            "batch_size":      args.batch_size,
            "lr":              args.lr,
            "weight_decay":    args.weight_decay,
            "unfreeze_last_n": args.unfreeze_last_n,
            "freeze_vision":   args.freeze_vision,
            "fp16":            args.fp16,
            "patience":        args.patience,
            "min_delta":       args.min_delta,
            "seed":            args.seed,
            "num_workers":     args.num_workers,
        },
        "dataset": {
            "train_manifest": args.manifest,
            "train_size":     len(dataset),
            "val_manifest":   args.val_manifest,
            "val_size":       len(val_dataset) if val_dataset else None,
        },
        "model": {
            "total_params":             total_p,
            "trainable_params":         trainable_p,
            "frozen_params":            total_p - trainable_p,
            "trainable_decoder_layers": trainable_layers,
            "frozen_decoder_layers":    frozen_layers,
            "cls_trainable":            True,
        },
        "epochs":  [],
        "summary": {
            "total_duration_s":    None,
            "best_epoch":          None,
            "best_val_loss":       None,
            "best_train_loss":     None,
            "final_train_loss":    None,
            "avg_train_loss":      None,
            "early_stopped":       False,
            "early_stopped_epoch": None,
        },
    }

    # ── MLflow setup ──────────────────────────────────────────────────────────
    if args.use_mlflow:
        import mlflow
        db = Path(args.output_dir).resolve() / "mlruns.db"
        mlflow.set_tracking_uri("sqlite:///" + str(db).replace("\\", "/"))
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow_run = mlflow.start_run(run_name=args.run_name)
        mlflow.log_params({
            "epochs":           args.epochs,
            "batch_size":       args.batch_size,
            "lr":               args.lr,
            "weight_decay":     args.weight_decay,
            "unfreeze_last_n":  args.unfreeze_last_n,
            "freeze_vision":    args.freeze_vision,
            "fp16":             args.fp16,
            "trainable_params": trainable_p,
            "dataset_size":     len(dataset),
            "patience":         args.patience,
            "min_delta":        args.min_delta,
            "seed":             args.seed,
            "run_folder":       str(run_dir),
        })

    LOG_EVERY        = 10
    best_loss        = float("inf")
    patience_counter = 0
    best_ckpt        = run_dir / "blip_artpedia_best.pth"
    train_start      = time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start  = time.time()
            model.train()
            running_loss = 0.0
            total_loss   = 0.0

            for step, batch in enumerate(loader, start=1):
                images   = batch["image"].to(device)
                captions = batch["text_input"]

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out  = model({"image": images, "text_input": captions})
                    loss = out["loss"]

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                total_loss   += loss.item()

                if step % LOG_EVERY == 0:
                    print(f"  Epoch {epoch} | step {step}/{len(loader)} | loss {running_loss / LOG_EVERY:.4f}")
                    running_loss = 0.0

            epoch_loss  = total_loss / len(loader)
            epoch_secs  = time.time() - epoch_start

            # Optional validation pass.
            val_loss = None
            if val_loader is not None:
                val_loss = run_validation(model, val_loader, device, use_amp)
                print(f"Epoch {epoch} complete — train: {epoch_loss:.4f}  val: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch} complete — avg loss: {epoch_loss:.4f}")

            # Record epoch.
            run_info["epochs"].append({
                "epoch":      epoch,
                "train_loss": round(epoch_loss, 6),
                "val_loss":   round(val_loss, 6) if val_loss is not None else None,
                "duration_s": round(epoch_secs, 2),
            })

            # Per-epoch checkpoint (inside run_dir).
            ckpt = run_dir / f"blip_artpedia_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"Checkpoint saved: {ckpt}")

            # MLflow per-epoch metrics.
            if args.use_mlflow:
                metrics = {"train_loss": epoch_loss}
                if val_loss is not None:
                    metrics["val_loss"] = val_loss
                mlflow.log_metrics(metrics, step=epoch)
                mlflow.set_tag(f"checkpoint_epoch_{epoch}", str(ckpt))

            # Early stopping — monitors val_loss when available, else train_loss.
            monitor = val_loss if val_loss is not None else epoch_loss
            if monitor < best_loss - args.min_delta:
                best_loss        = monitor
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt)
                run_info["summary"]["best_epoch"]      = epoch
                run_info["summary"]["best_val_loss"]   = (
                    round(val_loss, 6) if val_loss is not None else None
                )
                run_info["summary"]["best_train_loss"] = round(epoch_loss, 6)
                print(f"  ↓ New best {'val' if val_loss is not None else 'train'} loss "
                      f"{best_loss:.4f} — best checkpoint updated.")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{args.patience})")
                if patience_counter >= args.patience:
                    print(f"Early stopping after epoch {epoch} (patience={args.patience}).")
                    run_info["summary"]["early_stopped"]       = True
                    run_info["summary"]["early_stopped_epoch"] = epoch
                    if args.use_mlflow:
                        mlflow.set_tag("early_stopped_epoch", str(epoch))
                    break

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n[OOM] CUDA ran out of memory. Try a smaller --batch-size or enable --fp16.")
            sys.exit(1)
        raise
    finally:
        # Write reports whenever at least one epoch completed (including OOM / early stop).
        if run_info["epochs"]:
            end_dt = datetime.datetime.now()
            run_info["end_time"] = end_dt.isoformat(timespec="seconds")
            total_dur            = time.time() - train_start
            train_losses         = [r["train_loss"] for r in run_info["epochs"]]
            run_info["summary"]["total_duration_s"] = round(total_dur, 2)
            run_info["summary"]["final_train_loss"] = train_losses[-1]
            run_info["summary"]["avg_train_loss"]   = round(
                sum(train_losses) / len(train_losses), 6
            )
            write_reports(run_dir, run_info)

        if args.use_mlflow:
            mlflow.end_run()


if __name__ == "__main__":
    main()
