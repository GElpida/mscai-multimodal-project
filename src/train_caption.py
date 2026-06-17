import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Allow "python src/train_caption.py" from the repo root.
sys.path.insert(0, str(Path(__file__).parent))
from caption_dataset import CaptionDataset


def collate_fn(batch):
    """Stack image tensors; collect captions as a plain list of strings."""
    images = torch.stack([item["image"] for item in batch])
    captions = [item["text_input"] for item in batch]
    return {"image": images, "text_input": captions}


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune BLIP captioning on Artpedia.")
    p.add_argument("--manifest",   required=True, help="Path to train .jsonl manifest")
    p.add_argument("--base-dir",   required=True, help="Base dir for resolving relative image paths")
    p.add_argument("--output-dir", default="checkpoints", help="Checkpoint output directory (default: checkpoints)")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=1e-5)
    p.add_argument("--freeze-vision", action=argparse.BooleanOptionalAction, default=True,
                   help="Freeze BLIP vision encoder (default: True; use --no-freeze-vision to unfreeze)")
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                   help="fp16 mixed precision on CUDA (default: True; use --no-fp16 to disable)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader worker processes (default: 0, Windows-safe)")
    p.add_argument("--unfreeze-last-n", type=int, default=2,
                   help="Train only the last N decoder transformer layers + cls head (default: 2)")
    p.add_argument("--use-mlflow",        action="store_true", default=False,
                   help="Enable MLflow experiment tracking")
    p.add_argument("--mlflow-experiment", default="blip-artpedia",
                   help="MLflow experiment name (default: blip-artpedia)")
    p.add_argument("--run-name",          default=None,
                   help="MLflow run name (optional)")
    return p.parse_args()


def freeze_vision_encoder(model):
    """Freeze the vision encoder; detect its attribute name robustly."""
    encoder = getattr(model, "visual_encoder", None)
    if encoder is None:
        # Fall back: search top-level submodules for visual/vision names.
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
    If n >= total layers the full decoder stays trainable (only embeddings are frozen).
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
    freeze_upto = max(0, total - n)   # layers [0, freeze_upto) frozen; [freeze_upto, total) trained

    for i, layer in enumerate(layer_list):
        for p in layer.parameters():
            p.requires_grad = (i >= freeze_upto)

    # Step 3 — cls head: leave requires_grad=True (unchanged from model load).

    # Summary.
    if n >= total:
        print(f"  --unfreeze-last-n {n} >= {total} total layers: full decoder trainable.")
    else:
        print(f"  Decoder layers frozen   : {list(range(freeze_upto))}")
        print(f"  Decoder layers trainable: {list(range(freeze_upto, total))}")
    print("  text_decoder.cls: trainable")
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params — total: {total_p:,}  trainable: {train_p:,}  frozen: {total_p - train_p:,}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load BLIP model and LAVIS processors.
    from lavis.models import load_model_and_preprocess
    print("Loading BLIP model...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=False, device=device
    )

    # Build dataset and DataLoader.
    dataset = CaptionDataset(
        args.manifest,
        vis_processor=vis_processors["train"],
        txt_processor=txt_processors["train"],
        base_dir=args.base_dir,
    )
    print(f"Dataset: {len(dataset)} records")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Optionally freeze the vision encoder so only the text decoder trains.
    if args.freeze_vision:
        freeze_vision_encoder(model)

    # Partial decoder freeze: keep only the last N layers + cls head trainable.
    freeze_decoder_partial(model, args.unfreeze_last_n)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    use_amp = args.fp16 and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Lazy MLflow setup — only imported when --use-mlflow is set.
    if args.use_mlflow:
        import mlflow
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow_run = mlflow.start_run(run_name=args.run_name)
        mlflow.log_params({
            "epochs":           args.epochs,
            "batch_size":       args.batch_size,
            "lr":               args.lr,
            "unfreeze_last_n":  args.unfreeze_last_n,
            "freeze_vision":    args.freeze_vision,
            "fp16":             args.fp16,
            "trainable_params": trainable_params,
            "dataset_size":     len(dataset),
        })

    LOG_EVERY = 10  # print running-average loss every N steps

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            total_loss   = 0.0

            for step, batch in enumerate(loader, start=1):
                images   = batch["image"].to(device)
                captions = batch["text_input"]       # list of strings, stays on CPU

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=use_amp):
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

            epoch_loss = total_loss / len(loader)
            print(f"Epoch {epoch} complete — avg loss: {epoch_loss:.4f}")

            ckpt = output_dir / f"blip_artpedia_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"Checkpoint saved: {ckpt}")

            if args.use_mlflow:
                mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                mlflow.set_tag(f"checkpoint_epoch_{epoch}", str(ckpt))

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n[OOM] CUDA ran out of memory. Try a smaller --batch-size or enable --fp16.")
            sys.exit(1)
        raise
    finally:
        if args.use_mlflow:
            mlflow.end_run()


if __name__ == "__main__":
    main()
