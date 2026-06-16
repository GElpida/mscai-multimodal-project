"""Diagnostic: show which BLIP submodules are trainable vs frozen after freezing
the vision encoder (same logic as train_caption.py). Read-only — no training."""

import sys
from pathlib import Path

# Allow "python src/inspect_model.py" from the repo root.
sys.path.insert(0, str(Path(__file__).parent))
from train_caption import freeze_vision_encoder


def param_counts(module):
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def main():
    from lavis.models import load_model_and_preprocess

    print("Loading blip_caption / base_coco on CPU...")
    model, _, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=False
    )

    # Apply the same freezing logic used in train_caption.py.
    freeze_vision_encoder(model)

    print()
    print(f"{'Module':<30} {'Total':>12} {'Trainable':>12}  Status")
    print("-" * 62)

    for name, child in model.named_children():
        total, trainable = param_counts(child)
        frozen = total - trainable
        if trainable == 0:
            status = "FROZEN"
        elif frozen == 0:
            status = "trainable"
        else:
            status = "mixed"
        print(f"{name:<30} {total:>12,} {trainable:>12,}  {status}")

    print("-" * 62)
    total_all,     trainable_all     = param_counts(model)
    print(f"{'TOTAL':<30} {total_all:>12,} {trainable_all:>12,}")
    print(f"  Frozen : {total_all - trainable_all:,}")
    print(f"  Trainable: {trainable_all:,}")


if __name__ == "__main__":
    main()
