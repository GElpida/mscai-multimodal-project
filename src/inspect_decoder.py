"""Diagnostic: reveal text_decoder's internal transformer-layer structure.
Read-only — no training, no saving, no file modifications."""

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))


def find_module_lists(module, prefix=""):
    """Recursively yield (dotted_path, ModuleList) for every ModuleList found."""
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.ModuleList):
            yield path, child
        else:
            yield from find_module_lists(child, path)


def main():
    from lavis.models import load_model_and_preprocess

    print("Loading blip_caption / base_coco on CPU...")
    model, _, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=False
    )

    decoder = model.text_decoder
    print(f"\nmodel.text_decoder type: {type(decoder).__name__}\n")

    # --- Top-level children of text_decoder ---
    print("Top-level named_children of text_decoder:")
    for name, child in decoder.named_children():
        params = sum(p.numel() for p in child.parameters())
        print(f"  .{name:<30} {type(child).__name__:<25} {params:>12,} params")

    # --- Find all ModuleLists (transformer layer stacks) ---
    print("\nModuleLists found inside text_decoder (transformer layer stacks):")
    lists = list(find_module_lists(decoder, prefix="text_decoder"))
    if not lists:
        print("  (none found)")
    for path, ml in lists:
        print(f"  {path}  →  {len(ml)} layers  [{type(ml[0]).__name__}]")

    # --- Show naming pattern for the first (and typically only) large ModuleList ---
    if lists:
        # Pick the largest ModuleList as the main transformer stack.
        path, ml = max(lists, key=lambda x: len(x[1]))
        print(f"\nMain transformer stack: {path}  ({len(ml)} layers)")
        print(f"  First layer path : {path}.0")
        print(f"  Last  layer path : {path}.{len(ml) - 1}")
        print(f"\nNamed children of {path}.0 (single transformer layer):")
        for name, child in ml[0].named_children():
            params = sum(p.numel() for p in child.parameters())
            print(f"    .{name:<30} {type(child).__name__:<20} {params:>10,} params")


if __name__ == "__main__":
    main()
