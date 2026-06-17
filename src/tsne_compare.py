"""t-SNE comparison of BLIP decoder representations: pretrained vs fine-tuned.

WHY we go through the decoder:
  The vision encoder is frozen, so raw image embeddings are IDENTICAL for both
  models — a t-SNE on them would show no difference. Instead we feed each image
  embed as cross-attention context into text_decoder.bert (the fine-tuned BERT
  decoder) using a single BOS token as input. The decoder's last hidden state
  carries information shaped by the fine-tuned cross-attention weights, making
  the two sets of vectors meaningfully comparable.

Extraction:
  image_embeds = model.visual_encoder(image)          # frozen, same both models
  outputs = model.text_decoder.bert(                  # fine-tuned weights differ
      input_ids=[BOS],
      encoder_hidden_states=image_embeds,             # cross-attention
      encoder_attention_mask=ones,
  )
  feature = outputs.last_hidden_state.mean(dim=1)     # mean-pool over tokens
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from artpedia_dataset import ArtpediaDataset

ARTPEDIA_DEFAULT = str(
    Path(__file__).parent.parent / "dataset" / "artpedia" / "artpedia.json"
)


def parse_args():
    p = argparse.ArgumentParser(description="t-SNE of BLIP decoder representations.")
    p.add_argument("--manifest",   required=True)
    p.add_argument("--base-dir",   required=True)
    p.add_argument("--artpedia",   default=ARTPEDIA_DEFAULT)
    p.add_argument("--checkpoint", required=True, help="Fine-tuned state_dict .pth")
    p.add_argument("--split",      default="test")
    p.add_argument("--limit",      type=int, default=0, help="Max images (0=all)")
    p.add_argument("--output",     default="tsne_compare.png")
    return p.parse_args()


def load_eval_data(args):
    """Build list of (img_path, year) from manifest + artpedia."""
    artpedia_ds = ArtpediaDataset(args.artpedia, split=args.split)
    base_dir    = Path(args.base_dir)
    data = []

    with open(args.manifest, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                idx   = int(Path(entry["image_path"]).stem)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

            if idx >= len(artpedia_ds):
                continue

            record = artpedia_ds[idx]
            year   = record.get("year")
            if year is None:
                continue

            img_path = base_dir / Path(entry["image_path"])
            if not img_path.exists():
                print(f"  [WARN] Missing: {img_path} — skipping.")
                continue

            data.append((img_path, int(year)))
            if args.limit > 0 and len(data) >= args.limit:
                break

    return data


def extract_features(model, vis_processor, eval_data, device):
    """Return (N, hidden_dim) numpy array of decoder cross-attention features."""
    features = []
    model.eval()

    with torch.no_grad():
        for img_path, _ in eval_data:
            image        = Image.open(img_path).convert("RGB")
            image_tensor = vis_processor(image).unsqueeze(0).to(device)

            # Vision encoding — frozen, identical for both models.
            image_embeds = model.visual_encoder(image_tensor)
            image_atts   = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=device)

            # Single BOS token fed through the fine-tuned decoder with cross-attention.
            bos      = model.tokenizer.bos_token_id
            inp_ids  = torch.tensor([[bos]], dtype=torch.long, device=device)

            out = model.text_decoder.bert(
                input_ids=inp_ids,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # Mean-pool last_hidden_state over token dim → [hidden_dim].
            feat = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            features.append(feat)

    return torch.stack(features).numpy()


def century_label(year):
    """Map year → century string, e.g. 1480 → '15th c.'"""
    c = (year - 1) // 100 + 1
    if 11 <= c % 100 <= 13:
        sfx = "th"
    elif c % 10 == 1:
        sfx = "st"
    elif c % 10 == 2:
        sfx = "nd"
    elif c % 10 == 3:
        sfx = "rd"
    else:
        sfx = "th"
    return f"{c}{sfx} c."


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    eval_data = load_eval_data(args)
    print(f"Loaded {len(eval_data)} images.")
    if len(eval_data) < 5:
        print("Too few images for t-SNE — check manifest and base-dir.")
        sys.exit(1)

    from lavis.models import load_model_and_preprocess

    # Pretrained features.
    print("Loading pretrained model...")
    model_pre, vis_pre, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )
    print("Extracting pretrained features...")
    feats_pre = extract_features(model_pre, vis_pre["eval"], eval_data, device)

    # Fine-tuned features.
    print(f"Loading fine-tuned checkpoint: {args.checkpoint}")
    model_ft, vis_ft, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )
    model_ft.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model_ft.eval()
    print("Extracting fine-tuned features...")
    feats_ft = extract_features(model_ft, vis_ft["eval"], eval_data, device)

    # Century labels and color mapping.
    labels          = [century_label(y) for _, y in eval_data]
    unique_centuries = sorted(set(labels),
                              key=lambda s: int("".join(filter(str.isdigit, s))))
    cmap       = plt.cm.tab20(np.linspace(0, 1, len(unique_centuries)))
    color_map  = {c: cmap[i] for i, c in enumerate(unique_centuries)}
    point_colors = [color_map[c] for c in labels]

    # t-SNE (run separately — each set has its own coordinate system).
    perp = min(30, len(eval_data) - 1)
    print("Running t-SNE...")
    tsne_pre = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(feats_pre)
    tsne_ft  = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(feats_ft)

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("BLIP Decoder Representations — Pretrained vs Fine-tuned (Artpedia test)",
                 fontsize=12)

    for ax, coords, title in [
        (axes[0], tsne_pre, "Pretrained"),
        (axes[1], tsne_ft,  "Fine-tuned"),
    ]:
        ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=40, alpha=0.8, linewidths=0)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[c], markersize=8, label=c)
        for c in unique_centuries
    ]
    fig.legend(handles=handles, title="Century", loc="lower center",
               ncol=min(len(unique_centuries), 7), bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    out_path = Path(args.output)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
