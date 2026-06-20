"""
clean_captions.py — Remove contextual (non-visual) sentences from Artpedia manifests.

Rule-based cleaner that strips attribution, provenance, and cataloguing sentences
from the "caption" field of a .jsonl manifest, keeping only visually descriptive
content.  An optional --use-ner pass (spaCy) and an optional inscription rule group
can be enabled separately once the core rules are validated.

Usage:
    # Dry-run first — see what would be removed without writing anything:
    python src/clean_captions.py --input data/processed/train.jsonl --dry-run

    # Full run:
    python src/clean_captions.py \\
        --input  data/processed/train.jsonl \\
        --output data/processed/train_clean.jsonl

    # Also enable inscription rules (signed / inscribed):
    python src/clean_captions.py --input ... --output ... --include-inscription-rules

    # Also enable spaCy NER pass:
    python src/clean_captions.py --input ... --output ... --use-ner
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Rule patterns — CORE (always active)
# ---------------------------------------------------------------------------
# Format: (label, regex_string)
# A sentence is CONTEXTUAL (removed) if it matches ANY pattern (re.search,
# case-insensitive).  Label is free-form; it appears in the report.
#
# Design principle: prefer tighter patterns that require context around the
# keyword so that words like "gallery", "collection", or "museum" are only
# caught when clearly provenance, not when describing depicted content.
# ---------------------------------------------------------------------------
_RULES_CORE = [

    # ── Attribution ─────────────────────────────────────────────────────────
    # These caught artist-name sentences well in initial testing — keep as-is.
    ("attribution",  r"\bpainting by\b"),
    ("attribution",  r"\bpainted by\b"),
    # REMOVED: r"\bis an? (?:oil |watercolour |watercolor |tempera )?painting\b"
    # Reason: "is an oil painting" is a medium description (visual content).
    # The authorship case ("is an oil painting BY <artist>") is already caught
    # by "painting by" / "painted by" / "by the ... artist" above.
    ("attribution",  r"\bwork by\b"),
    ("attribution",  r"\bby the\b.{0,50}\bartist\b"),   # "by the Dutch artist ..."
    ("attribution",  r"\bcreated by\b"),
    ("attribution",  r"\battributed to\b"),
    ("attribution",  r"\bdrawn by\b"),
    ("attribution",  r"\bengraved? by\b"),

    # ── Provenance / physical location ──────────────────────────────────────
    # "now in/at" are safe: they almost always indicate current physical location.
    ("provenance",   r"\bnow in\b"),
    ("provenance",   r"\bnow at\b"),
    ("provenance",   r"\bhoused in\b"),
    ("provenance",   r"\bon (?:permanent )?display\b"),
    ("provenance",   r"\bon loan\b"),
    ("provenance",   r"\bbequeathed\b"),
    ("provenance",   r"\bpurchased (?:by|from)\b"),

    # Tightened: require "acquired/donated by/from/to" not bare word.
    ("provenance",   r"\bacquired (?:by|from)\b"),
    ("provenance",   r"\bdonated (?:to|by)\b"),

    # Tightened: require "exhibited at/in" — bare "exhibited" was too broad.
    ("provenance",   r"\bexhibited (?:at|in)\b"),
    ("provenance",   r"\bfirst (?:exhibited|shown) (?:at|in)\b"),

    # Tightened: require "in/at [the] museum/gallery" rather than bare word.
    # This avoids catching "a gallery of saints" or "museum-quality brushwork".
    ("provenance",   r"\b(?:in|at|to) (?:the |a |this )?museum\b"),
    ("provenance",   r"\b(?:in|at|to) (?:the |a |this )?(?:art |national |royal |city |public )?gallery\b"),

    # Tightened: "collection" only in clearly provenance phrasing.
    ("provenance",   r"\bprivate collection\b"),
    ("provenance",   r"\b(?:royal|national|permanent|public|civic|state) collection\b"),
    ("provenance",   r"\bin (?:the |a )?collection of\b"),
    ("provenance",   r"\bfrom (?:the |a )?.{0,30}collection\b"),

    # Tightened "located in": only when followed by a known venue type.
    # Avoids "located in a garden" or "located in the foreground".
    ("provenance",   r"\blocated in (?:the |a )?(?:museum|gallery|church|cathedral|basilica|chapel|palace|palazzo|collection|archive)\b"),

    # ── Cataloguing / dates of record ───────────────────────────────────────
    ("cataloguing",  r"\bcirca\b"),

    # Tightened: "dated" followed by a 4-digit year — avoids "dated manuscript".
    ("cataloguing",  r"\bdated (?:circa |c\.\s*)?\d{4}\b"),

    # Year-in-context patterns (almost never visual in art captions).
    ("cataloguing",  r"\bin 1[5-9]\d\d\b"),     # "in 1889", "in 1534"
    ("cataloguing",  r"\bin 20[0-2]\d\b"),       # "in 2005" (modern accessions)

    # REMOVED: r"\boil on (?:canvas|panel|board|wood|copper|linen)\b"
    # Reason: "oil on canvas / panel / wood" is a medium description — visual content.
    # Attribution sentences that mention the medium are caught by "painting by" etc.
    ("cataloguing",  r"\bmeasures?\b"),           # "measures 80 × 60 cm"
    ("cataloguing",  r"\bdimensions?\b"),
    ("cataloguing",  r"\binventory number\b"),
    ("cataloguing",  r"\bcatalogue(?: raisonn[eé])?\b"),
]


# ---------------------------------------------------------------------------
# Rule patterns — INSCRIPTION (off by default, enable with --include-inscription-rules)
# ---------------------------------------------------------------------------
# "signed" and "inscribed" are AMBIGUOUS:
#   - "The work is signed SYMON DE SENIS ME PINXIT" → visible inscription on the
#     artwork, arguably visual content.
#   - "signed and dated 1534 in the lower left" → cataloguing detail.
# Keep OFF by default.  Enable only after validating core rules, and be prepared
# to see some false removals of sentences describing visible text.
# ---------------------------------------------------------------------------
_RULES_INSCRIPTION = [
    ("cataloguing-inscription", r"\bsigned\b"),
    ("cataloguing-inscription", r"\binscribed\b"),
]


# Compiled rule sets — rebuilt in main() based on CLI flags.
_COMPILED_RULES: list = []   # filled by _build_rules()


def _build_rules(include_inscription):
    """Compile the active rule set into _COMPILED_RULES."""
    global _COMPILED_RULES
    rules = _RULES_CORE + (_RULES_INSCRIPTION if include_inscription else [])
    _COMPILED_RULES = [
        (label, re.compile(pattern, re.IGNORECASE))
        for label, pattern in rules
    ]


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_sentences(text):
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------

def first_matching_rule(sentence):
    """Return (label, pattern_string, matched_text) for the first hit, or None."""
    for label, pattern in _COMPILED_RULES:
        m = pattern.search(sentence)
        if m:
            matched = m.group()
            # Truncate very long matches (e.g. from .{0,50} patterns).
            if len(matched) > 40:
                matched = matched[:37] + "..."
            return label, pattern.pattern, matched
    return None


# ---------------------------------------------------------------------------
# Optional NER pass (off by default)
# ---------------------------------------------------------------------------
_NER_ENTITIES = {"PERSON", "ORG", "GPE", "FAC"}


def load_spacy():
    """Lazily import spaCy and load an English model; return nlp or None."""
    try:
        import spacy  # noqa: PLC0415
    except ImportError:
        print(
            "[NER] spaCy not installed.  Run:\n"
            "        pip install spacy\n"
            "        python -m spacy download en_core_web_sm\n"
            "      Continuing with rule-based pass only.",
            file=sys.stderr,
        )
        return None

    for model in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            return spacy.load(model)
        except OSError:
            continue

    print(
        "[NER] No spaCy English model found.  Run:\n"
        "        python -m spacy download en_core_web_sm\n"
        "      Continuing with rule-based pass only.",
        file=sys.stderr,
    )
    return None


def has_contextual_entity(sentence, nlp):
    """Return True if spaCy finds a PERSON / ORG / GPE / FAC entity."""
    doc = nlp(sentence)
    return any(ent.label_ in _NER_ENTITIES for ent in doc.ents)


# ---------------------------------------------------------------------------
# Per-caption cleaning
# ---------------------------------------------------------------------------

def clean_caption(caption, use_ner, nlp):
    """
    Split caption into sentences, filter contextual ones, rejoin.

    Returns:
        cleaned   (str)   — rejoined kept sentences; empty if all were removed
        kept      (list of str)
        removed   (list of (sentence, label, pattern_str, matched_text))
    """
    sentences = split_sentences(caption)
    kept, removed = [], []

    for sent in sentences:
        hit = first_matching_rule(sent)
        if hit:
            label, pattern_str, matched_text = hit
            removed.append((sent, label, pattern_str, matched_text))
            continue

        if use_ner and nlp is not None:
            if has_contextual_entity(sent, nlp):
                removed.append((sent, "ner", "(spaCy entity)", ""))
                continue

        kept.append(sent)

    return " ".join(kept), kept, removed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Remove contextual sentences from Artpedia caption manifests."
    )
    p.add_argument("--input",   required=True,
                   help="Input .jsonl manifest (image_path, caption, title, year, ...)")
    p.add_argument("--output",  default=None,
                   help="Output .jsonl path — must differ from --input. "
                        "Required unless --dry-run is set.")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Report what WOULD be removed without writing any file.")
    p.add_argument("--include-inscription-rules", action="store_true", default=False,
                   help="Also apply signed/inscribed rules (off by default — "
                        "these are ambiguous: they catch cataloguing details but "
                        "may also remove sentences describing visible text).")
    p.add_argument("--use-ner", action="store_true", default=False,
                   help="Enable spaCy NER pass to catch entity-heavy sentences "
                        "(requires: pip install spacy && python -m spacy download en_core_web_sm).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _fmt_rule(pattern_str, width=45):
    """Truncate long regex strings for display."""
    s = pattern_str
    return s if len(s) <= width else s[:width - 3] + "..."


def build_report(
    total_in, total_out, dropped, dropped_titles,
    total_kept, total_removed,
    examples, pattern_counts,
    dry_run,
):
    total_sents = total_kept + total_removed
    pct = (total_removed / total_sents * 100) if total_sents else 0.0
    mode = "DRY-RUN — no files written" if dry_run else "summary"

    lines = [
        "=" * 66,
        f"  clean_captions.py — {mode}",
        "=" * 66,
        f"  Records in              : {total_in}",
        f"  Records out             : {total_out}",
        f"  Records dropped         : {dropped}  (caption fully contextual)",
        f"  Sentences kept          : {total_kept}",
        f"  Sentences removed       : {total_removed}  ({pct:.1f}% of total)",
        "",
    ]

    # Per-pattern breakdown
    lines.append("  Per-pattern breakdown (sentences removed):")
    if pattern_counts:
        max_count = max(pattern_counts.values())
        for (label, pat), count in pattern_counts.most_common():
            bar = "#" * max(1, round(count / max_count * 20))
            lines.append(
                f"    {label:<28s}  {count:>4d}  {bar}  {_fmt_rule(pat)}"
            )
    else:
        lines.append("    (no sentences removed)")
    lines.append("")

    # Example removed sentences (up to 5) with matched keyword
    lines.append("  Example removed sentences (up to 5):")
    for i, (label, matched_text, sent) in enumerate(examples, 1):
        preview = sent if len(sent) <= 105 else sent[:102] + "..."
        lines.append(f"    [{i}] ({label})  matched: '{matched_text}'")
        lines.append(f"         {preview}")
    if not examples:
        lines.append("    (none — all sentences passed the filters)")
    lines.append("")

    # Dropped record titles (up to 20)
    lines.append(f"  Dropped record titles (up to 20 of {dropped}):")
    if dropped_titles:
        for t in dropped_titles:
            lines.append(f"    • {t}")
    else:
        lines.append("    (none)")
    lines.append("=" * 66)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    in_path = Path(args.input)

    if not args.dry_run and args.output is None:
        print("[ERROR] --output is required unless --dry-run is set.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else None

    if out_path and in_path.resolve() == out_path.resolve():
        print("[ERROR] --input and --output must be different paths.", file=sys.stderr)
        sys.exit(1)
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # ── Build active rule set ────────────────────────────────────────────────
    _build_rules(args.include_inscription_rules)
    inscription_note = " + inscription rules" if args.include_inscription_rules else ""
    print(
        f"Rules: {len(_COMPILED_RULES)} patterns (core{inscription_note})"
        + (" + NER" if args.use_ner else ""),
        flush=True,
    )

    # ── Optional NER setup ───────────────────────────────────────────────────
    nlp = None
    if args.use_ner:
        print("Loading spaCy model for NER pass ...", flush=True)
        nlp = load_spacy()
        if nlp is None:
            print("NER pass disabled — falling back to rules only.", flush=True)

    # ── Open output (skip in dry-run) ────────────────────────────────────────
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fout = open(out_path, "w", encoding="utf-8")
    else:
        fout = None

    # ── Process records ──────────────────────────────────────────────────────
    total_in  = 0
    total_out = 0
    dropped   = 0
    total_kept    = 0
    total_removed = 0
    dropped_titles = []          # up to 20 record titles dropped entirely
    examples       = []          # up to 5 (label, matched_text, sentence)
    pattern_counts = Counter()   # (label, pattern_str) → count

    try:
        with open(in_path, encoding="utf-8-sig") as fin:
            for raw in fin:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as e:
                    print(f"  [WARN] Skipping malformed JSON: {e}", file=sys.stderr)
                    continue

                total_in += 1
                caption = record.get("caption", "")

                cleaned, kept, removed = clean_caption(caption, args.use_ner, nlp)

                total_kept    += len(kept)
                total_removed += len(removed)

                for sent, label, pattern_str, matched_text in removed:
                    pattern_counts[(label, pattern_str)] += 1
                    if len(examples) < 5:
                        examples.append((label, matched_text, sent))

                if not cleaned:
                    dropped += 1
                    if len(dropped_titles) < 20:
                        dropped_titles.append(record.get("title", "<no title>"))
                    continue

                if fout is not None:
                    record["caption"] = cleaned
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_out += 1
    finally:
        if fout is not None:
            fout.close()

    # ── Report ───────────────────────────────────────────────────────────────
    report = build_report(
        total_in, total_out, dropped, dropped_titles,
        total_kept, total_removed,
        examples, pattern_counts,
        args.dry_run,
    )
    print(report)

    if not args.dry_run and out_path is not None:
        report_path = out_path.with_name(out_path.name + ".report.txt")
        report_path.write_text(report + "\n", encoding="utf-8")
        print(f"\n  Output  → {out_path}")
        print(f"  Report  → {report_path}")


if __name__ == "__main__":
    main()
