"""Build structured semantics from an Artpedia record and a generated caption.

Returned dict shape:
    {
        "title":          str,
        "caption":        str,
        "year":           int | None,
        "period":         str,
        "suggested_tone": "calm" | "engaging",
        "source":         str,
    }

Tone heuristic:
  Paintings from before 1800 (Renaissance, Baroque, Old Masters) get "calm" —
  a slower, lower-pitched British voice suits measured, descriptive narration.
  Paintings from 1800 onward (Romanticism, Impressionism, Modern) get "engaging" —
  a warmer American voice with slightly elevated pitch suits emotionally charged work.
"""

from __future__ import annotations


# ----- period lookup --------------------------------------------------------

_PERIODS = [
    (0,    1400, "Medieval"),
    (1400, 1527, "Renaissance"),
    (1527, 1600, "Mannerism"),
    (1600, 1750, "Baroque"),
    (1750, 1820, "Neoclassicism"),
    (1820, 1870, "Romanticism"),
    (1870, 1910, "Impressionism"),
    (1910, 1945, "Modernism"),
    (1945, 1970, "Abstract Expressionism"),
    (1970, 9999, "Contemporary"),
]

_TONE_CUTOFF = 1800  # pre-cutoff → calm; cutoff and after → engaging


def _year_to_period(year: int) -> str:
    for start, end, name in _PERIODS:
        if start <= year < end:
            return name
    return "Unknown"


def _year_to_tone(year: int) -> str:
    return "calm" if year < _TONE_CUTOFF else "engaging"


# ----- public API -----------------------------------------------------------

def build_semantics(record: dict, caption: str) -> dict:
    """Assemble structured semantics for one painting.

    Args:
        record:  A single Artpedia record dict (must have at minimum 'title').
        caption: Generated caption string from the BLIP captioner.

    Returns:
        A flat dict suitable for JSON serialisation and TTS selection.
    """
    title = record.get("title", "Untitled")
    year  = record.get("year")

    if year is not None:
        try:
            year = int(year)
        except (ValueError, TypeError):
            year = None

    period         = _year_to_period(year) if year is not None else "Unknown"
    suggested_tone = _year_to_tone(year)   if year is not None else "calm"
    source         = record.get("img_url", "")

    return {
        "title":          title,
        "caption":        caption,
        "year":           year,
        "period":         period,
        "suggested_tone": suggested_tone,
        "source":         source,
    }
