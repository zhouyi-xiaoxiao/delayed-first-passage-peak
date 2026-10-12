#!/usr/bin/env python3
"""Shared figure style and layout checks for the six figures of the paper.

Every figure script imports this module so that fonts, sizes, colours, panel labels and
output checks are identical across the six figures.

* Figures are drawn at final printed size (two-column journal text width 7.05 in, single column 3.4 in), so the
  font sizes below are the printed sizes.  Smallest base font: 6 pt.
* Panel labels: "(a) words" as a left-aligned axes title, 8.5 pt, normal weight.
* Colours: Okabe-Ito; curves are also told apart by line style and/or marker.
* pdf.fonttype 42 (TrueType); `save()` runs `pdffonts` (poppler, if installed) and refuses
  Type 3 fonts.
* PNG copies at 150 dpi.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.text import Text

FULL_WIDTH = 7.05      # in; two-column journal text width
COL_WIDTH = 3.4        # in; single journal column (8.5 cm)
PNG_DPI = 150
MIN_FONT_PT = 6.0

# Okabe-Ito palette
OI = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
INK = "#1a1a1a"
MUTED = "#555555"
HIST_FACE = "#d0d0d0"
HIST_EDGE = "#8c8c8c"

TITLE_PT = 8.5
LABEL_PT = 8.5
TICK_PT = 7.5
LEGEND_PT = 7.0
NOTE_PT = 7.0          # in-panel annotations
SMALL_PT = 6.5         # crowded in-panel annotations (never below MIN_FONT_PT)

RC = {
    "pdf.fonttype": 42,            # TrueType (Type 42), no Type 3
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": LABEL_PT,
    "axes.titlesize": TITLE_PT,
    "axes.titlelocation": "left",
    "axes.titlepad": 3.5,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": LEGEND_PT,
    "legend.title_fontsize": LEGEND_PT,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.3,
    "lines.markersize": 3.5,
    "hatch.linewidth": 0.6,
    "mathtext.fontset": "dejavusans",
    "savefig.facecolor": "white",
}


def apply_style() -> None:
    matplotlib.rcParams.update(RC)


def panel_title(ax, letter: str, text: str = "") -> None:
    """Panel label in the common position/style: '(a) text', left-aligned above the axes."""
    ax.set_title(f"({letter}) {text}".rstrip(), loc="left", fontsize=TITLE_PT,
                 fontweight="normal", pad=RC["axes.titlepad"])


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------ checks
def _drawn_texts(fig, dpi: int = 300):
    """Text artists (incl. tick labels, legend entries, annotations) actually drawn on the canvas.

    Recorded by wrapping Text.draw during one full draw, so hidden tick labels, axes with
    axis('off') and out-of-view ticks are excluded automatically.
    """
    drawn = []
    orig = Text.draw

    def recording_draw(self, renderer, *args, **kwargs):
        if self.get_visible() and self.get_text().strip():
            drawn.append(self)
        return orig(self, renderer, *args, **kwargs)

    fig.set_dpi(dpi)
    Text.draw = recording_draw
    try:
        fig.canvas.draw()
    finally:
        Text.draw = orig
    seen, out = set(), []
    for t in drawn:
        if id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


def check_min_font(fig, min_pt: float = MIN_FONT_PT) -> float:
    """Raise if any visible text is smaller than min_pt; return the smallest size found."""
    sizes = [(t.get_fontsize(), t.get_text()) for t in _drawn_texts(fig)]
    small = [s for s in sizes if s[0] < min_pt - 1e-9]
    if small:
        raise RuntimeError(f"text below {min_pt} pt: {small[:5]}")
    return min(s for s, _ in sizes)


def check_canvas(fig, dpi: int = 300) -> None:
    """Raise if any drawn text, axes or legend extends beyond the canvas (would be clipped)."""
    texts = _drawn_texts(fig, dpi)
    r = fig.canvas.get_renderer()
    w, h = fig.get_size_inches() * dpi
    boxes = [ax.get_tightbbox(r) for ax in fig.axes if ax.get_visible()]
    boxes += [t.get_window_extent(r) for t in texts]
    for b in boxes:
        if b.x0 < -1 or b.y0 < -1 or b.x1 > w + 1 or b.y1 > h + 1:
            raise RuntimeError(f"content overflows the {w:.0f}x{h:.0f} canvas: {b}")


def text_overlaps(fig, dpi: int = 300, pad_px: float = 0.0):
    """Pairs of visible text items (incl. tick labels and whole legends) whose boxes intersect.

    Texts inside one legend are grouped with that legend's frame, so only clashes between a
    legend and outside text (or between two free texts) are reported.
    """
    texts = _drawn_texts(fig, dpi)
    r = fig.canvas.get_renderer()
    items = []
    legend_texts = {}
    for lg in fig.findobj(Legend):
        if lg.get_visible():
            items.append((f"legend@{id(lg)}", "[legend]", lg.get_window_extent(r)))
            for t in lg.get_texts() + [lg.get_title()]:
                legend_texts[id(t)] = f"legend@{id(lg)}"
    for t in texts:
        if id(t) in legend_texts:
            continue
        items.append((f"text@{id(t)}", t.get_text(), t.get_window_extent(r)))
    clashes = []
    for (ka, ta, ba), (kb, tb, bb) in combinations(items, 2):
        if ka == kb:
            continue
        if (ba.x0 < bb.x1 - pad_px and bb.x0 < ba.x1 - pad_px and
                ba.y0 < bb.y1 - pad_px and bb.y0 < ba.y1 - pad_px):
            clashes.append((ta, tb))
    return clashes


def pdf_fonts(pdf_path: Path) -> list[str]:
    """Return the font table rows from `pdffonts`; raise if any font is Type 3.

    `pdffonts` (poppler-utils) is optional; without it the font check is skipped.
    """
    if shutil.which("pdffonts") is None:
        return ["(pdffonts not installed; font check skipped)"]
    out = subprocess.run(["pdffonts", str(pdf_path)], capture_output=True, text=True, check=True)
    rows = out.stdout.strip().splitlines()[2:]
    bad = [row for row in rows if "Type 3" in row]
    if bad:
        raise RuntimeError(f"Type 3 fonts in {pdf_path}: {bad}")
    return rows


def save(fig, outdir: Path, stem: str, check_overlap: bool = True) -> dict:
    """Run the layout checks, then write <stem>.pdf and <stem>.png (150 dpi) atomically."""
    min_pt = check_min_font(fig)
    try:
        check_canvas(fig)
    except RuntimeError as err:
        if not os.environ.get("FIG_DRAFT"):    # FIG_DRAFT=1 turns layout errors into warnings
            raise
        print(f"WARNING {err}")
    clashes = text_overlaps(fig) if check_overlap else []
    if clashes:
        if os.environ.get("FIG_DRAFT"):          # FIG_DRAFT=1 turns layout errors into warnings
            print(f"WARNING overlapping text in {stem}: {clashes}")
        else:
            raise RuntimeError(f"overlapping text in {stem}: {clashes}")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.set_dpi(PNG_DPI)
    written = {}
    # no creation date in the PDF, so that rebuilding a figure gives identical bytes
    for ext, kw in (("pdf", {"metadata": {"CreationDate": None}}), ("png", {"dpi": PNG_DPI})):
        target = outdir / f"{stem}.{ext}"
        tmp = outdir / f".{stem}.tmp.{ext}"
        fig.savefig(tmp, format=ext, **kw)
        tmp.replace(target)
        written[ext] = target
    fonts = pdf_fonts(written["pdf"])
    w, h = fig.get_size_inches()
    return dict(pdf=written["pdf"], png=written["png"], width_in=w, height_in=h,
                min_font_pt=min_pt, fonts=fonts)
