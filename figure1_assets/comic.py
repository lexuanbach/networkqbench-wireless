"""Reusable illustrated figure primitives, fonts, and spacing checks."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon  # noqa: E402
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from PIL import Image  # noqa: E402

KIT = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(KIT, "fonts")
ICONS = os.path.join(KIT, "icons")

for f in os.listdir(FONTS):
    if f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONTS, f))

FONT = "Comic Neue"
MONO = "Comic Mono"

# Mature, high-contrast palette. Ink and paper carry the cartoon look; four accents carry meaning.
# Accents stay distinguishable under the common colour-vision deficiencies (blue, vermillion,
# mustard, green are the Okabe-Ito hues, deepened). Plum and slate are for neutral accents only.
PAL = {
    "ink": "#1E2A38",       # outlines, text, number badges
    "paper": "#FFF8EC",     # figure background
    "card": "#FFFFFF",      # card fill
    "blue": "#2F5D8C",      # the mechanism / defender / proposed tactic
    "red": "#C8553D",       # attacker / leak / failure
    "mustard": "#E1A42B",   # data / memory records / highlights
    "green": "#2A8C6E",     # success / guarantee holds
    "plum": "#6D4C7D",      # secondary accent
    "slate": "#5B6770",     # muted text, baselines
    "tint_blue": "#E3ECF5", "tint_red": "#F8E3DD", "tint_mustard": "#FBF0D6", "tint_green": "#DDF0E8",
}
OUTLINE = 1.6   # points, card and box outlines
SHADOW = (0.004, -0.007)  # comic drop-shadow offset in figure fraction

plt.rcParams.update({
    "font.family": FONT, "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "legend.fontsize": 6.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": PAL["ink"], "axes.labelcolor": PAL["ink"], "xtick.color": PAL["ink"],
    "ytick.color": PAL["ink"], "text.color": PAL["ink"], "axes.linewidth": 1.2,
    "lines.linewidth": 1.8, "lines.markersize": 4, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def figure(width=7.0, height=3.4):
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor(PAL["paper"])
    return fig


def _box(fig, x, y, w, h, fc, ec, lw, rad, z, shadow=True):
    if shadow:
        fig.patches.append(FancyBboxPatch((x + SHADOW[0], y + SHADOW[1]), w, h,
                                          boxstyle=f"round,pad=0,rounding_size={rad}",
                                          transform=fig.transFigure, fc=PAL["ink"], ec="none", zorder=z - 0.1))
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                                      transform=fig.transFigure, fc=fc, ec=ec, lw=lw, zorder=z))


def card(fig, x, y, w, h, n=None, title=None, subtitle=None, color=None, fill=None, z=-5, hicon=None):
    """Stage card with a coloured header strip, numbered ink badge, bold title and muted subtitle.
    hicon names a formal Tabler icon drawn in white inside a reserved slot at the right end of the
    header, so it can never overlap the title. Keep the title short enough to end before that slot
    (check the preview). Cards sit at negative zorder so inset axes and icons draw above them."""
    color = color or PAL["blue"]
    H = fig.get_figheight()
    head = (0.255 if subtitle else 0.155) / H          # header height fixed in inches
    if hicon:
        from matplotlib.patches import Ellipse
        isz = min(0.024, (head * H - 0.05) / fig.get_figwidth())
        cx, cy = x + w - 0.012 - isz / 2, y + h - head / 2 - 0.002
        dr = isz * 0.62
        fig.patches.append(Ellipse((cx, cy), 2 * dr, 2 * dr * fig.get_figwidth() / H, transform=fig.transFigure,
                                   fc="white", ec="none", zorder=z + 0.35))
        if hicon.startswith(("fc_", "fs_")) or os.path.exists(os.path.join(ICONS, hicon + ".png")):
            icon(fig, hicon, cx, cy, size=isz * 0.92, z=z + 0.4)
        else:
            ticon(fig, hicon, cx, cy, size=isz * 0.92, color=color, z=z + 0.4)
    _box(fig, x, y, w, h, fill or PAL["card"], PAL["ink"], OUTLINE, 0.012, z)
    fig.patches[-1]._comic_kind = "card"
    fig.patches[-1]._comic_name = title or f"card {n}"
    fig.patches.append(FancyBboxPatch((x + 0.004, y + h - head), w - 0.008, head - 0.004,
                                      boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                                      fc=color, ec="none", zorder=z + 0.05))
    fig.patches[-1]._comic_kind = "header"
    fig.patches[-1]._comic_name = title or f"card {n}"
    tx = x + 0.012
    if n is not None:
        # The badge sits fully inside the header, so it is never clipped at the figure edge
        # and never covers the card outline.
        r = min(0.0125, (head * H - 0.05) / 2 / fig.get_figwidth())
        badge(fig, x + 0.010 + r, y + h - head / 2 - 0.002, n, r=r, z=z + 0.2, ring=color)
        tx = x + 0.018 + 2 * r
    if title:
        fig.text(tx, y + h - 0.032 / H, title, ha="left", va="top", fontsize=8.6, fontweight="bold",
                 color="white", zorder=z + 0.3)
    if subtitle:
        fig.text(tx, y + h - 0.150 / H, subtitle, ha="left", va="top", fontsize=6.6, color="white",
                 alpha=0.92, zorder=z + 0.3)


def badge(fig, x, y, n, r=0.016, z=5, ring=None):
    """Ink circle with a cream number, centred on (x, y) in figure fraction."""
    aspect = fig.get_figwidth() / fig.get_figheight()
    from matplotlib.patches import Ellipse
    fig.patches.append(Ellipse((x, y), 2 * r, 2 * r * aspect, transform=fig.transFigure,
                               fc=PAL["ink"], ec=PAL["paper"], lw=1.0, zorder=z))
    fig.text(x, y, str(n), ha="center", va="center", fontsize=7.5 if r < 0.014 else 8, fontweight="bold",
             color=PAL["paper"], zorder=z + 0.1)


def icon(fig, name, x, y, size=0.055, z=6, flip=False, rotate=0):
    """Place an icon centred on (x, y); size is the icon width in figure fraction.
    flip=True mirrors it left to right. rotate turns it clockwise in degrees: rotate=90 turns the green
    fc_comments chat icon so its tail points left from the upper left, toward a speaker on its left."""
    path = os.path.join(ICONS, name + ".png")
    if not os.path.exists(path):
        path = os.path.join(KIT, "flatcolor", name + ".png")
    if not os.path.exists(path):
        path = os.path.join(KIT, "fluentsys", name + ".png")   # Fluent UI System colour icons, fs_*
    if not os.path.exists(path):
        raise FileNotFoundError(f"icon {name!r} not in the kit; see icons/ and flatcolor/ for the available names")
    img = Image.open(path).convert("RGBA")
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rotate:
        # degrees, clockwise positive; e.g. rotate=90 turns fc_comments so its tail points left at the top
        img = img.rotate(-rotate, expand=True)
    px = size * fig.get_figwidth() * fig.dpi
    ab = AnnotationBbox(OffsetImage(img, zoom=px / img.width), (x, y), xycoords="figure fraction",
                        frameon=False, zorder=z)
    ab._comic_kind = "icon"
    ab._comic_name = name
    bb = img.getchannel("A").point(lambda a: 255 if a > 24 else 0).getbbox() or (0, 0, img.width, img.height)
    ab._comic_vis = (bb[0] / img.width, 1 - bb[3] / img.height, bb[2] / img.width, 1 - bb[1] / img.height)
    # visible box in figure fraction, used by link()
    wf = size
    hf = size * (img.height / img.width) * fig.get_figwidth() / fig.get_figheight()
    v = ab._comic_vis
    ab._comic_box = (x - wf / 2 + v[0] * wf, y - hf / 2 + v[1] * hf, x - wf / 2 + v[2] * wf, y - hf / 2 + v[3] * hf)
    fig.add_artist(ab)
    return ab


def code(fig, x, y, lines, w=None, fs=6.3, title=None, color=None, highlight=None, z=-1, lang="log"):
    """Code box (Comic Mono) with top-left at (x, y). w=None sizes the box to the longest line.
    highlight maps line index to a colour key in PAL and paints that line's background.
    lang selects syntax highlighting: "python", "bash", "yaml", "ini", "json", "diff" (Pygments),
    "log" (records such as `s1 dev: run pytest -q`, `rad=0`, ids and commands) or None for plain ink.
    Returns (width, height) in figure fraction."""
    lh = fs * 1.38 / 72 / fig.get_figheight()
    pad = 0.008
    if w is None:
        longest = max([len(s) for s in lines] + [len(title) if title else 0])
        w = longest * _mono_advance(fig, fs) + 2 * pad
    h = lh * len(lines) + 2 * pad + (lh if title else 0)
    _box(fig, x, y - h, w, h, "#FDFBF6", color or PAL["ink"], 1.2, 0.006, z, shadow=False)
    fig.patches[-1]._comic_kind = "code"
    fig.patches[-1]._comic_name = "code: " + (title or (lines[0] if lines else ""))[:28]
    yy = y - pad
    if title:
        fig.text(x + pad, yy, title, ha="left", va="top", fontsize=fs, fontweight="bold",
                 color=color or PAL["slate"], zorder=z + 0.3)
        yy -= lh
    for i, ln in enumerate(lines):
        if highlight and i in highlight:
            fig.patches.append(FancyBboxPatch((x + 0.003, yy - lh * 0.95), w - 0.006, lh * 0.95,
                                              boxstyle="round,pad=0,rounding_size=0.003",
                                              transform=fig.transFigure, fc=PAL["tint_" + highlight[i]],
                                              ec="none", zorder=z + 0.1))
        if lang:
            cw = _mono_advance(fig, fs)
            for col, tok, colour, bold in _highlight(ln, lang):
                fig.text(x + pad + col * cw, yy, tok, ha="left", va="top", fontsize=fs, family=MONO, color=colour,
                         fontweight="bold" if bold else "normal", zorder=z + 0.3)
        else:
            fig.text(x + pad, yy, ln, ha="left", va="top", fontsize=fs, family=MONO, color=PAL["ink"], zorder=z + 0.3)
        yy -= lh
    return w, h


def bubble(fig, x, y, text, tail=None, fc=None, ec=None, fs=6.8, w=None, z=8, bold=False, color=None,
           pad=0.35, side=None, tail_w_pt=10.0, tail_len_pt=8.0):
    """Speech bubble centred on (x, y) with an optional tail pointing at tail=(tx, ty).
    side="left" (or "right") draws a short, wide tail (tail_len_pt long) from the middle of that side,
    level with the bubble's centre; pass tail=(tx, any) only to end it at a specific x. Place the bubble so
    the tail tip sits about 3 pt from the speaker. Use fc=PAL["green"]-like fills with ec equal to fc and
    color="white" for a filled chat bubble such as the green comment bubble."""
    t = fig.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=z + 0.2, color=color or PAL["ink"],
                 fontweight="bold" if bold else "normal",
                 bbox=dict(boxstyle=f"round,pad={pad},rounding_size=0.6", fc=fc or "white",
                           ec=ec or PAL["ink"], lw=1.2))
    if side in ("left", "right"):
        fig.canvas.draw()
        bb = t.get_bbox_patch().get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        edge = bb.x0 if side == "left" else bb.x1
        inset = 6.0 / 72 / fig.get_figwidth() * (1 if side == "left" else -1)
        half = tail_w_pt / 2 / 72 / fig.get_figheight()
        cy = (bb.y0 + bb.y1) / 2
        tip = tail[0] if tail is not None else edge - (tail_len_pt / 72 / fig.get_figwidth()) * (1 if side == "left" else -1)
        fig.patches.append(Polygon([(edge + inset, cy - half), (edge + inset, cy + half), (tip, cy)], closed=True,
                                   transform=fig.transFigure, fc=fc or "white", ec=ec or PAL["ink"], lw=1.2, zorder=z + 0.1))
        t._comic_kind, t._comic_name = "bubble", "bubble: " + text.replace("\n", " ")[:28]
        return t
    t._comic_kind, t._comic_name = "bubble", "bubble: " + text.replace("\n", " ")[:28]
    if tail is not None:
        fig.patches.append(Polygon([(x - 0.008, y), (x + 0.008, y), tail], closed=True, transform=fig.transFigure,
                                   fc=fc or "white", ec=ec or PAL["ink"], lw=1.2, zorder=z))
    return t


def arrow(fig, x0, y0, x1, y1, color=None, lw=2.2, rad=0.0, style="-|>", z=4, ms=11):
    """Chunky ink arrow between two figure-fraction points."""
    a = FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle=style, mutation_scale=ms,
                        lw=lw, color=color or PAL["ink"], connectionstyle=f"arc3,rad={rad}", zorder=z,
                        shrinkA=0, shrinkB=0)
    fig.patches.append(a)
    return a


def label(fig, x, y, text, color=None, fs=7, bold=False, ha="center", va="center", z=7, outline=False):
    t = fig.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color or PAL["ink"],
                 fontweight="bold" if bold else "normal", zorder=z)
    t._comic_kind, t._comic_name = "label", "label: " + text.replace("\n", " ")[:28]
    if outline:
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground=PAL["paper"])])
    return t


STAMP_MIN_FS = 10.0


def stamp(fig, x, y, text, color=None, fs=10, rot=-8, z=9):
    """Rubber-stamp verdict (for example LEAKS, HOLDS, 0 of 1440) with a thick coloured border."""
    c = color or PAL["red"]
    fs = max(fs, STAMP_MIN_FS)   # author: "make LEAKS and HOLDS bigger"
    t = fig.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=c, rotation=rot,
                 zorder=z, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c, lw=1.8))
    t._comic_kind, t._comic_name = "stamp", "stamp: " + text.replace("\n", " ")[:28]
    return t


def plot_axes(fig, rect, sketch=False, facecolor="white"):
    """Axes for a real-data panel. rect is [left, bottom, width, height] in figure fraction.
    sketch=True gives a light hand-drawn wobble to lines and spines without hurting legibility."""
    ax = fig.add_axes(rect, facecolor=facecolor, zorder=2)
    ax._comic_kind, ax._comic_name = "axes", "card plot"
    ax.grid(True, alpha=0.25, lw=0.6, color=PAL["slate"])
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1.3)
        if sketch:
            ax.spines[s].set_sketch_params(scale=0.6, length=60, randomness=4)
    ax.tick_params(length=2.5, width=1.0, pad=1.5)
    return ax


def sketch_lines(ax, scale=0.7):
    """Disabled on 14 Oct 2026 at the author's request: plot axes and lines are drawn straight.
    Kept as a no-op so existing figure scripts still run; it also removes any wobble already set."""
    for a in list(ax.lines) + list(ax.patches) + list(ax.spines.values()):
        a.set_sketch_params(None)


def _add_margin(fig, margin):
    """Grow the canvas by `margin` inches on every side and move every artist inward, so all physical
    sizes (fonts, icons, line widths, card geometry in inches) stay the same."""
    from matplotlib.transforms import Affine2D
    W, H = fig.get_size_inches()
    W2, H2 = W + 2 * margin, H + 2 * margin
    A = Affine2D().scale(W / W2, H / H2).translate(margin / W2, margin / H2)
    for ax in fig.axes:
        l, b, w, h = ax.get_position().bounds
        ax.set_position([(l * W + margin) / W2, (b * H + margin) / H2, w * W / W2, h * H / H2])
    fig.set_size_inches(W2, H2)
    for a in list(fig.patches) + list(fig.texts):
        if a.get_transform() == fig.transFigure or getattr(a, "_comic_ff", False) or a.get_transform().contains_branch(fig.transFigure):
            a.set_transform(A + fig.transFigure)
    for a in list(fig.artists):
        if isinstance(a, AnnotationBbox):
            a.xy = tuple(A.transform(a.xy))
            if a.xybox is not None:
                a.xybox = tuple(A.transform(a.xybox))


def save(fig, path, margin=0.10):
    """Save PDF and a preview PNG. margin adds a uniform cream border in inches around the content."""
    if margin and not getattr(fig, "_comic_margin_done", False):
        _add_margin(fig, margin)
        fig._comic_margin_done = True
    problems = check_spacing(fig)
    # Write atomically: render to a temporary file, then rename in one step, so a LaTeX build running
    # at the same moment never reads a half-written figure ("PDF inclusion: type <error>").
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp-{os.getpid()}{ext}"
    fig.savefig(tmp, facecolor=fig.get_facecolor(), metadata={"CreationDate": None, "ModDate": None})
    os.replace(tmp, path)
    png = os.path.splitext(path)[0] + "_preview.png"
    tmp_png = f"{os.path.splitext(png)[0]}.tmp-{os.getpid()}.png"
    fig.savefig(tmp_png, dpi=170, facecolor=fig.get_facecolor())
    os.replace(tmp_png, png)
    plt.close(fig)
    print("wrote", path, "and", png)


# ---------------------------------------------------------------------------------------------
# Formal data panels (added 14 Oct 2026 at the author's request: "use formal plots").
# The comic cards stay above; the measured panels below use a conventional, journal-style look:
# Arial, no hand-drawn wobble, thin spines, light dashed y-grid, outward ticks, panel letters.
FORMAL_FONT = "DejaVu Sans"
FORMAL = {"size": 6.8, "title": 7.4, "lw": 0.7, "grid": "#C9CED3", "text": "#1A1A1A"}


def formal_axes(fig, rect, title=None, letter=None):
    """Axes for a measured panel in formal style. rect is [left, bottom, width, height] in figure
    fraction. title is set left-aligned in bold Arial, prefixed by the panel letter, e.g. "(a)"."""
    ax = fig.add_axes(rect, facecolor="white", zorder=2)
    ax._comic_kind, ax._comic_name = "panel", f"panel {letter or title}"
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(FORMAL["lw"])
        ax.spines[s].set_color(FORMAL["text"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=FORMAL["grid"], lw=0.5, ls=(0, (3, 2)))
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=2.5, width=FORMAL["lw"], pad=1.5, labelsize=FORMAL["size"],
                   colors=FORMAL["text"])
    if title or letter:
        t = f"({letter}) {title}" if letter and title else (f"({letter})" if letter else title)
        ax.set_title(t, loc="left", fontsize=FORMAL["title"], fontweight="bold", pad=3,
                     fontfamily=FORMAL_FONT, color=FORMAL["text"])
    return ax


def formalize(ax, legend_kw=None):
    """Call after plotting a panel. Sets every text in the axes (tick labels, axis labels, title,
    value annotations, legend) to Arial, removes any sketch wobble, and thins bar edges and error bars.
    legend_kw, if given, rebuilds the legend frameless with those keyword arguments."""
    for a in list(ax.lines) + list(ax.patches) + list(ax.collections) + list(ax.spines.values()):
        try:
            a.set_sketch_params(None)
        except Exception:
            pass
    for p in ax.patches:
        if p.get_edgecolor()[3] > 0 and p.get_linewidth() > 0:
            p.set_linewidth(0.5)
            p.set_edgecolor(FORMAL["text"])
    for c in ax.collections:
        try:
            c.set_linewidth(0.7)
        except Exception:
            pass
    if legend_kw is not None:
        ax.legend(**dict(dict(frameon=False, fontsize=FORMAL["size"], handlelength=1.2, borderaxespad=0.2,
                              prop={"family": FORMAL_FONT, "size": FORMAL["size"]}), **legend_kw))
    texts = list(ax.texts) + [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
    texts += ax.get_xticklabels() + ax.get_yticklabels()
    if ax.get_legend() is not None:
        texts += ax.get_legend().get_texts()
    for t in texts:
        t.set_fontfamily(FORMAL_FONT)
        t.set_color(FORMAL["text"]) if t.get_color() in (PAL["ink"], "black", "#000000") else None
    ax.xaxis.label.set_fontsize(FORMAL["size"] + 0.4)
    ax.yaxis.label.set_fontsize(FORMAL["size"] + 0.4)
    ax.tick_params(labelsize=FORMAL["size"])
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_fontfamily(FORMAL_FONT)
    return ax


def formal_label(fig, x, y, text, fs=None, bold=False, ha="left", va="bottom", color=None):
    """Figure-level text in the formal font, for panel captions placed outside the axes."""
    return fig.text(x, y, text, ha=ha, va=va, fontsize=fs or FORMAL["title"], fontfamily=FORMAL_FONT,
                    fontweight="bold" if bold else "normal", color=color or FORMAL["text"], zorder=7)


# ---------------------------------------------------------------------------------------------
# Syntax highlighting for code boxes (added 14 Oct 2026: "for code, use syntax highlight with
# contrastive colors"). A dedicated syntax palette, distinct from the semantic colours of the figure,
# chosen for strong contrast on the light code background (#FDFBF6).
SYN = {
    "ink": "#1E2A38",       # identifiers, punctuation
    "keyword": "#1F4E9C",   # keywords (bold)
    "func": "#7A3E9D",      # functions, builtins, keys and attributes
    "string": "#9C5B00",    # strings, paths, values
    "number": "#0B7A75",    # numbers, ids
    "comment": "#6B7280",   # comments
    "command": "#B3245D",   # shell commands and flags
    "label": "#3F4A55",     # record prefixes such as "s1 dev:" (bold)
}
_ADV = {}


def _mono_advance(fig, fs):
    """Advance width of one Comic Mono character at fs points, in figure-width fraction."""
    key = (fs, fig.get_figwidth())
    if key not in _ADV:
        from matplotlib.font_manager import FontProperties, findfont, get_font
        f = get_font(findfont(FontProperties(family=MONO)))
        f.set_size(fs, 72)
        f.set_text("M" * 20)
        _ADV[key] = f.get_width_height()[0] / 64.0 / 20 / 72 / fig.get_figwidth()
    return _ADV[key]


_SHELL = {"pytest", "curl", "sh", "bash", "pip", "git", "python", "python3", "npm", "make", "docker", "kubectl",
          "wget", "chmod", "export", "cat", "echo", "rm", "ssh", "uv"}


def _highlight(line, lang):
    """Yield (column, token, colour, bold) for one line."""
    import re
    if lang == "log":
        out = []
        pat = re.compile(r"(?P<com>(?:^|(?<=\s))#(?=\s|$).*$)|(?P<id>#\d+)|(?P<str>\"[^\"]*\"|'[^']*')|(?P<label>^\s*[A-Za-z]?\d+\s+[A-Za-z_]+:)"
                         r"|(?P<kv>[A-Za-z_][\w.\-]*=)|(?P<arrow>->|=>|\|)|(?P<flag>(?<!\w)--?[A-Za-z][\w\-]*)"
                         r"|(?P<path>[\w.\-]+/[\w./\-]*|[\w\-]+\.(?:py|sh|yml|yaml|json|toml|cfg|md|txt)\b)"
                         r"|(?P<num>(?<![\w.])\d+(?:\.\d+)?%?(?![\w]))|(?P<word>[A-Za-z_][\w\-]*)|(?P<other>\s+|.)")
        for m in pat.finditer(line):
            kind, tok = m.lastgroup, m.group(0)
            if kind == "word" and tok in _SHELL:
                kind = "cmd"
            colour, bold = {
                "com": (SYN["comment"], False), "id": (SYN["number"], True), "str": (SYN["string"], False), "label": (SYN["label"], True),
                "kv": (SYN["func"], False), "arrow": (SYN["keyword"], True), "flag": (SYN["command"], False),
                "path": (SYN["string"], False), "num": (SYN["number"], False), "cmd": (SYN["command"], True),
            }.get(kind, (SYN["ink"], False))
            out.append((m.start(), tok, colour, bold))
        return [t for t in out if t[1].strip()]
    from pygments import lex
    from pygments.lexers import get_lexer_by_name
    from pygments.token import Keyword, Name, String, Number, Comment, Operator, Literal
    lexer = get_lexer_by_name({"yml": "yaml"}.get(lang, lang), stripnl=False, ensurenl=False)
    col, out = 0, []
    for ttype, tok in lex(line, lexer):
        if tok.strip():
            if ttype in Comment:
                c, b = SYN["comment"], False
            elif ttype in Keyword:
                c, b = SYN["keyword"], True
            elif ttype in String or ttype in Literal.String:
                c, b = SYN["string"], False
            elif ttype in Number:
                c, b = SYN["number"], False
            elif ttype in Name.Builtin or ttype in Name.Function or ttype in Name.Attribute or ttype in Name.Tag \
                    or ttype in Name.Decorator or ttype in Name.Variable:
                c, b = SYN["func"], False
            elif ttype in Operator:
                c, b = SYN["keyword"], False
            elif lang == "bash" and tok in _SHELL:
                c, b = SYN["command"], True
            else:
                c, b = SYN["ink"], False
            out.append((col, tok, c, b))
        col += len(tok)
    return out


# ---------------------------------------------------------------------------------------------
# Formal concept icons (added 14 Oct 2026: "use formal icons", "more relevant icons").
# Tabler outline icons (MIT), recoloured on the fly. Use them for concepts and infrastructure
# (server, cloud, eye-off, calendar, certificate, ledger ...); keep the 3D Fluent icons for the
# people and the coding assistant only.
TABLER = os.path.join(KIT, "tabler")
_CACHE = os.path.join(KIT, ".cache")


def ticon(fig, name, x, y, size=0.04, color=None, z=6, stroke=1.6):
    """Place a formal Tabler outline icon centred on (x, y), drawn in `color` (default ink).
    size is the icon width in figure fraction. See tabler/ for the available names."""
    import hashlib
    import subprocess
    src = os.path.join(TABLER, name + ".svg")
    if not os.path.exists(src):
        raise FileNotFoundError(f"tabler icon {name!r} not in the kit; see tabler/ for the available names")
    color = color or PAL["ink"]
    os.makedirs(_CACHE, exist_ok=True)
    key = hashlib.sha1(f"{name}|{color}|{stroke}".encode()).hexdigest()[:12]
    png = os.path.join(_CACHE, f"{name}_{key}.png")
    if not os.path.exists(png):
        svg = open(src).read().replace("currentColor", color).replace('stroke-width="2"', f'stroke-width="{stroke}"')
        tmp = png + ".svg"
        open(tmp, "w").write(svg)
        subprocess.run(["rsvg-convert", "-w", "256", "-h", "256", "-b", "none", tmp, "-o", png], check=True)
        os.remove(tmp)
    img = Image.open(png).convert("RGBA")
    px = size * fig.get_figwidth() * fig.dpi
    ab = AnnotationBbox(OffsetImage(img, zoom=px / img.width), (x, y), xycoords="figure fraction",
                        frameon=False, zorder=z)
    fig.add_artist(ab)
    return ab


# ---------------------------------------------------------------------------------------------
# Spacing check (added 14 Oct 2026: "some boxes are too close, add space"). Runs in save() and
# prints every violation, so layouts are fixed by measurement rather than by eye.
GAP_IN = 3.0 / 72  # minimum clear space between two elements: 3 pt (author: "gaps 3-4pt")
PAD_IN = 4.0 / 72  # minimum clear space to a card outline or header: 4 pt
ROW_GAP_IN = 6.0 / 72  # minimum clear space between the card row and the panel row: 6 pt
SLACK_IN = 0.18    # maximum unused height inside a card (author: "now too much space, more compact")
MIN_ACTOR_IN = 0.30   # minimum width of people and assistant icons (author: "some icons too small")
MIN_ICON_IN = 0.30    # minimum visible width of concept icons and logos (author: "icon to ra", bigger)
MIN_FILL = 0.40       # minimum share of a card's inner area covered by content
PANEL_MIN_H = 0.95 # minimum height of a measured panel's plot area, inches ("taller so plots look balanced")


def _elements(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    def box(bb):
        return (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi)
    els = []
    for p in fig.patches:
        k = getattr(p, "_comic_kind", None)
        if k:
            els.append((k, p._comic_name, box(p.get_window_extent(r))))
    for t in fig.texts:
        k = getattr(t, "_comic_kind", None)
        if k:
            bp = t.get_bbox_patch()
            bb = bp.get_window_extent(r) if bp is not None else t.get_window_extent(r)
            els.append((k, t._comic_name, box(bb)))
    for a in fig.artists:
        if getattr(a, "_comic_kind", None) == "icon":
            x0, y0, x1, y1 = box(a.get_window_extent(r))
            v = getattr(a, "_comic_vis", (0, 0, 1, 1))
            w, h = x1 - x0, y1 - y0
            # measure the visible pixels only, not the icon's transparent padding
            els.append(("icon", "icon: " + a._comic_name, (x0 + v[0] * w, y0 + v[1] * h, x0 + v[2] * w, y0 + v[3] * h)))
    for ax in fig.axes:
        k = getattr(ax, "_comic_kind", None)
        if k:
            els.append((k, ax._comic_name, box(ax.get_tightbbox(r))))
    return els


def _gap(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    if dx == 0 and dy == 0:
        return -1.0  # overlap
    return (dx ** 2 + dy ** 2) ** 0.5 if dx and dy else max(dx, dy)


def _inside(inner, outer):
    c = ((inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2)
    return outer[0] <= c[0] <= outer[2] and outer[1] <= c[1] <= outer[3]


def check_spacing(fig, gap=GAP_IN, pad=PAD_IN, row_gap=ROW_GAP_IN, verbose=True):
    """Return a list of spacing violations (and print them). Content inside a card must keep `pad`
    inches from the card outline and from the header; any two content elements in the same card
    must keep `gap` inches apart; the card row must keep `row_gap` inches from the panels below."""
    els = _elements(fig)
    cards = [e for e in els if e[0] == "card"]
    headers = [e for e in els if e[0] == "header"]
    content = [e for e in els if e[0] in ("code", "bubble", "label", "stamp", "icon", "axes")
               and not any(_inside(e[2], hd[2]) for hd in headers)]  # header icons live in their slot
    panels = [e for e in els if e[0] == "panel"]
    out = []
    for ax in fig.axes:
        if getattr(ax, "_comic_kind", None) == "panel":
            hgt = ax.get_position().height * fig.get_figheight()
            if hgt < PANEL_MIN_H:
                out.append(f"{ax._comic_name} plot area is {hgt:.2f} in tall (need at least {PANEL_MIN_H} in for balanced panels)")
    owner = {}
    for e in content:
        for c in cards:
            if _inside(e[2], c[2]):
                owner[id(e)] = c
                b, o = e[2], c[2]
                m = min(b[0] - o[0], o[2] - b[2], b[1] - o[1])
                if m < pad:
                    out.append(f"{e[1]} is {m:.3f} in from the outline of card '{c[1]}' (need {pad})")
                for hd in headers:
                    if hd[1] == c[1]:
                        g = hd[2][1] - b[3]
                        if g < pad:
                            out.append(f"{e[1]} is {g:.3f} in below the header of card '{c[1]}' (need {pad})")
                break
    for i in range(len(content)):
        for j in range(i + 1, len(content)):
            a, b = content[i], content[j]
            if owner.get(id(a)) is not owner.get(id(b)):
                continue
            if _inside(a[2], b[2]) and a[0] != "icon" and b[0] in ("code", "bubble", "axes"):
                continue  # a label placed inside a box or plot by design
            if _inside(b[2], a[2]) and b[0] != "icon" and a[0] in ("code", "bubble", "axes"):
                continue
            g = _gap(a[2], b[2])
            if g < gap:
                what = "overlaps" if g < 0 else f"is {g:.3f} in from"
                out.append(f"{a[1]} {what} {b[1]} (need {gap})")
    for c in cards:
        mine = [e for e in content if owner.get(id(e)) is c]
        hd = [h for h in headers if h[1] == c[1]]
        if mine and hd:
            inner_top = hd[0][2][1]
            used_top = max(e[2][3] for e in mine)
            used_bottom = min(e[2][1] for e in mine)
            slack = (inner_top - used_top) + (used_bottom - c[2][1])
            # vertical gaps between stacked content also count as slack beyond the 3 pt minimum
            ys = sorted((e[2][1], e[2][3]) for e in mine)
            covered, cur_lo, cur_hi = 0.0, None, None
            for lo, hi in ys:
                if cur_hi is None or lo > cur_hi:
                    if cur_hi is not None:
                        covered += cur_hi - cur_lo
                    cur_lo, cur_hi = lo, hi
                else:
                    cur_hi = max(cur_hi, hi)
            covered += cur_hi - cur_lo
            unused = (inner_top - c[2][1]) - covered
            if unused > SLACK_IN + 2 * PAD_IN:
                out.append(f"card '{c[1]}' has {unused:.2f} in of unused height (allow {SLACK_IN + 2 * PAD_IN:.2f}); make the card shorter or the content larger")
    actors = {"technologist", "robot", "bust_in_silhouette", "man_office_worker", "office_worker",
              "woman_office_worker", "man_technologist", "woman_technologist", "ninja", "detective"}
    for e in content:
        if e[0] == "icon":
            name = e[1].replace("icon: ", "")
            wdt = e[2][2] - e[2][0]
            need = MIN_ACTOR_IN if name in actors else MIN_ICON_IN
            if wdt < need - 1e-3:
                out.append(f"{e[1]} is {wdt:.2f} in wide (need at least {need:.2f} in)")
    for c in cards:
        mine = [e for e in content if owner.get(id(e)) is c]
        hd = [h for h in headers if h[1] == c[1]]
        if not mine or not hd:
            continue
        x0, y0, x1, y1 = c[2][0], c[2][1], c[2][2], hd[0][2][1]
        n = 48
        covered = 0
        for i in range(n):
            for j in range(n):
                px = x0 + (i + 0.5) * (x1 - x0) / n
                py = y0 + (j + 0.5) * (y1 - y0) / n
                if any(e[2][0] <= px <= e[2][2] and e[2][1] <= py <= e[2][3] for e in mine):
                    covered += 1
        fill = covered / (n * n)
        if fill < MIN_FILL:
            out.append(f"card '{c[1]}' is sparse: content covers {fill:.0%} of its area (need {MIN_FILL:.0%}); enlarge content or narrow empty regions")
    # arrows and arcs (kit links and hand-drawn FancyArrowPatch) must not cross elements they do not connect
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    near = 5.0 / 72
    for pa in fig.patches:
        if not isinstance(pa, FancyArrowPatch):
            continue
        try:
            path = pa.get_transform().transform_path(pa.get_path())
        except Exception:
            continue
        from matplotlib.path import Path as _P
        samp, prev, firsts = [], None, []
        for verts, code in path.iter_segments(curves=False, simplify=False):
            if code == _P.MOVETO:
                prev = (verts[0] / dpi, verts[1] / dpi)
                firsts.append(prev)
                continue
            if code in (_P.CLOSEPOLY, _P.STOP) or prev is None:
                prev = None
                continue
            cur = (verts[-2] / dpi, verts[-1] / dpi)
            for k in range(9):
                samp.append((prev[0] + (cur[0] - prev[0]) * k / 8, prev[1] + (cur[1] - prev[1]) * k / 8))
            prev = cur
        if not samp:
            continue
        pa_pos = getattr(pa, "_posA_posB", None)
        if pa_pos and None not in pa_pos:
            W, H = fig.get_figwidth(), fig.get_figheight()
            ends = [(pa_pos[0][0] * W, pa_pos[0][1] * H), (pa_pos[1][0] * W, pa_pos[1][1] * H)]
        else:
            ends = [samp[0], samp[-1]]
        for e in content:
            if e[0] == "icon":
                continue
            b = e[2]
            if any(b[0] - near <= ex <= b[2] + near and b[1] - near <= ey <= b[3] + near for ex, ey in ends):
                continue  # this element is one of the arrow's endpoints
            hits = [1 for (sx, sy) in samp if b[0] < sx < b[2] and b[1] < sy < b[3]]
            if len(hits) >= 2:
                out.append(f"an arrow or arc crosses {e[1]}")
                break
    if cards and panels:
        card_bottom = min(c[2][1] for c in cards)
        top = max(p[2][3] for p in panels)
        if card_bottom - top < row_gap:
            out.append(f"panel row is {card_bottom - top:.3f} in below the card row (need {row_gap})")
    if verbose:
        print(f"spacing check: {len(out)} violation(s)")
        for v in out:
            print("  SPACING:", v)
    return out



# ---------------------------------------------------------------------------------------------
# Connections between icons (added 14 Oct 2026: "add connections between the icons").
def link(fig, a, b, label=None, color=None, both=False, dashed=False, gap_pt=2.5, fs=6.4, label_side="above",
         rad=0.0, lw=1.3):
    """Draw a short arrow from a to b, where each is an icon (from icon()), a bubble or label artist, or a
    (x0, y0, x1, y1) box in figure fraction (for a code box at top-left (x, y) with size (w, h) returned by
    code(), pass (x, y - h, x + w, y)). The arrow starts and
    ending gap_pt points outside their visible edges, with an optional relation label ("asks", "reads",
    "merges", "watches"). Picks horizontal or vertical attachment from the icons' relative position.
    both=True draws a two-headed arrow, dashed=True a dashed one (for an observed or indirect relation)."""
    def _box_of(o):
        if hasattr(o, "_comic_box"):
            return o._comic_box
        if isinstance(o, (tuple, list)) and len(o) == 4:
            return tuple(o)                       # (x0, y0, x1, y1) in figure fraction, e.g. a code box
        if hasattr(o, "get_window_extent"):       # a text or bubble artist
            fig.canvas.draw()
            bp = o.get_bbox_patch() if hasattr(o, "get_bbox_patch") else None
            bb = (bp or o).get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            return (bb.x0, bb.y0, bb.x1, bb.y1)
        raise TypeError("link() joins icons, bubbles/labels, or (x0, y0, x1, y1) boxes in figure fraction")
    A, B = _box_of(a), _box_of(b)
    W, H = fig.get_figwidth(), fig.get_figheight()
    gx, gy = gap_pt / 72 / W, gap_pt / 72 / H
    ca = ((A[0] + A[2]) / 2, (A[1] + A[3]) / 2)
    cb = ((B[0] + B[2]) / 2, (B[1] + B[3]) / 2)
    if abs(cb[0] - ca[0]) * W >= abs(cb[1] - ca[1]) * H:      # mostly horizontal: keep it level
        lo, hi = max(A[1], B[1]), min(A[3], B[3])
        yy = (lo + hi) / 2 if hi > lo else (ca[1] + cb[1]) / 2
        if cb[0] > ca[0]:
            p0, p1 = (A[2] + gx, yy), (B[0] - gx, yy)
        else:
            p0, p1 = (A[0] - gx, yy), (B[2] + gx, yy)
    else:                                                      # mostly vertical: keep it plumb
        lo, hi = max(A[0], B[0]), min(A[2], B[2])
        xx = (lo + hi) / 2 if hi > lo else (ca[0] + cb[0]) / 2
        if cb[1] > ca[1]:
            p0, p1 = (xx, A[3] + gy), (xx, B[1] - gy)
        else:
            p0, p1 = (xx, A[1] - gy), (xx, B[3] + gy)
    style = "<|-|>" if both else "-|>"
    arr = FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle=style, mutation_scale=8, lw=lw,
                          color=color or PAL["slate"], linestyle=(0, (3, 2)) if dashed else "-",
                          connectionstyle=f"arc3,rad={rad}", zorder=5, shrinkA=0, shrinkB=0)
    fig.patches.append(arr)
    t = None
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        off = 5.5 / 72 / H
        t = fig.text(mx, my + (off if label_side == "above" else -off), label, ha="center",
                     va="bottom" if label_side == "above" else "top", fontsize=fs, color=color or PAL["slate"],
                     zorder=7)
        t._comic_kind, t._comic_name = "label", "link label: " + label
    return arr, t


# ---------------------------------------------------------------------------------------------
# Stage arrows between cards (added 14 Oct 2026: "the arrows between the panels are not clear").
def step_arrow(fig, left_card_right_x, right_card_left_x, y, r_in=0.095, color=None, z=6):
    """Draw a round ink badge with a white arrow, centred in the gap between two cards at height y
    (figure fraction), straddling both card edges so it reads clearly even when the gap is narrow.
    r_in is the badge radius in inches. Use one per gap, at the cards' common mid-height."""
    from matplotlib.patches import Ellipse, FancyArrow
    W, H = fig.get_figwidth(), fig.get_figheight()
    cx = (left_card_right_x + right_card_left_x) / 2
    rx, ry = r_in / W, r_in / H
    fig.patches.append(Ellipse((cx + 0.6 / 72 / W, y - 0.9 / 72 / H), 2 * rx, 2 * ry, transform=fig.transFigure,
                               fc=PAL["ink"], ec="none", zorder=z - 0.1, alpha=0.35))   # soft comic shadow
    fig.patches.append(Ellipse((cx, y), 2 * rx, 2 * ry, transform=fig.transFigure, fc=color or PAL["ink"],
                               ec=PAL["paper"], lw=1.4, zorder=z))
    L = 1.15 * rx
    fig.patches.append(FancyArrow(cx - L / 2, y, L, 0, width=0.20 * ry, head_width=0.75 * ry, head_length=0.42 * rx,
                                  length_includes_head=True, transform=fig.transFigure, fc="white", ec="none",
                                  zorder=z + 0.1))
