"""Typeset plot text with the manuscript's LaTeX txfonts Times family.

Requires LaTeX with txfonts, type1cm/type1ec, and dvipng for PNG previews.
Call after adding plot text, before saving. Cartoon lettering may be retained
by selecting axes_only=True.
"""
import re
import matplotlib as mpl
from matplotlib.text import Text

PREAMBLE = r'''\usepackage[T1]{fontenc}
\usepackage{txfonts}
\DeclareUnicodeCharacter{00B5}{\ensuremath{\mu}}
\DeclareUnicodeCharacter{03BC}{\ensuremath{\mu}}
\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}
\DeclareUnicodeCharacter{2212}{\ensuremath{-}}
'''


def apply_paper_font(fig, axes_only=False):
    mpl.rcParams.update({'font.serif': ['Times'], 'text.latex.preamble': PREAMBLE})
    owners = fig.axes if axes_only else [fig]
    for owner in owners:
        # Materialize tick labels before walking the text objects.
        if hasattr(owner, 'get_xticklabels'):
            owner.get_xticklabels(); owner.get_yticklabels()
        for text in owner.findobj(match=Text):
            if not getattr(text, '_paper_font_applied', False):
                parts = re.split(r'(\$[^$]*\$)', text.get_text())
                for i in range(0, len(parts), 2):
                    parts[i] = re.sub(r'(?<!\\)([%&#_])', r'\\\1', parts[i])
                value = ''.join(parts)
                if text.get_fontweight() in ('bold', 'heavy', 'semibold', 600, 700, 800, 900):
                    value = r'\textbf{' + value + '}'
                text.set_text(value)
                text._paper_font_applied = True
            text.set_fontfamily('serif')
            text.set_usetex(True)


def save_figure(fig, path, **kwargs):
    apply_paper_font(fig)
    fig.savefig(path, **kwargs)
