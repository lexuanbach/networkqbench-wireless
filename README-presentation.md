# Current local manuscript presentation

The public v1.2.2 release supplies historical evidence through the template study. New direct-family and deadline-scaling evidence is local; see README-round4.md. Local presentation edits retain the same results: cartoon Figure 1, Times plot typography, and the two-scale horizontal component decomposition in Figure 4. They are included with the current manuscript-source package, not a replacement of the immutable public ZIP.

Install the pinned Python requirements and a TeX distribution with `latex`, `txfonts`, `type1cm`, `type1ec`, and `dvipng`. The latter renders PNG previews. Selected cartoon icons, fonts, and their licenses are bundled under `figure1_assets/`; no external plotting-kit folder is needed.

From the artifact directory:

```sh
python reproduce_current.py --output output/current
python render_fig1_cartoon.py --output output/figure1-check
```

The full renderer finishes by generating the current cartoon and its numeric macros. Figure 1 validates stored HiGHS actions, preceding-incumbent optimality, all 30 sampled outcomes, independent event integration, and all 60-instance computation means. It writes `fig1_consistency.json`. Figure 4 retains quantum-minus-HiGHS signs and joules; its two horizontal axes have explicitly different scales. Labels are rounded independently.

The optional source-package build is documented in [README-current.md](README-current.md), with the supplement compiled before the article. Table 3 uses natural-width single-line columns. No experiment rerun or new publication is necessary to reproduce these presentation edits.
