# Current local artifact

This is the current revision of the manuscript evidence and presentation, published as **networkqbench-wireless-v1.2.3**. Release v1.2.2 covers the historical records through the template study. Release v1.2.3 adds direct-family fitting, deadline-limited scaling, ordering diagnostics, the cartoon renderer, and current tables. No deposit or live-QPU measurement is claimed.

## Reproduce stored evidence

Use Python **3.14.6** from `.python-version`. Install `requirements-energy.txt` and `requirements-validation.txt`. The recorded external rendering environment is TeX Live 2020 (pdfTeX 1.40.21), with `elsarticle`, `txfonts`, `type1cm`, `type1ec`, `xr`, latexmk, dvipng, and Poppler. `preflight.py` checks versions and tool/resource availability; it prints the installed TeX version rather than pretending TeX is a pip dependency. A C++17 compiler is needed by the density-matrix validation.

From the artifact directory in a fresh extraction:

```sh
python preflight.py
python reproduce_current.py --output output/current --validate
```

This runs the current analyses, every historical table producer used by the supplement, current renderers, and validation suites. It does not refit or remeasure. All derived results, figures, captioned tables, numerical macros, logs, and the reproduction manifest go under `output/`. Validation reports go under `output/validation/`. Frozen `results/` and `source_snapshots/` are checked before and after execution. Analyzers reject derived output destinations inside these trees. Raw campaigns have independent `--campaign` arguments. No manuscript directory is required.

The complete local source archive also supplies a sibling `manuscript/` directory. To copy generated assets and compile it:

```sh
python reproduce_current.py --output output/current --manuscript ../manuscript
cd ../manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error submission-materials/cover-letter.tex
```

Compile the supplement first: main-paper pointers use its stable labels through `xr`. `--manuscript` copies only generated tables and figures. Renderers never rewrite scientific prose or the bibliography. Cartoon letters retain their illustrated font; plot labels use the article's Times family. Numerical signs and intervals are formatted centrally, and Figure 1 validates its simulated actions against independent event integration.

Historical manifests retain their original bytes, including obsolete absolute provenance paths. The audit analyzer emits portable projections under `output/current/analysis/audit/portable_manifests/`, with the original SHA256 and relative-path mapping in `manifest.json`. New manifests use relative input identifiers. Frozen execution snapshots deliberately preserve the code actually run; current safe entry points are at the artifact root.

See `README-round4.md` for optional training and scaling remeasurement and `README-round5.md` for the ordering experiment. Run timing experiments sequentially using new output directories. Repeated timing and sampling remain repeated measurements of the same instances. Intervals use 20,000 size-stratified paired bootstrap resamples after within-instance averaging. Demand growth with infeasible incumbents, higher-utilization redesign, direct-family depth-two fitting, second-host replication, and live hardware remain future work.
