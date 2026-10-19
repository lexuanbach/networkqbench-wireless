# Local condensed manuscript and additional diagnostics

This package supersedes the round-four presentation while preserving its experiments, historical releases, and measurements. Release v1.2.3 publishes the round-four and round-five additions, which v1.2.2 does not contain. No DOI or hardware measurement is claimed. The requested direct-family and deadline-scaling campaigns remain documented in `README-round4.md`.

## Reproduce current documents

Use the environment in `requirements-energy.txt`, LaTeX/latexmk and the plotting dependencies in `README-round4.md`. From `artifact/` in a fresh extraction:

```sh
python reproduce_current.py --output output/current --validate
```

See [README-current.md](README-current.md) for optional asset copying and PDF compilation when the local manuscript-source package is present. The standalone artifact requires no manuscript tree.

The renderer runs the full legacy/round-four chain, then generates the compact sensitivity and campaign-specific comparison tables, per-instance shot requirements, medians, and diagnostic macros. Figure 1 retains the cartoon layout and shows 94.9% useful candidates and 8.39% applied actions. Plot typography remains Times. The first-batch selection profile and other uncertainty statements are retained.

`results/round5_20261019/` contains three crossed overhead/envelope replays, exact per-instance larger-case shot requirements, training/test medians, original-scaling and timing diagnostics, and a bounded six-variable reversed-ring experiment. Bootstrap intervals use 20,000 size-stratified paired resamples with seed 20261017 after averaging repetitions within each instance. Crossed scenarios change only the declared power, utilization, repetition delay, and setup duration. They retain admission/benefit gates and computation charges.

## Optional ordering rerun

```sh
python analyze_round5.py --input results --output output/remeasured-ordering --run-ordering
```

Use a new output directory. The frozen source snapshot is `source_snapshots/round5_ordering/`. This exploratory probability-only diagnostic uses expectation training, the same 7×7 grid, at most 40 Nelder–Mead iterations, and each seed's historical individual evaluation cap. Seeds 0–14 train, seeds 15–29 test. It compares forward and reversed edge ordering, not penalty tuning, and makes no new compiled-energy claim. Recorded simulator timing is research cost only. The manifest preserves exact source/input hashes and environment information.

## Validate

Run the eight suites in `README-round4.md` and `python test_round5.py` in a fresh extraction; all validation reports are written under output/validation. The new checks compare the reversed circuit against an independently constructed Qiskit circuit, enforce training separation and fitting caps, check each minimum shot threshold, and independently integrate 666 compressed crossed-scenario event records (with original multiplicities retained by the analysis). A source/archive checksum audit protects the original records outside the extraction.

Limitations remain explicit: no higher-utilization redesign, no direct family depth-two fit, no infeasible-incumbent growth study, and no second-host or live-QPU validation. Inline RQ answers are retained as requested.
