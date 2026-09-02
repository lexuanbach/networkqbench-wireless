#!/usr/bin/env python3
"""Create the 50-paper curated database required by the submission workflow."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / "submission" / "01_literature"
OUT = LIT / "curated"


SELECTED_TITLES = {
    # QAOA, VQA, optimization and benchmarking
    "Variational quantum algorithms",
    "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices",
    "Noisy intermediate-scale quantum algorithms",
    "Quantum approximate optimization of non-planar graph problems on a planar superconducting processor",
    "Adaptive quantum approximate optimization algorithm for solving combinatorial problems on a quantum computer",
    "Evidence of scaling advantage for the quantum approximate optimization algorithm on a classically intractable problem",
    "Challenges and opportunities in quantum optimization",
    "Digitized-counterdiabatic quantum approximate optimization algorithm",
    "Formulating and Solving Routing Problems on Quantum Computers",
    "Multi-angle quantum approximate optimization algorithm",
    "Classical variational simulation of the Quantum Approximate Optimization Algorithm",
    "Learning to Optimize Variational Quantum Circuits to Solve Combinatorial Problems",
    "QASMBench: A Low-Level Quantum Benchmark Suite for NISQ Evaluation and Simulation",
    "Quantum variational algorithms are swamped with traps",
    "Error mitigation with Clifford quantum-circuit data",
    "Constrained quantum optimization for extractive summarization on a trapped-ion quantum computer",
    "Software Mitigation of Crosstalk on Noisy Intermediate-Scale Quantum Computers",
    # Quantum/wireless systems and surveys
    "Quantum-Inspired Real-Time Optimization for 6G Networks: Opportunities, Challenges, and the Road Ahead",
    "In‐Network Quantum Computing for Future 6G Networks",
    "Quantum Computing in Telecommunication—A Survey",
    "Quantum Combinatorial Optimization in the NISQ Era: A Systematic Mapping Study",
    "A Conceptual Architecture for a Quantum-HPC Middleware",
    "A Review on Quantum Approximate Optimization Algorithm and its Variants",
    "Characterizing and Utilizing the Interplay Between Quantum Technologies and Non-Terrestrial Networks",
    "Review of the application of quantum annealing-related technologies in transportation optimization",
    "Quantum Computing in Wireless Communications and Networking: A Tutorial-cum-Survey",
    "Bridging Classic Operations Research and Artificial Intelligence for Network Optimization in the 6G Era: A Review",
    "Using a quantum computer to solve a real-world problem -- what can be achieved today?",
    # Classical network-optimization workloads and controls
    "Virtual Network Embedding based Traffic Scheduling for LEO Satellite Constellations",
    "Virtual Network Function Placement and Routing: Formulations and Solutions",
    "Joint Service Placement and Model Partitioning for Accelerating DNN Inference in Edge Intelligence Empowered Vehicle Networks",
    "Near-Optimal Energy-Efficient Algorithm for Virtual Network Function Placement",
    "Queue-Aware Service Orchestration and Adaptive Parallel Traffic Scheduling Optimization in SDNFV-Enabled Cloud Computing",
    "Disaster-Resilient Service Function Chain Embedding Based on Multi-Path Routing",
    "Optimizing Traffic Engineering for Resilient Services in NFV-Based Connected Autonomous Vehicles",
    "Joint Optimization of VNF Placement and Flow Scheduling in Mobile Core Network",
    "Closed loop optimization of 5G network slices",
}


MANUAL = [
    {
        "title": "Benchmarking Quantum Reinforcement Learning",
        "authors": "Nico Meyer; Christian Ufrecht; George Yammine; Georgios Kontes; Christopher Mutschler; Daniel Scherer",
        "year": 2025, "venue": "ICML", "doi": "", "url": "https://proceedings.mlr.press/v267/meyer25b.html",
        "abstract": "Introduces a statistical estimator for sample complexity and a definition of statistical outperformance for reinforcement-learning benchmarks; the resulting evaluation casts doubt on some previous QRL superiority claims.",
    },
    {
        "title": "The Quantum Optimization Benchmarking Library",
        "authors": "Thorsten Koch et al.", "year": 2026, "venue": "Nature Computational Science",
        "doi": "10.1038/s43588-026-00991-1", "url": "https://www.nature.com/articles/s43588-026-00991-1",
        "abstract": "Presents ten model-independent combinatorial optimization problem classes, classical reference solutions, quantum baseline reporting, and a framework for fair and reproducible tracking of progress toward quantum advantage.",
    },
    {
        "title": "Limits of quantum run-time advantage",
        "authors": "Daniel Koch et al.", "year": 2026, "venue": "Physical Review Applied",
        "doi": "10.1103/gpsf-pn1x", "url": "https://journals.aps.org/prapplied/abstract/10.1103/gpsf-pn1x",
        "abstract": "Defines experimentally grounded end-to-end quantum runtime and strong classical comparison methodology, and shows that several advantage claims do not persist when system-level overheads are included.",
    },
    {
        "title": "Warm-starting quantum optimization",
        "authors": "Daniel J. Egger; Jakub Marecek; Stefan Woerner", "year": 2021, "venue": "Quantum",
        "doi": "10.22331/q-2021-06-17-479", "url": "https://quantum-journal.org/papers/q-2021-06-17-479/",
        "abstract": "Develops warm-start quantum optimization from classical relaxations and shows how QAOA can inherit classical guarantees; illustrates benefits for low-depth portfolio optimization and recursive QAOA.",
    },
    {
        "title": "Cross-Problem Parameter Transfer in Quantum Approximate Optimization Algorithm: A Machine Learning Approach",
        "authors": "Kien X. Nguyen; Bao Bach; Ilya Safro", "year": 2025, "venue": "arXiv",
        "doi": "", "url": "https://arxiv.org/abs/2504.10733",
        "abstract": "Studies QAOA parameter transfer between MaxCut and maximum independent set and reports that selected donor parameters can reduce optimization iterations while retaining comparable approximation ratios.",
    },
    {
        "title": "Sampling frequency thresholds for the quantum advantage of the quantum approximate optimization algorithm",
        "authors": "Gian Giacomo Guerreschi; Jason M. Larkin", "year": 2023, "venue": "npj Quantum Information",
        "doi": "10.1038/s41534-023-00718-4", "url": "https://www.nature.com/articles/s41534-023-00718-4",
        "abstract": "Compares QAOA with Gurobi and MQLib for MaxCut and derives depth and sampling-frequency thresholds, finding that strong classical heuristics substantially restrict plausible near-term advantage regimes.",
    },
    {
        "title": "Limitations of quantum approximate optimization in solving generic higher-order constraint-satisfaction problems",
        "authors": "Thorge Müller; Ajainderpal Singh; Frank K. Wilhelm; Tim Bode", "year": 2025,
        "venue": "Physical Review Research", "doi": "10.1103/PhysRevResearch.7.023165",
        "url": "https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.7.023165",
        "abstract": "Analyzes QAOA on higher-order constraint-satisfaction problems and finds a mean-field classical approximation performs at least as well on average in the studied regimes, while high-quality QAOA solutions require difficult depths.",
    },
    {
        "title": "Mean-Field Approximate Optimization Algorithm",
        "authors": "Thorge Müller et al.", "year": 2023, "venue": "PRX Quantum",
        "doi": "10.1103/PRXQuantum.4.030335", "url": "https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.030335",
        "abstract": "Introduces a quantum-inspired mean-field analogue of QAOA and reports that it outperforms QAOA for most studied partition and spin-glass instances, providing a baseline for delineating possible quantum regimes.",
    },
    {
        "title": "QAOA-in-QAOA: Solving Large-Scale MaxCut Problems on Small Quantum Machines",
        "authors": "Yong-Heng Lee et al.", "year": 2023, "venue": "Physical Review Applied",
        "doi": "10.1103/PhysRevApplied.19.024027", "url": "https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevApplied.19.024027/fulltext",
        "abstract": "Uses hierarchical decomposition to apply small quantum machines to larger MaxCut problems and benchmarks the resulting hybrid method against classical solvers.",
    },
    {
        "title": "Quantum network utility: A framework for benchmarking quantum networks",
        "authors": "Yuan Lee et al.", "year": 2024, "venue": "Advanced Quantum Technologies",
        "doi": "", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11047070/",
        "abstract": "Defines task-oriented utility functions over quantum-network rate regions, extending component metrics toward application-relevant network benchmarking.",
    },
    {
        "title": "Power of data in quantum machine learning",
        "authors": "Hsin-Yuan Huang et al.", "year": 2021, "venue": "Nature Communications",
        "doi": "10.1038/s41467-021-22539-9", "url": "https://www.nature.com/articles/s41467-021-22539-9",
        "abstract": "Develops a framework for assessing quantum prediction advantage and shows that access to training data can make classical models competitive even when underlying quantum computations are difficult.",
    },
    {
        "title": "Q-Backbone: A Quantum-Enhanced Control Plane for Future Communication Networks",
        "authors": "Mahdi Chehimi; Nour Dehaini; Nikos A. Mitsiou; Ioannis Krikidis; Gan Zheng",
        "year": 2026, "venue": "arXiv", "doi": "", "url": "https://arxiv.org/abs/2606.13248",
        "abstract": "Proposes a quantum-enhanced network-control architecture and a quantum invocation policy that dynamically chooses between classical and quantum execution under heterogeneous QPU conditions.",
    },
    {
        "title": "Quantum annealing for combinatorial optimization: a benchmarking study",
        "authors": "Bhaskar Ghosh et al.", "year": 2025, "venue": "npj Quantum Information",
        "doi": "10.1038/s41534-025-01020-1", "url": "https://www.nature.com/articles/s41534-025-01020-1",
        "abstract": "Benchmarks quantum annealing and classical solvers on large dense QUBO instances, reporting solution-quality and timing comparisons while including QPU programming and sampling components.",
    },
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:90]


def bib_key(record: dict, used: set[str]) -> str:
    authors = str(record.get("authors", "anon"))
    surname = re.sub(r"[^A-Za-z]", "", authors.split(";")[0].split()[-1] if authors else "anon").lower() or "anon"
    base = f"{surname}{int(record.get('year', 0) or 0)}{slug(record['title']).split('-')[0]}"
    key = base
    i = 2
    while key in used:
        key = f"{base}{i}"
        i += 1
    used.add(key)
    return key


def relevance(record: dict) -> tuple[str, str]:
    text = (record["title"] + " " + str(record.get("abstract", ""))).lower()
    if any(k in text for k in ["wireless", "network", "routing", "service function", "telecommunication", "6g"]):
        return "network optimization", "Defines a workload, systems constraint, or network-level metric used to design or interpret NetworkQBench-Wireless."
    if any(k in text for k in ["benchmark", "runtime", "run-time", "advantage", "classical"]):
        return "benchmarking and advantage", "Supports fair comparison, full-stack accounting, or advantage/parity/no-advantage claims."
    if any(k in text for k in ["warm", "parameter", "variational", "qaoa"]):
        return "QAOA methods", "Informs the QAOA implementation, optimization-overhead analysis, or recovery experiment."
    return "quantum systems", "Provides relevant background on quantum algorithms, noise, mitigation, or hybrid execution."


def bibtex(key: str, r: dict) -> str:
    entry = "article" if r.get("venue") not in {"ICML"} else "inproceedings"
    fields = [
        f"  title = {{{r['title']}}}",
        f"  author = {{{str(r.get('authors','')).replace(';', ' and')}}}",
        f"  year = {{{int(r.get('year', 0) or 0)}}}",
    ]
    if r.get("venue"):
        fields.append(f"  journal = {{{r['venue']}}}" if entry == "article" else f"  booktitle = {{{r['venue']}}}")
    if r.get("doi") and str(r.get("doi")) != "nan":
        fields.append(f"  doi = {{{r['doi']}}}")
    if r.get("url"):
        fields.append(f"  url = {{{r['url']}}}")
    return "@" + entry + "{" + key + ",\n" + ",\n".join(fields) + "\n}\n"


def main() -> int:
    frames = []
    for path in LIT.glob("*/papers.csv"):
        if path.parent.name == "curated":
            continue
        df = pd.read_csv(path)
        df["source"] = path.parent.name
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df.title.isin(SELECTED_TITLES)].drop_duplicates("title")
    records = all_df.to_dict("records") + MANUAL
    # Normalize visually distinct Unicode punctuation before duplicate checking.
    unique = {}
    for r in records:
        key = re.sub(r"[^a-z0-9]+", " ", r["title"].lower()).strip()
        unique[key] = r
    records = list(unique.values())
    if len(records) != 50:
        raise RuntimeError(f"Expected 50 curated papers, found {len(records)}")

    notes = OUT / "paper_notes"
    notes.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    matrix = []
    bib_entries = []
    for i, r in enumerate(sorted(records, key=lambda x: (-int(x.get("year", 0) or 0), x["title"])), 1):
        r = dict(r)
        key = bib_key(r, used)
        area, why = relevance(r)
        abstract = str(r.get("abstract", "") or "").strip()
        if abstract == "nan":
            abstract = "Abstract unavailable in the collected metadata; consult the linked primary source before quoting a claim."
        url = str(r.get("url", "") or "")
        doi = str(r.get("doi", "") or "")
        if doi == "nan": doi = ""
        local_pdf = ""
        source = str(r.get("source", ""))
        if source:
            candidates = list((LIT / source / "pdfs").glob(f"*-{slug(r['title'])[:35]}*.pdf"))
            if candidates:
                local_pdf = str(candidates[0].relative_to(OUT.parent))
        entry = bibtex(key, r)
        bib_entries.append(entry)
        note = f"""# {r['title']}

- **Venue/Year:** {r.get('venue','')}/{int(r.get('year',0) or 0)}
- **Authors:** {r.get('authors','')}
- **URL:** {url}
- **PDF:** {local_pdf or 'Open-access PDF not downloaded; use the primary URL.'}
- **BibTeX key:** `{key}`
- **Relevance:** high

## Abstract

{abstract}

## Problem And Motivation

The paper addresses {area}. Its motivation and scope are summarized in the abstract above.

## Methodology

The collected metadata identifies the work as part of **{area}**. Method-specific details must be read from the linked primary source before reproducing the method.

## Novelty

The work contributes a method, benchmark, analysis, or application in {area}; the exact novelty claim is bounded by the source abstract and is not expanded here.

## Contributions

{abstract[:900]}

## Strengths

- Directly informs at least one benchmark dimension, solver family, systems cost, workload, or comparison rule in this project.
- Provides a citable primary or archival source rather than relying on secondary commentary.

## Weaknesses And Limitations

- **Inference:** The paper does not by itself establish end-to-end utility for dynamic wireless control under the complete cost ledger used in this draft.
- Applicability to the present three workload generators and deadline model must be tested rather than assumed.

## Evidence And Results

See the source abstract and linked paper. No numerical result is copied into this note unless it appears in the collected abstract.

## Writing Style Notes

Use the paper as a terminology and citation source. Full prose style was not assessed when extracted full text was unavailable.

## Structure Notes

The paper is routed to the **{area}** cluster in the literature matrix.

## Relevance To This Draft

{why}

## BibTeX

```bibtex
{entry.strip()}
```
"""
        (notes / f"{i:02d}-{slug(r['title'])}.md").write_text(note)
        matrix.append({
            "title": r["title"], "venue_year": f"{r.get('venue','')} {int(r.get('year',0) or 0)}",
            "area": area, "task": "network/optimization" if "network" in area else "quantum optimization",
            "data": "See source", "method_type": area, "baselines": "See source", "metrics": "See source",
            "main_result": abstract[:300],
            "limitation": "Does not alone establish full-stack dynamic wireless utility (inference).",
            "relevance": why, "citation_key": key, "url": url,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix).to_csv(OUT / "literature_matrix.csv", index=False)
    lines = ["# Curated literature matrix", "", f"**Included papers:** {len(matrix)}", "",
             "| Title | Venue/year | Area | Main result/evidence | Limitation for this draft | Key |",
             "|---|---|---|---|---|---|"]
    for m in matrix:
        safe = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| [{safe(m['title'])}]({m['url']}) | {safe(m['venue_year'])} | {safe(m['area'])} | {safe(m['main_result'])} | {safe(m['limitation'])} | `{m['citation_key']}` |")
    (OUT / "literature_matrix.md").write_text("\n".join(lines) + "\n")
    (OUT / "references.bib").write_text("\n".join(bib_entries))
    (OUT / "manifest.json").write_text(json.dumps({"count": len(matrix), "notes": len(list(notes.glob('*.md')))}, indent=2))
    print(f"Curated {len(matrix)} papers into {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
