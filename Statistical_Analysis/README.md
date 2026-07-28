# Statistical Significance Analysis

Seed variance and significance testing for the ISMIR 2026 paper
*Stem-Specialized Multi-Gate Mixture-of-Experts for Joint Music Structure
Analysis*. These results are provided here rather than in the paper due to
the six-page limit.

---

## Protocol

The MMoE framework and the All-In-One baseline were each trained with three
random seeds $\{0, 1, 2\}$ under identical conditions: the same data, the
same train/test split, and the same training schedule. Only the random seed
differs between runs, controlling weight initialisation, batch shuffling,
and dropout.

For each run, per-track metrics of the best epoch were recorded for every
test track. Metrics were then averaged over seeds within each system, and
the two systems compared with a **two-sided Wilcoxon signed-rank test paired
at the track level**, with **Holm correction** across the five reported
metrics. The Wilcoxon test is used as the primary test because per-track F1
distributions are bounded and skewed; a paired t-test is reported alongside
in the full output for reference.

---

## Results

Mean ± standard deviation over three seeds:

| Metric | MMoE | All-In-One | Δ | *p* (Wilcoxon) |
|---|---|---|---|---|
| **Boundary detection** | | | | |
| F1(0.5) | 0.656 ± 0.012 | 0.653 ± 0.003 | +0.003 | 0.73 (n.s.) |
| F1(3)   | 0.744 ± 0.003 | 0.754 ± 0.003 | −0.010 | 0.19 (n.s.) |
| **Functional labeling** | | | | |
| F1(P)   | 0.659 ± 0.005 | 0.629 ± 0.009 | +0.030 | 0.12 (n.s.) |
| ACCa    | **0.617 ± 0.008** | 0.579 ± 0.007 | +0.038 | **0.011** |
| ACCs    | **0.619 ± 0.012** | 0.583 ± 0.007 | +0.036 | **0.033** |

MMoE significantly improves frame-level functional accuracy (ACCa +0.038,
ACCs +0.036; both *p* < 0.05), while the two systems are statistically
indistinguishable on boundary detection (F1(0.5), F1(3); both n.s.), as is
the boundary-dependent pairwise F-measure (F1(P); n.s.). This asymmetry is
consistent with the paper's central motivation: task-specific gating
concentrates capacity on labeling-relevant stem cues without sacrificing
boundary-detection performance.

After Holm correction across the five metrics, ACCa (0.055) and ACCs (0.133)
become marginal; both uncorrected and corrected values are reported in
`significance_report.txt` so that readers can apply their preferred
criterion.

---

## Contents

```
Statistical_Analysis/
├── significance_analysis.py      # Aggregation + Wilcoxon signed-rank tests
├── significance_report.txt       # Full output: means, std, all p-values
├── per_track_metrics/            # Raw inputs (one CSV per system per seed)
│   ├── mmoe_seed0.csv
│   ├── mmoe_seed1.csv
│   ├── mmoe_seed2.csv
│   ├── allinone_seed0.csv
│   ├── allinone_seed1.csv
│   └── allinone_seed2.csv
└── README.md
```

Each CSV contains one row per test track with the columns
`system, seed, song, F1_05, P_05, R_05, F1_3, P_3, R_3, F1_P, P_pair,
R_pair, ACCa, ACCs`.

---

## How the inputs were produced

The per-track CSVs were written by the seeded training scripts, which log
per-track metrics for the best epoch of each run:

- **MMoE** — [`mmoe/train_mmoe_seeded.py`](../mmoe/train_mmoe_seeded.py),
  run with `--seed 0`, `--seed 1`, `--seed 2`
- **All-In-One** —
  [`baselines/train_allinone_seeded.py`](../baselines/train_allinone_seeded.py),
  run with the same three seeds
