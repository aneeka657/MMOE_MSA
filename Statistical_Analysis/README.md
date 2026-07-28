# Statistical Significance Analysis

Seed variance and significance testing for the ISMIR 2026 paper,
provided here rather than in the paper due to the six-page limit.

## Protocol
MMoE and the All-In-One baseline were each trained with three random
seeds {0, 1, 2}. Per-track metrics were averaged over seeds within
each system, then compared with a two-sided Wilcoxon signed-rank test
paired at the track level, with Holm correction across the five
metrics.

## Results (mean ± std over three seeds)

| Metric | MMoE | All-In-One | p (Wilcoxon) |
|---|---|---|---|
| F1(0.5) | 0.656 ± 0.012 | 0.653 ± 0.003 | 0.73 (n.s.) |
| F1(3)   | 0.744 ± 0.003 | 0.754 ± 0.003 | 0.19 (n.s.) |
| F1(P)   | 0.659 ± 0.005 | 0.629 ± 0.009 | 0.12 (n.s.) |
| ACCa    | 0.617 ± 0.008 | 0.579 ± 0.007 | **0.011** |
| ACCs    | 0.619 ± 0.012 | 0.583 ± 0.007 | **0.033** |

MMoE significantly improves frame-level functional accuracy
(ACCa, ACCs), while the two systems are statistically
indistinguishable on boundary detection.

## Reproducing

```bash
python significance_analysis.py \
  --system-a "per_track_metrics/mmoe_seed*.csv"     --name-a MMoE \
  --system-b "per_track_metrics/allinone_seed*.csv" --name-b All-In-One
```

Full output: `significance_report.txt`
