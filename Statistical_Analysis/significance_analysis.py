"""
Multi-seed significance analysis for ISMIR 2026 camera-ready (reviewer item 1).

Inputs: per-track CSVs produced by the seeded training scripts, one per seed
per system, each with columns:
    system, seed, song, F1_05, P_05, R_05, F1_3, P_3, R_3,
    F1_P, P_pair, R_pair, ACCa, ACCs

Usage:
    python significance_analysis.py \
        --system-a "mmoe_results_with_rwc/seed_*/best_models/per_track_metrics_seed*.csv" \
        --name-a MMoE \
        --system-b "allinone_results/seed_*/best_models/per_track_metrics_seed*.csv" \
        --name-b All-In-One

Outputs (printed and written to significance_report.txt):
  1. Per-system mean +/- std across seeds for every paper metric
     (aggregate = mean over test tracks within a seed, then mean/std over seeds).
  2. Paired Wilcoxon signed-rank test per metric: for each track, the metric
     is averaged over seeds within each system, then the two systems are
     compared pairwise over the N test tracks. A paired t-test is reported
     alongside for reference.
  3. A LaTeX-ready "mean +/- std" row for each system, and suggested
     significance markers.

Statistical choices (state these in the paper):
  - Wilcoxon signed-rank is the primary test: paired at the track level,
    non-parametric (per-track F1 distributions are bounded and skewed).
  - Averaging over seeds before pairing tests the systems' expected
    performance per track and avoids treating seeds as independent samples.
  - With 5 primary metrics, note whether conclusions survive a
    Holm-Bonferroni correction (reported below).
"""

import argparse
import glob
import sys

import numpy as np
import pandas as pd
from scipy import stats

PAPER_METRICS = ['F1_05', 'F1_3', 'F1_P', 'ACCa', 'ACCs']
ALL_METRICS = ['F1_05', 'P_05', 'R_05', 'F1_3', 'P_3', 'R_3',
               'F1_P', 'P_pair', 'R_pair', 'ACCa', 'ACCs']


def load_system(patterns, name, exclude=None):
    """patterns: one or more glob patterns (recursive '**' supported).
    exclude: list of substrings; any matching path is skipped."""
    if isinstance(patterns, str):
        patterns = [patterns]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    # de-duplicate while keeping order stable
    files = sorted(set(files))
    if exclude:
        kept = []
        for f in files:
            if any(x in f for x in exclude):
                print(f'  [skip] {f}')
            else:
                kept.append(f)
        files = kept
    if not files:
        sys.exit(f'ERROR: no CSVs match pattern(s) for {name}: {patterns}')
    print(f'{name}: loading {len(files)} file(s)')
    for f in files:
        print(f'  [load] {f}')
    frames = []
    for f in files:
        df = pd.read_csv(f)
        missing = [c for c in ['seed', 'song'] + ALL_METRICS if c not in df.columns]
        if missing:
            sys.exit(f'ERROR: {f} is missing columns: {missing}')
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df['system'] = name
    seeds = sorted(df['seed'].unique())
    n_tracks = df.groupby('seed')['song'].nunique()
    print(f'{name}: {len(files)} CSV(s), seeds={seeds}, '
          f'tracks per seed={dict(n_tracks)}')
    if n_tracks.nunique() != 1:
        print(f'  WARNING: track counts differ across seeds for {name}')
    return df


def seedwise_aggregates(df):
    """Mean over tracks within each seed -> one row per seed."""
    return df.groupby('seed')[ALL_METRICS].mean()


def mean_std_line(agg, metrics=PAPER_METRICS, decimals=3):
    parts = []
    for m in metrics:
        parts.append(f'{agg[m].mean():.{decimals}f} ± {agg[m].std(ddof=1):.{decimals}f}')
    return parts


def holm_bonferroni(pvals):
    """Return Holm-adjusted p-values in original order."""
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (len(pvals) - rank) * pvals[idx]
        running_max = max(running_max, min(adj, 1.0))
        adjusted[idx] = running_max
    return adjusted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system-a', required=True, nargs='+',
                    help="one or more glob patterns for system A per-track CSVs "
                         "(quote them; '**' matches any depth)")
    ap.add_argument('--system-b', required=True, nargs='+',
                    help='one or more glob patterns for system B per-track CSVs')
    ap.add_argument('--exclude', nargs='*', default=['ablation'],
                    help="substrings; any matching file path is skipped "
                         "(default: 'ablation', so gating-ablation CSVs are not "
                         "mixed into the main comparison). Pass --exclude with no "
                         "values to disable.")
    ap.add_argument('--name-a', default='SystemA')
    ap.add_argument('--name-b', default='SystemB')
    ap.add_argument('--out', default='significance_report.txt')
    args = ap.parse_args()

    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    df_a = load_system(args.system_a, args.name_a, exclude=args.exclude)
    df_b = load_system(args.system_b, args.name_b, exclude=args.exclude)

    # ── 1. mean ± std across seeds ────────────────────────────────────────────
    agg_a = seedwise_aggregates(df_a)
    agg_b = seedwise_aggregates(df_b)

    emit()
    emit('=' * 78)
    emit('1. AGGREGATE RESULTS ACROSS SEEDS (mean over tracks, then mean±std over seeds)')
    emit('=' * 78)
    header = f'{"Metric":<8} | {args.name_a:>20} | {args.name_b:>20} | {"Δ (A−B)":>9}'
    emit(header)
    emit('-' * len(header))
    for m in PAPER_METRICS:
        ma, sa = agg_a[m].mean(), agg_a[m].std(ddof=1)
        mb, sb = agg_b[m].mean(), agg_b[m].std(ddof=1)
        emit(f'{m:<8} | {ma:>12.3f} ± {sa:.3f} | {mb:>12.3f} ± {sb:.3f} | {ma - mb:>+9.3f}')

    emit()
    emit('LaTeX rows (paper metric order F1_0.5, F1_3, F1_P, ACCa, ACCs):')
    emit(f'  {args.name_a} & ' + ' & '.join(mean_std_line(agg_a)) + r' \\')
    emit(f'  {args.name_b} & ' + ' & '.join(mean_std_line(agg_b)) + r' \\')

    # ── 2. paired tests at the track level ───────────────────────────────────
    # Average each track's metric over seeds within each system, pair on song.
    per_track_a = df_a.groupby('song')[ALL_METRICS].mean()
    per_track_b = df_b.groupby('song')[ALL_METRICS].mean()
    common = per_track_a.index.intersection(per_track_b.index)
    only_a = per_track_a.index.difference(per_track_b.index)
    only_b = per_track_b.index.difference(per_track_a.index)
    if len(only_a) or len(only_b):
        emit()
        emit(f'WARNING: song ID mismatch — {len(only_a)} only in {args.name_a}, '
             f'{len(only_b)} only in {args.name_b}. Pairing on {len(common)} common tracks.')
    a = per_track_a.loc[common]
    b = per_track_b.loc[common]

    emit()
    emit('=' * 78)
    emit(f'2. PAIRED SIGNIFICANCE TESTS over {len(common)} test tracks '
         f'({args.name_a} vs {args.name_b})')
    emit('=' * 78)
    header = (f'{"Metric":<8} | {"mean Δ":>8} | {"Wilcoxon W":>10} | {"p (Wilcoxon)":>12} '
              f'| {"p (t-test)":>10} | {"p (Holm)":>9} | sig')
    emit(header)
    emit('-' * len(header))

    raw_p = []
    rows = []
    for m in PAPER_METRICS:
        diff = a[m].values - b[m].values
        # Wilcoxon requires at least one non-zero difference
        if np.allclose(diff, 0):
            raw_p.append(1.0)
            rows.append((m, 0.0, np.nan, 1.0, 1.0))
            continue
        w_stat, w_p = stats.wilcoxon(a[m].values, b[m].values,
                                     zero_method='pratt', alternative='two-sided')
        t_stat, t_p = stats.ttest_rel(a[m].values, b[m].values)
        raw_p.append(w_p)
        rows.append((m, diff.mean(), w_stat, w_p, t_p))

    holm = holm_bonferroni(np.array(raw_p))
    for (m, d, w, wp, tp), hp in zip(rows, holm):
        sig = '***' if wp < 0.001 else '**' if wp < 0.01 else '*' if wp < 0.05 else 'n.s.'
        emit(f'{m:<8} | {d:>+8.4f} | {w:>10.1f} | {wp:>12.2e} '
             f'| {tp:>10.2e} | {hp:>9.2e} | {sig}')

    emit()
    emit('Interpretation guide for the camera-ready:')
    emit('  * p<0.05, ** p<0.01, *** p<0.001 (uncorrected Wilcoxon).')
    emit('  - If F1_05 / F1_3 are n.s.: re-scope the boundary claim to "on par with"')
    emit('    and let the labeling metrics carry the contribution (per meta-review).')
    emit('  - Report: "significance assessed with a two-sided Wilcoxon signed-rank')
    emit('    test paired at the track level (N=<n>), metrics averaged over')
    emit('    <k> random seeds per system; Holm-corrected p-values reported."')

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nReport written to {args.out}')


if __name__ == '__main__':
    main()
