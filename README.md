# Stem-Specialized Multi-gate Mixture-of-Experts for Joint Music Structure Analysis

Code and pretrained models for our ISMIR 2026 paper.

> **Stem-Specialized Multi-gate Mixture-of-Experts for Joint Music Structure Analysis**
> *(authors)* — Proceedings of the International Society for Music Information
> Retrieval Conference (ISMIR), 2026.

We address joint music boundary detection and functional structure labeling.
Four stem-specialized experts (vocals+mix, drums+mix, bass+mix, others+mix) are
combined by two task-specific gating networks, so boundary detection and
functional labeling can weight the expert pool differently at every time frame.

---

## Repository structure

```
.
├── model.py                          # Shared architecture: SpecTNT-Enhanced encoder,
│                                     #   cross-attention fusion, stem encoder
├── config.py                         # All dataset and checkpoint paths — EDIT FIRST
├── requirements.txt
│
├── preprocessing/
│   ├── separate_stems_beatles.py     # HTDemucs-FT separation
│   ├── separate_stems_salami.py
│   ├── separate_stems_rwc.py
│   ├── map_annotations_beatles.py    # Map raw annotations to the 7-class taxonomy
│   ├── map_annotations_salami.py
│   ├── map_annotations_rwc.py
│   ├── extract_features_mix.py       # Mel + chroma for the mixture
│   ├── extract_features_mix_aug.py   #   ... with pitch shift / pre-emphasis
│   ├── extract_features_stem.py      # Mel + chroma for one stem
│   └── extract_features_stem_aug.py  #   ... with pitch shift / pre-emphasis
│
├── splits/
│   └── train_test_split.json         # Fixed track-level manifest
│
├── experts/                          # Stage 1 — stem-specialized experts
│   ├── train_expert_vocals.py
│   ├── train_expert_drums.py
│   ├── train_expert_bass.py
│   └── train_expert_others.py
│
├── mmoe/                             # Stage 2 — MMoE framework
│   ├── train_mmoe.py                 # Main model; also logs gate weights
│   └── train_mmoe_seeded.py          # --seed, --gating, --drop-expert
│
├── baselines/
│   ├── train_allinone.py
│   └── train_allinone_seeded.py
│
├── ablations/                        # Expert-pool ablations
│   ├── mmoe_spectnt.py
│   ├── mmoe_dual_attention.py
│   └── mmoe_allinone.py
│
├── cross_dataset/                    # Leave-one-dataset-out generalization
│   ├── mmoe_cv_beatles.py            # train SALAMI + RWC  → test Beatles
│   ├── mmoe_cv_salami.py             # train Beatles + RWC → test SALAMI
│   └── mmoe_cv_rwc.py                # train Beatles + SALAMI → test RWC
│
├── analysis/
     └── significance_analysis.py      # Multi-seed aggregation + Wilcoxon tests
```

Pretrained weights are hosted separately — see [Pretrained models](#pretrained-models).

---

## Setup

```bash
git clone https://github.com/<user>/mmoe-msa.git
cd mmoe-msa
pip install -r requirements.txt
```

Python 3.10, TensorFlow 2.19, NVIDIA RTX A6000. Edit `config.py` to point at
your data directory and expert checkpoints before running anything.

---

## Data preparation

Audio must be obtained from the original dataset sources; this repository
provides the split manifest and the full preprocessing pipeline.

```bash
# 1. Separate each track into vocals / drums / bass / other
python preprocessing/separate_stems_beatles.py

# 2. Map dataset annotations to the shared 7-class functional taxonomy
python preprocessing/map_annotations_beatles.py

# 3. Extract features for the mixture and for each stem
python preprocessing/extract_features_mix.py
python preprocessing/extract_features_stem.py --stem vocals

# 4. Augmented copies (training tracks only)
python preprocessing/extract_features_mix_aug.py
python preprocessing/extract_features_stem_aug.py --stem vocals
```

**Feature configuration.** Audio is resampled to 40,960 Hz and z-score
normalized. Mel-spectrograms use a 2,048-point FFT with 1,024-sample hop and 80
mel bands; chromagrams use 12 pitch classes at the same hop. Both are median
filtered and temporally downsampled by 20x, giving 0.5 s per frame (2 fps).
Sequences are capped at 935 frames.

**Splits.** The track-level train/test split is fixed in
`/data/dataset_splits.json`. Training tracks are augmented by pitch shifting
(±2 semitones) and pre-emphasis filtering (0.95, 0.97); test tracks are never
augmented. The same preprocessing code is used for every dataset and every stem.

---

## Reproducing the paper

### Main results

```bash
# Stage 1 — train the four experts (or download pretrained weights)
python experts/train_expert_vocals.py
python experts/train_expert_drums.py
python experts/train_expert_bass.py
python experts/train_expert_others.py

# Stage 2 — train MMoE on the frozen expert pool
python mmoe/train_mmoe.py

# Baseline
python baselines/train_allinone.py
```

### Expert-pool ablations

Each script substitutes a different model into the expert pool:

```bash
python ablations/mmoe_spectnt.py
python ablations/mmoe_dual_attention.py
python ablations/mmoe_allinone.py
```

### Gate weight analysis

Gate weights are logged automatically during MMoE training to
`<output_dir>/test_predictions/epoch_<N>/gate_weights.csv`, giving the per-track
average softmax weight of each expert under the boundary gate and the label gate.

### Cross-dataset generalization

```bash
python cross_dataset/mmoe_cv_beatles.py
python cross_dataset/mmoe_cv_salami.py
python cross_dataset/mmoe_cv_rwc.py
```

### Statistical significance

Every training run writes per-track metrics for its best epoch to
`per_track_metrics_seed<N>.csv`.

```bash
# Train both systems across seeds
bash scripts/run_seed_sweep.sh

# Aggregate and test
python analysis/significance_analysis.py \
    --system-a "mmoe/results/seed_*/best_models/per_track_metrics_seed*.csv"  --name-a MMoE \
    --system-b "baselines/results/seed_*/best_models/per_track_metrics_seed*.csv" --name-b All-In-One
```

Reports mean ± standard deviation across seeds, plus a two-sided Wilcoxon
signed-rank test paired at the track level with Holm-corrected p-values.

### Gating mechanism ablations

```bash
python mmoe/train_mmoe_seeded.py --gating multi     # full model (separate dynamic gate per task)
python mmoe/train_mmoe_seeded.py --gating shared    # one dynamic gate shared by both tasks
python mmoe/train_mmoe_seeded.py --gating static    # learned per-task weights, input-independent
python mmoe/train_mmoe_seeded.py --gating uniform   # fixed 1/N mixing
```

`--drop-expert {vocals,drums,bass,other}` removes one expert for leave-one-out
analysis.

---

## Pretrained models

Expert checkpoints and the trained MMoE model:

**(Zenodo DOI — to be added)**

```
checkpoints/
├── expert_vocals/
├── expert_drums/
├── expert_bass/
├── expert_others/
└── mmoe/
```

Set the paths in `config.py` after downloading.

---

## What is not included

- **Audio.** Beatles, SALAMI and RWC-Pop audio must be obtained from their
  original sources; the manifest references track IDs only.
- **Baseline checkpoints trained by their original authors.** Where a baseline
  is used as a component, we cite the original work rather than redistributing
  weights.

---

## Citation

```bibtex
@inproceedings{TODO2026mmoe,
  title     = {Stem-Specialized Multi-gate Mixture-of-Experts for Joint
               Music Structure Analysis},
  author    = {TODO},
  booktitle = {Proceedings of the International Society for Music Information
               Retrieval Conference (ISMIR)},
  year      = {2026}
}
```

## Acknowledgements

The expert encoders extend the dual-attention architecture of Chen et al. and
build on SpecTNT (Lu et al.). Source separation uses HTDemucs-FT (Rouard, Massa
and Défossez). Evaluation uses `mir_eval`.

## License

MIT — see [LICENSE](LICENSE).
