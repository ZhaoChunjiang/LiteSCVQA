# Reproducibility and evaluation protocol

This document records how to reproduce the **reported comparisons** from available feature tables. It also distinguishes checks achievable from CSVs from checks that need videos.

## Required inputs

1. **MSU-SCV** distorted videos, only for feature extraction, sampling comparisons or timing. These must be obtained from the dataset provider.
2. `msu/mos.csv`: `video,mos` with one row for each distorted video; videos are identified by paths like `1/zoom_1.mp4`.
3. `msu_feats/b_*.csv`: 67 predictor columns (including `key_ratio`) plus `video` and `seconds`; the extraction program generates these files.
4. Optional `msu_deep.csv`: extracted `cg_*`, `cl_*`, `brisque*`, `sec` columns for the CNN and BRISQUE baselines.

Video IDs and quality labels must match exactly. The standard MSU release contains 1,600 distorted videos, 100 source-content groups, and 16 variants per group. **Never** randomly split individual distorted videos without grouping by source.

Use `python scripts/check_msu_inputs.py --features 'msu_feats/b_*.csv' --mos msu/mos.csv` before training.

## Primary analysis

The paper's main fixed-regressor evaluation uses 100 random source-disjoint splits: 20 held-out source contents of 100 per split. Training uses the other 80 content groups, not the test groups. The random selection is generated with `numpy.random.default_rng(0)` over sorted group IDs in `full_eval.py`. Hyperparameters of the published default GBR: 300 estimators; max depth 3; learning rate 0.05; subsample 0.8; random state 0.

```bash
python full_eval.py 100 GBR
```

This also runs a 70/30 protocol and first-30-split feature ablation, using its own explicitly documented groupings.

Do not substitute the `litescvqa_eval.py` CLI for exact article reproduction: that helper performs nested hyperparameter selection and has different model defaults.

## Deep / BRISQUE baselines

Obtain a compatible MobileNetV3-Small ONNX backbone and generate `msu_deep.csv` with `deep_feats.py`, or supply an existing matching table. The ONNX architecture and exported tensor name must match `deep_feats.py`. Regressors must be fitted only on training groups, and PCA must be fitted exclusively on the training rows.

```bash
python hybrid_eval.py 100 'BRISQUE+SVR,CNN+SVR,HC-GBR,Hybrid-early'
```

The scripts use `msu_deep.csv` and `msu_feats/b_*.csv` in the repository working directory. No pretrained backbone weights are included here.

## Metrics

- **Pooled SRCC/KRCC:** computed over all distorted videos in the held-out source groups.
- **Per-content SRCC:** rank correlation among the 16 distorted versions *within each* held-out source; average those source-level correlations.
- **Raw PLCC:** correlation between unmodified predicted and subjective scores.
- **Post-fit PLCC:** four-parameter logistic fit using the held-out subjective labels, followed by Pearson correlation; this is a VQA reporting convention, not a zero-calibration deployment performance measurement.
- The 100 repeated train/test partitions overlap; their per-split values are therefore *not independent experiments*.

## Key-frame and timing analyses

Matched-frame-count adaptive versus uniform comparisons require the original **video files**, not just the 67-D feature table. In the study, three upper bounds K=8, 16, 40 were compared, with the actual number of spatial frames matched for each video. Adaptive and uniform both used a shared low-resolution first pass, so this is **not** an equal-total-compute comparison against an independently optimized uniform-only pipeline. Small improvements in mean per-content SRCC were not statistically established.

CPU timings were measured on a **2-vCPU ARM Neoverse-V2 (aarch64) Debian 13 VM**, Python 3.12.15, OpenCV 5.0.0 with FFmpeg decoder, NumPy 2.5.4, 7 GB RAM; one Python process, with OpenCV configured for two parallel threads and decoder threads not separately capped. Counterbalanced tests on 30 source contents were repeated twice. Timing should not be generalized to a strictly single-threaded setup or different hardware.

## Limitations

- Specialist SCVQA algorithms published elsewhere are described and cited in the manuscript, but no performance numbers for those algorithms are fabricated or inferred from unrelated test protocols.
- The stall-detection rule shows failure modes for natural freeze-and-jump movements such as discrete wheel scrolling.
- Region features and full-frame blockiness should be interpreted according to the implemented operators, not as pixel-perfect segmentation.
- Raw videos and subjective ratings are under third-party dataset terms and are not included in this code repository.
- The study may be extended with independently implemented specialist baselines and cross-database validation when sufficiently reproducible implementations and data permissions exist.
