# LiteSCVQA

**LiteSCVQA** is a CPU-oriented, no-reference screen-content video quality assessment (SCVQA) research implementation for video-conferencing, screen sharing, and teaching recordings.

It combines a full-sequence low-resolution change signal, content-guided high-resolution key-frame selection, text/graphic and pictorial features, temporal diagnostics, and gradient-boosting regression. No GPU is required for the default hand-crafted-feature pipeline.

> **Research status:** Code accompanies a manuscript being prepared for journal submission. These files are research implementations, not a production service or an official benchmark release. Results depend on the same dataset, source-separated splits, feature versions, and evaluation settings.

## Method and descriptor

- Pass 1 processes all frames at 480-pixel width to compute a changed-pixel fraction and temporal features.
- Pass 2 processes a limited set of high-resolution key frames (up to 40) at width 1280.
- Each selected frame provides 19 spatial values: 8 readability-associated, 6 pictorial NSS, and 5 global/content values.
- Pooling yields **57 spatial values + 1 sharpness-fluctuation value + 8 temporal values = 66 values** returned by `extract_video_features()`.
- The **67th value** is `key_ratio = selected_frames / total_frames`; the batch CSV exporter writes it as a separate column. The regression scripts include it among the predictors. Do not treat `seconds` or `video` as predictors.
- The reference regressor is a 300-tree gradient-boosting regressor (depth 3, learning rate 0.05, subsample 0.8).

The project includes illustrative stall-injection experiments. The frame-difference stall rule can confuse natural discrete scrolling with transmission stalls; it is **not** a general-purpose, perfectly reliable stall detector.

## Files

| File | Role |
| --- | --- |
| `litescvqa_features.py` | Video decoding, key-frame selection, partition, 66-value extractor and 67-column feature export |
| `full_eval.py` | Fixed GBR/SVR/RF/MLP evaluation, source-disjoint splits and ablation on the MSU feature tables |
| `hybrid_eval.py` | BRISQUE/MobileNetV3 baselines and feature fusion (requires additional extracted features) |
| `deep_feats.py` | BRISQUE and MobileNetV3-Small ONNX feature extraction |
| `litescvqa_stall_inject.py` | Controlled synthetic stall injection and interval detection analysis |
| `litescvqa_eval.py` | General-purpose, **separately tuned** model training and evaluation; not identical to the paper's fixed-hyperparameter protocol |
| `extra_eval.py`, `abl_full30.py` | Additional metrics and feature-group diagnostics |
| `scripts/check_msu_inputs.py` | Check feature/score alignment before regression |
| `docs/REPRODUCIBILITY.md` | Data preparation, expected formats, exact evaluation boundaries and interpretation |

## Installation

Python 3.12 was used for the reported research; a CPU environment with NumPy, SciPy, OpenCV, pandas and scikit-learn is sufficient for the default method.

```bash
python -m pip install -r requirements.txt
```

MobileNetV3 experiments additionally need ONNX / ONNX Runtime and a compatible `models/mnv3s.onnx` model, which is **not supplied** in this repository.

## Feature extraction

```bash
python litescvqa_features.py path/to/example.mp4
mkdir -p msu_feats
python litescvqa_features.py msu/ -o msu_feats/b_full.csv --skip-orig
```

Single-video mode prints 66 extracted values and reports `key_ratio` in the accompanying metadata. Batch mode writes `video,key_ratio,seconds,<66 features>`. Regression consumes `key_ratio` as an additional predictor.

Expected MSU directory structure:

```text
msu/
  1/
    <distorted-video>.mp4
  ...
  100/
    <distorted-video>.mp4
  mos.csv             # columns: video,mos
```

The `video` key in the score CSV must match the relative path used by the feature CSV (e.g., `1/zoom_1.mp4`). A full benchmark contains 100 source contents, 16 distorted versions each.

## Basic reproduction

```bash
python scripts/check_msu_inputs.py --features 'msu_feats/b_*.csv' --mos msu/mos.csv
python full_eval.py 100 GBR
```

`full_eval.py` requires score tables and precomputed features at the paths shown. The exact baseline and alternate protocol instructions are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

**Important evaluation detail:** the published-style PLCC uses a four-parameter logistic fit using each test set's subjective scores. This is a post-fit *evaluation metric*, **not** an independently deployed calibrated PLCC; rank metrics do not use that calibration.

## Results and claims

On the 1,600-video MSU-SCV benchmark, CSV-level re-evaluation under 100 source-content-disjoint 80/20 splits gave median pooled SRCC **0.7994**, median post-fit PLCC **0.8857**, and median mean-per-content SRCC **0.8508** for the default GBR model. These metrics compare favourably with the BRISQUE and MobileNetV3 feature baselines implemented here. They are **not** evidence of superiority over every specialist SCVQA method.

An equal-actual-frame-count comparison with uniform sampling showed small average adaptive gains that were not statistically established; controlled CPU tests demonstrated faster-than-playback processing, **not** a speed advantage over uniform sampling.

## Dataset and availability

The MSU-SCV raw videos, subjective scores, and third-party pretrained models are **not redistributed here**. Obtain the dataset and comply with its creators' access and reuse terms. A public repository does not automatically grant redistribution rights for third-party videos or documents.

The repository is maintained by **Chunjiang Zhao**, sole author of the associated manuscript. The research reports **no specific grant funding**.

**License:** no software reuse license has yet been granted. Public readability does not imply permission to redistribute or adapt this source; licensing can be added after the author selects one.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for reproducibility limitations.
