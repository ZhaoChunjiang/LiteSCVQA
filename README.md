# LiteSCVQA

A lightweight, CPU-only no-reference quality assessment model for screen content videos (online teaching and screen sharing).

LiteSCVQA selects key frames from a low-resolution inter-frame change signal, splits each key frame into text/graphic and pictorial regions, extracts readability, NSS, global and temporal (stall) features, pools them into a 67-dimensional descriptor and maps it to quality with a gradient-boosting regressor. No GPU and no deep network are required.

## Files

| File | Purpose |
|---|---|
| `litescvqa_features.py` | Key-frame selection, region partition and feature extraction (67-dim descriptor) |
| `litescvqa_stall_inject.py` | Controlled stall injection and stall-rule validation on teaching videos |
| `deep_feats.py` | BRISQUE-style features and MobileNetV3-Small (ONNX, CPU) features for baselines / LiteSCVQA+ |
| `full_eval.py` | 100 content-separated splits, regressor comparison (SVR, RF, GBR, MLP) |
| `hybrid_eval.py` | BRISQUE, MobileNetV3 and LiteSCVQA+ comparison on the same splits |
| `abl_full30.py`, `extra_eval.py` | Ablation reference, RMSE, significance tests and model size |
| `litescvqa_eval.py` | Generic training / evaluation helper |
| `make_synth.py` | Synthetic distortion generator for quick tests |
| `paper/fig1.py` | Framework figure |

## Usage

```bash
pip install -r requirements.txt

# features of one video
python litescvqa_features.py video.mp4

# batch extraction (recursive) to CSV
python litescvqa_features.py videos_dir/ -o msu_feats/b_all.csv
```

The evaluation scripts expect the MSU screen content video dataset (not included; obtain it from its authors) with features in `msu_feats/b_*.csv` and subjective scores in `msu/mos.csv` (columns `video`, `mos`). The MobileNetV3-Small ONNX model is expected at `models/mnv3s.onnx`.

## Status

Code accompanying a manuscript under review. Citation information will be added after publication.
