# Reported research results (provenance and scope)

These are summaries of the author's saved experiment tables and follow-up CSV-level re-evaluations. They are **not** new performance measurements produced by uploading this repository and they are **not** comparable to specialist methods evaluated on different databases.

## 100 source-content-disjoint 80/20 splits, MSU-SCV

1,600 distorted clips from 100 sources (16 per source). The reported figures are medians across 100 random group splits. PLCC uses a **test-set-fitted four-parameter logistic mapping**, unlike raw PLCC.

| Features / regressor | Median pooled SRCC | Median post-fit PLCC | Median per-content SRCC |
| --- | ---: | ---: | ---: |
| BRISQUE + SVR | 0.6936 | 0.8047 | 0.8190 |
| MobileNetV3-Small features + SVR | 0.5707 | 0.6932 | 0.8306 |
| LiteSCVQA (67 D) + GBR | **0.7994** | **0.8857** | **0.8508** |
| LiteSCVQA + MobileNetV3 features | 0.8023 | 0.8911 | 0.8668 |

Note: the deep feature baseline requires a separate ONNX model, which is not included here. The combined version is not the default CPU-only/no-deep-network configuration.

## Non-overlapping five-fold source-content evaluation

Each of the 100 contents is held out once and predicted by a model trained on the other folds.

| Feature set / regressor | Mean per-content SRCC across 100 sources |
| --- | ---: |
| BRISQUE + GBR | 0.7565 |
| MobileNetV3 + GBR | 0.6989 |
| LiteSCVQA + GBR | **0.8523** |

These metrics summarize a *different* experiment from the 100 repeated 80/20 splits and should not be pooled into a single result.

## Equal-actual-frame-count spatial sampling comparison

The low-resolution first pass and temporal feature values were shared between the methods in each video. Only selected spatial frame locations differed.

| Maximum K | Adaptive mean per-content SRCC | Uniform mean per-content SRCC | Difference |
| --- | ---: | ---: | ---: |
| 8 | 0.8516 | 0.8418 | +0.0098 |
| 16 | 0.8540 | 0.8441 | +0.0099 |
| 40 | 0.8546 | 0.8407 | +0.0139 |

Source-content paired confidence intervals included zero at all three budgets. **Do not claim a statistically established adaptive sampling gain.**

## Counterbalanced CPU timing on 30 contents

One Python process on a 2-vCPU ARM Neoverse-V2 (aarch64) VM; OpenCV configured for 2 parallel threads, FFmpeg decoder-thread count not explicitly capped; two repeats per selected content. Both alternatives are charged for the common first pass.

| Maximum K | Adaptive wall time (mean s/video) | Uniform wall time (mean s/video) | Adaptive throughput (input fps) |
| --- | ---: | ---: | ---: |
| 8 | 4.003 | 4.001 | 74.3 |
| 16 | 4.610 | 4.612 | 64.5 |
| 40 | 6.091 | 6.077 | 48.8 |

No statistically established wall-time advantage for adaptive sampling; a baseline using independent sparse seeking and no low-resolution temporal pass can be faster. These values are specific to the measured environment.

## Scope of conclusions

These data support the performance of the evaluated model **relative to the implemented baselines**, not a universal SCVQA state-of-the-art claim. Specialist NR-SCVQA methods from other publications were not reimplemented without a verified compatible source release. The stall heuristic has content-dependent detection limitations. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for protocol details.
