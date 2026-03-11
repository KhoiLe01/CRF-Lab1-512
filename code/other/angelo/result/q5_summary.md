# Question 5 Summary

- Chosen CRF C value: `1000.0`
- Chosen SVM-MC C value for Q5a: `1.0`
- Chosen SVM-MC C value for Q5b: `1.0`
- CRF training maxfun used for this run: `200`

## Accuracy Table

| Transforms | CRF letter acc | SVM-MC letter acc | CRF word acc | SVM-MC word acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.837507 | 0.699557 | 0.472230 | 0.172725 |
| 500 | 0.809146 | 0.668333 | 0.408549 | 0.137831 |
| 1000 | 0.784335 | 0.644286 | 0.359698 | 0.118058 |
| 1500 | 0.761165 | 0.621307 | 0.325967 | 0.097994 |
| 2000 | 0.731812 | 0.595580 | 0.287584 | 0.081419 |

## Observations

- CRF reuses sequence structure and is expected to retain higher word-wise accuracy as more training words are distorted.
- SVM-MC classifies letters independently, so distortion in the training images tends to hurt word-wise accuracy more sharply.
- The exact robustness trend is captured numerically in the table above and visually in the saved plots.
