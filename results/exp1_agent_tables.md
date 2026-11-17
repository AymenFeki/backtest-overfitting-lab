Cells are FDR / pFDR / power in percent; `-` means undefined.

## Grid search against the LLM agent

| Filter | No signal, grid | No signal, agent | Medium, grid | Medium, agent |
|---|---|---|---|---|
| Naive Sharpe | 4.0 / 100.0 / - | 0.0 / - / - | 1.6 / 2.2 / 16.4 | 1.1 / 1.6 / 37.4 |
| Holdout | 0.0 / - / - | 0.0 / - / - | 0.3 / 0.6 / 6.6 | 0.0 / 0.0 / 17.9 |
| Walk-forward | 0.5 / 100.0 / - | 0.0 / - / - | 0.7 / 1.1 / 11.8 | 0.0 / 0.0 / 29.2 |
| Deflated Sharpe (paper) | 0.0 / - / - | 0.0 / - / - | 0.0 / - / 0.0 | 0.0 / 0.0 / 0.2 |
| Deflated Sharpe (variant) | 0.0 / - / - | 0.0 / - / - | 0.2 / 0.7 / 4.4 | 0.0 / 0.0 / 26.0 |
| CPCV | 1.0 / 100.0 / - | 3.3 / 100.0 / - | 1.1 / 2.1 / 8.1 | 0.0 / 0.0 / 22.9 |
| Benjamini-Hochberg | 0.0 / - / - | 0.0 / - / - | 1.1 / 2.1 / 18.5 | 1.4 / 2.0 / 54.1 |
| Benjamini-Yekutieli | 0.0 / - / - | 0.0 / - / - | 0.2 / 0.6 / 8.1 | 0.0 / 0.0 / 38.5 |
| Romano-Wolf | 0.5 / 100.0 / - | 3.3 / 100.0 / - | 0.8 / 1.5 / 9.5 | 1.1 / 1.6 / 42.9 |
| PBO (best strategy only) | 1.0 / 100.0 / - | 0.0 / - / - | 0.8 / 1.8 / 1.8 | 0.0 / 0.0 / 1.2 |

## LLM agent by signal type

| Filter | Momentum | Cross-sectional | Mean reversion |
|---|---|---|---|
| Naive Sharpe | 0.0 / 0.0 / 33.0 | 3.3 / 4.5 / 38.5 | 0.0 / 0.0 / 40.7 |
| Holdout | 0.0 / 0.0 / 16.6 | 0.0 / 0.0 / 15.6 | 0.0 / 0.0 / 21.7 |
| Walk-forward | 0.0 / 0.0 / 30.2 | 0.0 / 0.0 / 28.7 | 0.0 / 0.0 / 28.5 |
| Deflated Sharpe (paper) | 0.0 / 0.0 / 0.5 | 0.0 / 0.0 / 0.2 | 0.0 / - / 0.0 |
| Deflated Sharpe (variant) | 0.0 / 0.0 / 24.3 | 0.0 / 0.0 / 26.7 | 0.0 / 0.0 / 27.1 |
| CPCV | 0.0 / 0.0 / 20.5 | 0.0 / 0.0 / 21.5 | 0.0 / 0.0 / 26.8 |
| Benjamini-Hochberg | 0.7 / 1.0 / 49.2 | 3.6 / 4.7 / 57.5 | 0.0 / 0.0 / 55.5 |
| Benjamini-Yekutieli | 0.0 / 0.0 / 35.5 | 0.0 / 0.0 / 40.3 | 0.0 / 0.0 / 39.8 |
| Romano-Wolf | 0.0 / 0.0 / 36.5 | 3.3 / 4.3 / 44.9 | 0.0 / 0.0 / 47.5 |
| PBO (best strategy only) | 0.0 / 0.0 / 1.5 | 0.0 / 0.0 / 0.4 | 0.0 / 0.0 / 1.8 |
