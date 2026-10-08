# TruthfulQA-MC1: entropy / uncertainty benchmark

## Qwen/Qwen3-4B-Instruct-2507

| Stage | Acc | 95% CI | Δ vs parent | McNemar p |
|---|---|---|---|---|
| S0 original prompt, option-text likelihood | 0.3848 | [0.352, 0.419] |  |  |
| S1 letter score (1 order, 1 wording) | 0.6544 | [0.623, 0.686] | +0.270 | 7.99e-34 |
| S3 + option-order averaging | 0.6810 | [0.647, 0.711] | +0.027 | 0.00646 |
| S4 + prompt-wording averaging (final) | 0.7063 | [0.675, 0.737] | +0.025 | 0.00222 |
| control: always the longest option | 0.3696 | [0.335, 0.404] |  |  |
| control: always the shortest option | 0.1759 | [0.151, 0.203] |  |  |

Uncertainty as an error detector (AUROC 0.5 = no information; acc@X% = accuracy on the X% most-confident questions):

| Signal | AUROC | 95% CI | @100% | @80% | @60% | @40% | @20% |
|---|---|---|---|---|---|---|---|
| [S4] max-prob | 0.858 | [0.830, 0.884] | 0.706 | 0.805 | 0.911 | 0.965 | 0.981 |
| [S4] -option-level entropy | 0.853 | [0.825, 0.880] | 0.706 | 0.802 | 0.911 | 0.965 | 0.981 |
| [S4] -MI across orders/wordings | 0.852 | [0.823, 0.879] | 0.706 | 0.805 | 0.903 | 0.965 | 0.981 |
| [S0] -token entropy of its own pick (original idea) | 0.554 | [0.513, 0.594] | 0.385 | 0.410 | 0.395 | 0.415 | 0.487 |

## Qwen/Qwen3-8B

| Stage | Acc | 95% CI | Δ vs parent | McNemar p |
|---|---|---|---|---|
| S0 original prompt, option-text likelihood | 0.2835 | [0.253, 0.315] |  |  |
| S1 letter score (1 order, 1 wording) | 0.6646 | [0.632, 0.696] | +0.381 | 8.3e-64 |
| S3 + option-order averaging | 0.7013 | [0.668, 0.734] | +0.037 | 0.000767 |
| S4 + prompt-wording averaging (final) | 0.7127 | [0.681, 0.742] | +0.011 | 0.15 |
| control: always the longest option | 0.3696 | [0.335, 0.404] |  |  |
| control: always the shortest option | 0.1759 | [0.151, 0.203] |  |  |

Uncertainty as an error detector (AUROC 0.5 = no information; acc@X% = accuracy on the X% most-confident questions):

| Signal | AUROC | 95% CI | @100% | @80% | @60% | @40% | @20% |
|---|---|---|---|---|---|---|---|
| [S4] max-prob | 0.855 | [0.826, 0.880] | 0.713 | 0.802 | 0.901 | 0.972 | 1.000 |
| [S4] -option-level entropy | 0.853 | [0.826, 0.879] | 0.713 | 0.802 | 0.899 | 0.972 | 1.000 |
| [S4] -MI across orders/wordings | 0.844 | [0.816, 0.870] | 0.713 | 0.807 | 0.882 | 0.968 | 1.000 |
| [S0] -token entropy of its own pick (original idea) | 0.580 | [0.537, 0.619] | 0.284 | 0.301 | 0.321 | 0.354 | 0.354 |

## Cross-model

Unweighted ensemble of all models: **0.7316** [0.701, 0.762]; vs best single (Qwen/Qwen3-8B): +0.0190 (McNemar p=0.0769)

| Pair | both right | only first | only second | neither | either right | same wrong answer |
|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 vs Qwen3-8B | 0.629 | 0.077 | 0.084 | 0.210 | 0.790 | 0.699 |

'same wrong answer' = of the questions both get wrong, the share where they pick the same wrong option (shared misconceptions that no ensemble can fix).

| Signal for the ensemble answer | AUROC | 95% CI | @100% | @80% | @60% | @40% | @20% |
|---|---|---|---|---|---|---|---|
| ensemble max-prob | 0.858 | [0.831, 0.883] | 0.732 | 0.826 | 0.920 | 0.975 | 1.000 |
| -cross-model disagreement (JS) | 0.769 | [0.735, 0.800] | 0.732 | 0.782 | 0.857 | 0.943 | 1.000 |