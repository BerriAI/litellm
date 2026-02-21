# Topic Blocker Benchmarks

## Investment Questions Eval (207 cases)

Eval set: `evals/block_investment.jsonl` — Emirates airline chatbot, "Block investment questions" policy.

| Approach | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN |
|----------|-----------|--------|----|----------|----|----|----|----|
| Keyword Blocker | **100.0%** | 47.1% | 64.0% | 78.3% | 40 | 0 | 45 | 122 |
| Embedding MiniLM (80MB) | 98.4% | **74.1%** | **84.6%** | **88.9%** | 63 | 1 | 22 | 121 |
| Embedding MPNet (420MB) | 98.3% | 68.2% | 80.6% | 86.5% | 58 | 1 | 27 | 121 |
| TF-IDF (no model) | 81.1% | 70.6% | 75.5% | 81.2% | 60 | 14 | 25 | 108 |

### Key takeaways

- **Keyword** has perfect precision (zero false positives) but misses 53% of investment questions — paraphrases, synonyms, and indirect references slip through.
- **MiniLM** is the best overall — 98% precision with 74% recall. Only 1 false positive on 207 cases. 80MB model, ~3ms inference on CPU.
- **MPNet** is 5x larger (420MB) but performs worse than MiniLM on this task.
- **TF-IDF** needs no model download but has 14 false positives — blocks legitimate airline questions.

## Engine Eval (34 cases)

Eval set: `evals/engine.jsonl` — synthetic policy (alpha/bravo + red/blue) testing the matching engine itself.

| Approach | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN |
|----------|-----------|--------|----|----------|----|----|----|----|
| Keyword Blocker | **100.0%** | **95.2%** | **97.6%** | **97.1%** | 20 | 0 | 1 | 13 |

The engine works well — the single miss is a unicode evasion case (Greek alpha lookalike).

## Confusion Matrix Key

```
                  Predicted BLOCK    Predicted ALLOW
Actually BLOCK        TP                  FN
Actually ALLOW        FP                  TN
```

- **Precision** = TP / (TP + FP) — "When we block, are we right?"
- **Recall** = TP / (TP + FN) — "Do we catch everything that should be blocked?"
- **F1** = harmonic mean of Precision and Recall
- **Accuracy** = (TP + TN) / Total
