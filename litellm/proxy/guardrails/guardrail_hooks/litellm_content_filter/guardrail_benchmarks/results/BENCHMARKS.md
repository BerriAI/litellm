# Content Filter Benchmarks

## Investment Questions Eval (207 cases)

Eval set: `evals/block_investment.jsonl` — Emirates airline chatbot, "Block investment questions" policy.
85 BLOCK cases (investment advice), 122 ALLOW cases (airline queries, greetings, ambiguous terms).

### Production Results

| Approach | Precision | Recall | F1 | Latency p50 | Deps | Cost/req |
|----------|-----------|--------|----|-------------|------|----------|
| **ContentFilter (denied_financial_advice.yaml)** | **100.0%** | **100.0%** | **100.0%** | **<0.1ms** | None | $0 |
| LLM Judge (gpt-4o-mini) | — | — | — | ~200ms | API key | ~$0.0001 |
| LLM Judge (claude-haiku-4.5) | — | — | — | ~300ms | API key | ~$0.0001 |

> LLM Judge results: run with `OPENAI_API_KEY=... pytest ... -k LlmJudgeGpt4oMini -v -s`
> or `ANTHROPIC_API_KEY=... pytest ... -k LlmJudgeClaude -v -s`

### Historical Comparison (earlier iterations)

| Approach | Precision | Recall | F1 | FP | FN | Latency p50 | Extra Deps |
|----------|-----------|--------|----|----|----|-------------|------------|
| ContentFilter YAML | **100.0%** | **100.0%** | **100.0%** | 0 | 0 | <0.1ms | None |
| ONNX MiniLM | 95.3% | 96.5% | 95.9% | 4 | 3 | 2.4ms | onnxruntime (~15MB) |
| Embedding MiniLM (80MB) | 98.4% | 74.1% | 84.6% | 1 | 22 | ~3ms | sentence-transformers, torch |
| NLI DeBERTa-xsmall | 82.7% | 100.0% | 90.5% | 18 | 0 | ~20ms | transformers, torch |
| TF-IDF (numpy only) | 47.2% | 100.0% | 64.2% | 95 | 0 | <0.1ms | None |
| Embedding MPNet (420MB) | 98.3% | 68.2% | 80.6% | 1 | 27 | ~5ms | sentence-transformers, torch |

### How the ContentFilter works

The `denied_financial_advice.yaml` category uses three layers of matching:

1. **Always-block keywords** — specific phrases like "investment advice", "stock tips", "retirement planning" that are unambiguously financial. Matched as substrings.

2. **Conditional matching** — an identifier word (e.g., "stock", "bitcoin", "401k") + a block word (e.g., "buy", "should i", "best") in the same sentence. This avoids false positives like "in stock" or "bond with my team".

3. **Phrase patterns** — regex patterns for paraphrased financial advice (e.g., "put my money to make it grow", "park my cash", "spare cash"). Catches cases without explicit financial vocabulary.

4. **Exceptions** — phrases that override matches in their sentence (e.g., "emirates flight", "return policy", "gold medal", "trading card").

## Running evals

```bash
# Run content filter eval:
pytest litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/guardrail_benchmarks/test_eval.py -v -s

# Run specific eval:
pytest ... -k "InvestmentContentFilter" -v -s

# Run LLM judge evals (requires API keys):
OPENAI_API_KEY=sk-... pytest ... -k "LlmJudgeGpt4oMini" -v -s
ANTHROPIC_API_KEY=sk-... pytest ... -k "LlmJudgeClaude" -v -s
```

## Confusion Matrix Key

```
                  Predicted BLOCK    Predicted ALLOW
Actually BLOCK        TP                  FN
Actually ALLOW        FP                  TN
```

- **Precision** = TP / (TP + FP) — "When we block, are we right?"
- **Recall** = TP / (TP + FN) — "Do we catch everything that should be blocked?"
- **F1** = harmonic mean of Precision and Recall
