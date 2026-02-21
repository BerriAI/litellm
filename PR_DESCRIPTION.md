# feat(router): Add complexity-based auto routing strategy

## Summary

This PR adds a new routing strategy called `complexity_router` that classifies requests by complexity using rule-based scoring and routes them to appropriate models - **with zero API calls and sub-millisecond latency**.

Unlike the existing `auto_router` which uses embedding-based semantic matching, this approach:
- **Zero external API calls** - all scoring is local
- **Sub-millisecond latency** - typically <1ms per classification (vs 100-500ms for embedding API)
- **Predictable behavior** - deterministic rule-based scoring
- **No training required** - works out of the box, no utterance examples needed

Inspired by [ClawRouter](https://github.com/BlockRunAI/ClawRouter).

## How It Works

The router scores each request across 7 weighted dimensions:

| Dimension | Description | Weight |
|-----------|-------------|--------|
| `tokenCount` | Short prompts = simple, long = complex | 0.15 |
| `codePresence` | Code keywords (function, class, async, etc.) | 0.20 |
| `reasoningMarkers` | "step by step", "think through", etc. | 0.25 |
| `technicalTerms` | Domain complexity indicators | 0.15 |
| `simpleIndicators` | "what is", "define" (negative weight) | 0.15 |
| `multiStepPatterns` | "first...then", numbered steps | 0.05 |
| `questionComplexity` | Multiple question marks | 0.05 |

The weighted sum maps to tiers:
- **SIMPLE** (< 0.25): Basic questions, greetings → cheap/fast models
- **MEDIUM** (0.25 - 0.50): Standard queries → balanced models
- **COMPLEX** (0.50 - 0.75): Technical, multi-part requests → capable models
- **REASONING** (> 0.75): Chain-of-thought, analysis → reasoning models

### Special: Reasoning Override
If 2+ reasoning markers are detected in the user message, the request automatically routes to REASONING tier regardless of score.

## Usage

```yaml
model_list:
  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        tiers:
          SIMPLE: gpt-4o-mini
          MEDIUM: gpt-4o  
          COMPLEX: claude-sonnet-4
          REASONING: o1-preview
```

Then use like any other model:
```python
response = litellm.completion(
    model="smart-router",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
# Routes to SIMPLE tier (gpt-4o-mini)
```

## Full Configuration Options

```yaml
complexity_router_config:
  tiers:
    SIMPLE: gpt-4o-mini
    MEDIUM: gpt-4o  
    COMPLEX: claude-sonnet-4
    REASONING: o1-preview
  
  # Optional: override tier boundaries (normalized scores)
  tier_boundaries:
    simple_medium: 0.25
    medium_complex: 0.50
    complex_reasoning: 0.75
  
  # Optional: override token count thresholds
  token_thresholds:
    simple: 50    # Below this = "short"
    complex: 500  # Above this = "long"
  
  # Optional: override dimension weights
  dimension_weights:
    tokenCount: 0.15
    codePresence: 0.20
    reasoningMarkers: 0.25
    technicalTerms: 0.15
    simpleIndicators: 0.15
    multiStepPatterns: 0.05
    questionComplexity: 0.05
  
  # Optional: fallback model
  default_model: gpt-4o
```

## Files Changed

### New Files
- `litellm/router_strategy/complexity_router/complexity_router.py` - Main router class
- `litellm/router_strategy/complexity_router/config.py` - Configuration and defaults
- `litellm/router_strategy/complexity_router/__init__.py` - Package exports
- `litellm/router_strategy/complexity_router/README.md` - Documentation
- `tests/test_litellm/router_strategy/test_complexity_router.py` - Test suite (37 tests)

### Modified Files
- `litellm/router.py` - Integration with pre_routing_hook
- `litellm/types/router.py` - New config params

## Testing

```bash
pytest tests/test_litellm/router_strategy/test_complexity_router.py -v
# 37 tests pass
```

## Use Cases

1. **Cost optimization**: Route simple queries ("What is X?") to cheap models, complex queries to capable models
2. **Latency optimization**: Simple greetings get fast responses, complex analysis gets thorough responses
3. **Resource management**: Expensive reasoning models only used when actually needed

## Comparison with auto_router

| Feature | complexity_router | auto_router |
|---------|-------------------|-------------|
| Classification | Rule-based scoring | Semantic embedding |
| Latency | <1ms | ~100-500ms (embedding API) |
| API Calls | None | Requires embedding model |
| Training | None | Requires utterance examples |
| Best For | Cost optimization | Intent routing |

---

cc @ishaan-jaff for review
