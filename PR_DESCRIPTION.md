# feat(router): Add complexity-based auto routing strategy

## Summary

This PR adds a new `complexity_router` - a rule-based routing strategy that uses weighted scoring to classify requests by complexity and route them to appropriate models. Unlike the existing `auto_router` which uses semantic/embedding matching (requiring API calls), the complexity router operates entirely locally in <1ms with zero cost.

**Also included:** UI updates to make complexity routing accessible via a simple 4-dropdown interface.

## Motivation

Many users want intelligent model routing based on query complexity without:
- The latency of embedding API calls
- The cost of embedding API calls  
- The complexity of configuring semantic routes with utterances

The complexity router provides a simple, fast alternative that handles 70-80% of routing decisions well.

## How It Works

### Weighted Scoring Across 7 Dimensions

| Dimension | What It Detects | Score Range |
|-----------|-----------------|-------------|
| `tokenCount` | Short prompts → simple, long → complex | -1.0 to 1.0 |
| `codePresence` | Code keywords (function, class, python, etc.) | 0 to 1.0 |
| `reasoningMarkers` | "step by step", "think through", etc. | 0 to 1.0 |
| `technicalTerms` | Architecture, distributed, ML terms | 0 to 1.0 |
| `simpleIndicators` | "what is", "define", greetings | -1.0 to 0 |
| `multiStepPatterns` | "first...then", numbered steps | 0 to 0.5 |
| `questionComplexity` | Multiple questions | 0 to 0.5 |

### Tier Assignment

The weighted score maps to 4 tiers:
- **SIMPLE** (score < 0.25): Quick factual questions, greetings
- **MEDIUM** (0.25 ≤ score < 0.50): Moderate complexity
- **COMPLEX** (0.50 ≤ score < 0.75): Technical, code-heavy requests
- **REASONING** (score ≥ 0.75): Multi-step reasoning required

**Special Override:** If 2+ reasoning markers are detected in the user message, the request is automatically routed to REASONING tier regardless of overall score.

## Configuration

### Via proxy config.yaml

```yaml
model_list:
  - model_name: complexity_router_1
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        tiers:
          SIMPLE: gemini-2.0-flash
          MEDIUM: gpt-4o-mini
          COMPLEX: claude-sonnet-4
          REASONING: claude-opus-4
        # Optional: adjust tier boundaries (defaults shown)
        tier_boundaries:
          simple_medium: 0.25
          medium_complex: 0.50
          complex_reasoning: 0.75
        # Optional: adjust token count thresholds
        token_thresholds:
          simple: 15   # Below = "short"
          complex: 400 # Above = "long"
```

### Via UI (New!)

The UI now has a "Router Type" selector with two options:

1. **Complexity Router (Recommended)** - Simple 4-dropdown interface:
   - Simple Tasks: [model dropdown]
   - Medium Tasks: [model dropdown]
   - Complex Tasks: [model dropdown]
   - Reasoning Tasks: [model dropdown]

2. **Semantic Router** - Existing utterance-based configuration

Users can now set up smart routing in ~30 seconds by just picking 4 models.

### Programmatic Usage

```python
from litellm import Router

router = Router(model_list=[{
    "model_name": "smart-router",
    "litellm_params": {
        "model": "auto_router/complexity_router",
        "complexity_router_config": {
            "tiers": {
                "SIMPLE": "gpt-4o-mini",
                "MEDIUM": "gpt-4o", 
                "COMPLEX": "claude-sonnet-4-20250514",
                "REASONING": "claude-sonnet-4-20250514",
            }
        }
    }
}])

# Routes automatically based on complexity
response = await router.acompletion(
    model="smart-router",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)  # → Routes to gpt-4o-mini (SIMPLE)

response = await router.acompletion(
    model="smart-router", 
    messages=[{"role": "user", "content": "Think step by step about this distributed systems architecture problem..."}]
)  # → Routes to claude-sonnet-4-20250514 (REASONING)
```

## Files Changed

### New Files - Backend
- `litellm/router_strategy/complexity_router/__init__.py`
- `litellm/router_strategy/complexity_router/complexity_router.py` - Main router implementation
- `litellm/router_strategy/complexity_router/config.py` - Configuration and defaults
- `litellm/router_strategy/complexity_router/README.md` - Documentation
- `tests/test_litellm/router_strategy/test_complexity_router.py` - 37 tests

### New Files - UI
- `ui/litellm-dashboard/src/components/add_model/ComplexityRouterConfig.tsx` - 4-dropdown tier configuration component

### Modified Files - Backend
- `litellm/router.py` - Added complexity router initialization and pre-routing hook
- `litellm/types/router.py` - Added `complexity_router_config` and `complexity_router_default_model` params

### Modified Files - UI
- `ui/litellm-dashboard/src/components/add_model/add_auto_router_tab.tsx` - Added router type selector, integrated ComplexityRouterConfig
- `ui/litellm-dashboard/src/components/add_model/handle_add_auto_router_submit.tsx` - Handle complexity_router model type in submit

## UI Changes

### Router Type Selector
![Router Type Selector](docs/images/router-type-selector.png)

The "Add Auto Router" page now shows:
- **Complexity Router (Recommended)** with a green badge - default selected
- **Semantic Router** for the existing utterance-based approach

### Complexity Router Configuration
![Complexity Router Config](docs/images/complexity-router-config.png)

Simple 4-dropdown interface:
- Each dropdown shows available models
- Tooltips explain what each tier handles
- Recommendation card with suggested model types

## Testing

```bash
pytest tests/test_litellm/router_strategy/test_complexity_router.py -v
# 37 passed in 0.21s
```

Tests cover:
- Scoring logic for each dimension
- Tier assignment at boundaries
- Reasoning marker override
- Model selection
- Pre-routing hook integration
- Edge cases (empty prompts, unicode, very long prompts)
- Configuration overrides

## Performance

- **Latency:** <1ms per classification (all local regex/string matching)
- **Cost:** $0 (no API calls)
- **Memory:** Minimal (pre-compiled regex patterns)

## Inspiration

This implementation is inspired by [ClawRouter](https://github.com/BlockRunAI/ClawRouter), which uses similar weighted scoring for complexity classification.

## Checklist

- [x] Tests added (37 tests)
- [x] Backend implementation complete
- [x] UI implementation complete
- [x] Documentation in PR description
- [x] No breaking changes
- [x] Follows existing patterns (like `auto_router`)

cc @ishaan-jaff for review
