# Complexity Router

A rule-based routing strategy that classifies requests by complexity and routes them to appropriate models - with zero API calls and sub-millisecond latency.

## Overview

Unlike the semantic `auto_router` which uses embedding-based matching, the `complexity_router` uses weighted rule-based scoring across multiple dimensions to classify request complexity. This approach:

- **Zero external API calls** - all scoring is local
- **Sub-millisecond latency** - typically <1ms per classification
- **Predictable behavior** - rule-based scoring is deterministic
- **Fully configurable** - weights, thresholds, and keyword lists can be customized

## How It Works

The router scores each request across 7 dimensions:

| Dimension | Description | Weight |
|-----------|-------------|--------|
| `tokenCount` | Short prompts = simple, long = complex | 0.10 |
| `codePresence` | Code keywords (function, class, etc.) | 0.30 |
| `reasoningMarkers` | "step by step", "think through", etc. | 0.25 |
| `technicalTerms` | Domain complexity indicators | 0.25 |
| `simpleIndicators` | "what is", "define" (negative weight) | 0.05 |
| `multiStepPatterns` | "first...then", numbered steps | 0.03 |
| `questionComplexity` | Multiple question marks | 0.02 |

The weighted sum is mapped to tiers using configurable boundaries:

| Tier | Score Range | Typical Use |
|------|-------------|-------------|
| SIMPLE | < 0.15 | Basic questions, greetings |
| MEDIUM | 0.15 - 0.35 | Standard queries |
| COMPLEX | 0.35 - 0.60 | Technical, multi-part requests |
| REASONING | > 0.60 | Chain-of-thought, analysis |

A prompt that matches nothing at all never reaches that table. When no dimension fires, the scorer has no evidence in either direction, so the request goes to `default_tier` (MEDIUM unless you change it) rather than falling through a score of 0.0 into SIMPLE. See [No-Signal Default](#no-signal-default) below.

## Configuration

### Basic Configuration

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

### Full Configuration

```yaml
model_list:
  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        # Tier to model mapping
        tiers:
          SIMPLE: gpt-4o-mini
          MEDIUM: gpt-4o  
          COMPLEX: claude-sonnet-4
          REASONING: o1-preview
        
        # Tier boundaries (normalized scores)
        tier_boundaries:
          simple_medium: 0.15
          medium_complex: 0.35
          complex_reasoning: 0.60

        # Tier for prompts that match nothing at all (default: MEDIUM)
        default_tier: MEDIUM
        
        # Token count thresholds
        token_thresholds:
          simple: 15    # Below this = "short" (default: 15)
          complex: 400  # Above this = "long" (default: 400)
        
        # Dimension weights (must sum to ~1.0)
        dimension_weights:
          tokenCount: 0.10
          codePresence: 0.30
          reasoningMarkers: 0.25
          technicalTerms: 0.25
          simpleIndicators: 0.05
          multiStepPatterns: 0.03
          questionComplexity: 0.02
        
        # Override default keyword lists
        code_keywords:
          - function
          - class
          - def
          - async
          - database
        
        reasoning_keywords:
          - step by step
          - think through
          - analyze
        
        # Fallback model if tier cannot be determined
        default_model: gpt-4o
```

## Usage

Once configured, use the model name like any other:

```python
import litellm

response = litellm.completion(
    model="smart-router",  # Your complexity_router model name
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
# Routes to SIMPLE tier (gpt-4o-mini)

response = litellm.completion(
    model="smart-router",
    messages=[{"role": "user", "content": "Think step by step: analyze the performance implications of implementing a distributed consensus algorithm for our microservices architecture."}]
)
# Routes to REASONING tier (o1-preview)
```

## Special Behaviors

### No-Signal Default

The scorer only recognizes what is on its keyword lists, so a prompt can be genuinely hard and still match nothing: a logic puzzle, a business tradeoff, a piece of prose to edit. Every dimension then scores 0, and treating that silence as evidence of simplicity is how unmatched traffic used to end up on the cheapest model.

When no dimension fires, the router returns `default_tier` instead of consulting the boundaries, and logs the classification with `signals=['no-signal']`:

```yaml
complexity_router_config:
  default_tier: MEDIUM  # SIMPLE | MEDIUM | COMPLEX | REASONING
```

Set `default_tier: SIMPLE` to keep unmatched traffic on the cheapest tier, which is how the router behaved before this setting existed.

`default_tier` names which tier no-signal traffic is classified as; it is resolved to a model by the same chain as any classified tier, so it can be served by its own entry in `tiers`, by `default_model`, or by the MEDIUM tier. It has to be servable by one of those whether you set it or take the default, and the config is rejected at load time when it is not, rather than failing on the first unmatched request. A partial `tiers` map is unaffected, since the proxy always derives a `default_model` from the MEDIUM-then-SIMPLE tier and the chain terminates there.

With routing plugins configured that chain stops at the tier's own models, since a model the plugins never vetted must not serve. So a plugin config has to give `default_tier` models in `tiers`, or point it at a tier that has them; `default_model` will not stand in.

The check is on the individual dimensions, not on the weighted score, because contributions cancel. `"hi, quick python question"` scores zero with three dimensions firing (short prompt, a simple indicator, a code keyword); it has real evidence of being simple and stays SIMPLE. Prompts shorter than the `simple` token threshold or longer than the `complex` one also fire `tokenCount`, so they score normally and are outside this path.

The LLM classifier falls back to this scorer when it times out or errors, so `default_tier` also decides where unmatched traffic lands during a classifier outage.

### Reasoning Override

If 2+ reasoning markers are detected in the user message, the request is automatically routed to the REASONING tier regardless of the weighted score. This ensures complex reasoning tasks get the appropriate model.

### System Prompt Handling

Reasoning markers in the system prompt do **not** trigger the reasoning override. This prevents system prompts like "Think step by step before answering" from forcing all requests to the reasoning tier.

### Code Detection

Technical code keywords are detected case-insensitively and include:
- Language keywords: `function`, `class`, `def`, `const`, `let`, `var`
- Operations: `import`, `export`, `return`, `async`, `await`
- Infrastructure: `database`, `api`, `endpoint`, `docker`, `kubernetes`
- Actions: `debug`, `implement`, `refactor`, `optimize`

## Performance

- **Classification time**: <1ms typical
- **Memory usage**: Minimal (compiled regex patterns + keyword sets)
- **No external dependencies**: Works offline with no API calls

## Comparison with auto_router

| Feature | complexity_router | auto_router |
|---------|-------------------|-------------|
| Classification | Rule-based scoring | Semantic embedding |
| Latency | <1ms | ~100-500ms (embedding API) |
| API Calls | None | Requires embedding model |
| Training | None | Requires utterance examples |
| Customization | Weights, keywords, thresholds | Utterance examples |
| Best For | Cost optimization | Intent routing |

Use `complexity_router` when you want to optimize costs by routing simple queries to cheaper models. Use `auto_router` when you need semantic intent matching (e.g., routing "customer support" queries to a specialized model).
