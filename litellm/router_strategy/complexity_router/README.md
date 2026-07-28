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

## Cache Warming

Provider prompt caches (Anthropic, Bedrock) are per-model, so a mid-session tier switch pays a fresh cache write on the new model and loses the cache-read discount. `cache_warming` keeps every tier model's prompt cache warm for active sessions: the proxy captures each session's latest payload and a background refresher replays it (`max_tokens=1`) against the other tier models before the provider's ~5 minute cache TTL expires. When the router later switches tiers, the switched-to model already has the session's prefix cached, and the routing pick prefers models whose cache is verifiably warm.

```yaml
model_list:
  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        tiers:
          SIMPLE: fast-claude
          COMPLEX: smart-claude
        session_affinity: false   # warming is the alternative to pinning; see below
        cache_warming:
          enabled: true
          refresh_interval_seconds: 270   # keep under the provider cache TTL (Anthropic: 5 min)
          session_ttl_seconds: 3600
          idle_timeout_seconds: 600       # stop warming a session this long after its last real request
          max_sessions: 1000
          # warm_models: [fast-claude, smart-claude]  # default: first member of each tier pool

general_settings:
  store_prompts_in_spend_logs: true   # consent gate; warming stores full payloads in Redis

router_settings:
  redis_host: localhost
  redis_port: 6379
```

Requirements and semantics:

- **Redis is required.** Session payloads, per-model warmth stamps, and a per-router session index live in Redis so all pods share them and a single pod (via a Redis cron lock) runs the replays. Without Redis, warming logs a warning once and no-ops; requests are unaffected. Sessions are tracked through the index rather than keyspace scans, so Redis Cluster is supported.
- **`store_prompts_in_spend_logs: true` is a prerequisite.** Warming persists full request payloads (messages, system, tools) in Redis, so it is gated on the same consent flag that governs storing prompts in spend logs. With the flag off, capture warns once and skips.
- **Only Anthropic and Bedrock models that support prompt caching are warmed.** Other models in the tier pools are left alone. Requests must carry a `metadata.session_id` and exceed the warm set's minimum cacheable token count (`prompt_cache_min_tokens`, default 1024) to be captured.
- **Spend attribution.** Replays run under the originating key: they appear in spend logs attributed to that key/team/user, tagged `litellm_cache_warming` so they are filterable in the Logs UI. A `max_tokens=1` replay of a warm prefix bills roughly 10% of the input cost. Warming stops for keys that are deleted, blocked, expired, or at 95% of their `max_budget` (fails open if the lookup errors).
- **`max_sessions`** caps concurrently warmed sessions per auto-router, enforced atomically at capture; once reached, new sessions are not admitted until existing ones expire.
- **Interplay with `session_affinity`** (default on): affinity pins a session to its first-turn model, so no tier switch happens and warming buys nothing; with affinity on, captured sessions are still warmed but the pin decides routing. Disable `session_affinity` to let per-turn classification switch tiers and have warming make those switches cache hits.

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
