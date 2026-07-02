# Local-First Routing

Route requests to local models first, then fall back to cloud models. This minimizes latency and cost while maintaining privacy.

## Overview

Local-first routing prioritizes local models (e.g., Ollama) for:
- **Lower latency**: Local models respond faster
- **Zero cost**: No API charges
- **Better privacy**: Data stays on your machine

## Configuration

```yaml
router_settings:
  routing_strategy: local-first-routing
  routing_strategy_args:
    local_provider: ollama
    local_fallback_order:
      - local
      - domestic_free
      - openrouter_free
      - openrouter_paid
```

## Usage

```python
import litellm

# This will prefer local models if available
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## How It Works

1. Request comes in for a model
2. Router checks if a local alternative exists
3. If local model available, use it
4. If not, fall back to cloud model

## Local Model Detection

Models are considered "local" if they contain:
- `ollama/`
- `localhost:`
- `127.0.0.1:`
- `local/`

## Related

- [Task-Aware Routing](./task-aware-routing.md)
- [China Domestic Free Sources](./china-domestic-free-sources.md)
