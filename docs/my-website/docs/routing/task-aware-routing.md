# Task-Aware Routing

Route requests based on task type (coding, chinese, general, english, fast). This allows you to define task-specific model preferences.

## Overview

Task-aware routing lets you specify different models for different tasks. For example:
- **coding**: Prefer fast coding models (devstral, qwen3-coder)
- **chinese**: Prefer Chinese-optimized models (qwen3, glm-4)
- **general**: Prefer general-purpose models (gemma4, nemotron)

## Configuration

```yaml
router_settings:
  routing_strategy: task-aware-routing
  routing_strategy_args:
    task_mapping:
      coding: [devstral, qwen3-coder, claude-sonnet]
      chinese: [qwen3, sf-qwen2.5-72b, claude-sonnet]
      general: [gemma4, nemotron-ultra-free, claude-sonnet]
      english: [ornith-35b, nemotron-ultra-free, claude-opus]
      fast: [deepseek-r1-14b, sf-deepseek-r1, glm-4-flash]
    default_task: general
```

## Usage

When making a request, include the `task` parameter in `metadata`:

```python
import litellm

# Pass task via metadata (recommended)
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Write a Python function"}],
    metadata={"task": "coding"}
)
```

### Alternative: Pass via kwargs

```python
# Pass task directly as kwargs
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Write a Python function"}],
    task="coding"
)
```

### HTTP API

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Write code"}],
    "metadata": {"task": "coding"}
  }'
```

## Task Types

| Task | Description | Recommended Models |
|------|-------------|-------------------|
| `coding` | Code generation, debugging | devstral, qwen3-coder |
| `chinese` | Chinese language tasks | qwen3, glm-4 |
| `general` | General purpose | gemma4, nemotron |
| `english` | English language tasks | ornith, claude |
| `fast` | Quick responses | deepseek-r1-14b |

## How It Works

1. Request comes in with `task` parameter
2. Router looks up `task_mapping[task]` to get model list
3. First available model from the list is selected
4. If no model found, falls back to `default_task`

## Related

- [Local-First Routing](./local-first-routing.md)
- [China Domestic Free Sources](./china-domestic-free-sources.md)
