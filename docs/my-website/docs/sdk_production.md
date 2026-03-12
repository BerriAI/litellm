import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Production Best Practices (Python SDK)

A quick reference for running LiteLLM's Python SDK reliably in production.

---

## 1. Use async calls

Always prefer `acompletion` / `aembedding` over their sync counterparts. This avoids blocking your event loop and dramatically improves throughput.

```python
import asyncio
import litellm

async def call_llm(prompt: str) -> str:
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

asyncio.run(call_llm("Hello!"))
```

---

## 2. Set timeouts

Avoid hanging requests. Set a `timeout` (seconds) on every call.

```python
response = await litellm.acompletion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    timeout=30,  # raise litellm.Timeout after 30 s
)
```

You can also set a global default:

```python
import litellm
litellm.request_timeout = 30
```

---

## 3. Use the Router for reliability

The [`Router`](./routing.md) wraps multiple model deployments and handles retries, cooldowns, and fallbacks automatically.

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": "sk-...",
            },
        },
        {
            # Fallback: cheaper / different provider
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "azure/gpt-4o",
                "api_key": "...",
                "api_base": "https://my-azure.openai.azure.com",
                "api_version": "2024-02-01",
            },
        },
    ],
    # Retry on rate-limit / transient errors
    num_retries=3,
    retry_after=5,          # seconds between retries
    # Cool a deployment down after 3 consecutive errors for 60 s
    allowed_fails=3,
    cooldown_time=60,
)

response = await router.acompletion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

See [Router docs](./routing.md) for the full list of strategies (least-busy, latency-based, usage-based, etc.).

---

## 4. Configure fallbacks

Fallbacks let you automatically switch to a different model when the primary fails.

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "gpt-4o",
            "litellm_params": {"model": "gpt-4o", "api_key": "sk-..."},
        },
        {
            "model_name": "claude-3-5-sonnet",
            "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022", "api_key": "sk-ant-..."},
        },
    ],
    fallbacks=[
        # If gpt-4o fails, try claude-3-5-sonnet
        {"gpt-4o": ["claude-3-5-sonnet"]}
    ],
    # Optionally fall back only on specific errors:
    # context_window_fallbacks=[{"gpt-4o": ["claude-3-5-sonnet"]}],
)

response = await router.acompletion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

:::tip
Use `context_window_fallbacks` to automatically retry with a model that has a larger context window when you hit a token limit error.
:::

---

## 5. Cap spend with BudgetManager

Use [`BudgetManager`](./budget_manager.md) to set per-user or global spending caps.

```python
from litellm import BudgetManager, completion

budget_manager = BudgetManager(project_name="my-app")

user_id = "user-42"

if not budget_manager.is_valid_user(user_id):
    budget_manager.create_budget(total_budget=5.0, user=user_id)  # $5 cap

if budget_manager.get_current_cost(user=user_id) <= budget_manager.get_total_budget(user_id):
    response = completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
    budget_manager.update_cost(completion_obj=response, user=user_id)
else:
    raise Exception("Budget exceeded for user")
```

For a global cap across all calls:

```python
import litellm
litellm.max_budget = 10.0  # $10 hard cap — raises BudgetExceededError when hit
```

---

## 6. Enable caching

Avoid redundant API calls (and cost) by caching repeated prompts.

```python
import litellm
from litellm.caching import Cache

litellm.cache = Cache()  # in-memory by default; pass type="redis" for shared cache

response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    caching=True,
)
```

See [Caching docs](./caching/all_caches.md) for Redis, S3, and Qdrant options.

---

## 7. Handle errors gracefully

LiteLLM maps all provider errors to OpenAI-compatible exceptions. Catch them at the right level.

```python
import litellm

try:
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
except litellm.RateLimitError as e:
    # 429 from any provider
    print("Rate limited — back off and retry:", e)
except litellm.ContextWindowExceededError as e:
    # Prompt too long
    print("Context window exceeded:", e)
except litellm.APIConnectionError as e:
    # Network / provider unreachable
    print("Connection error:", e)
except litellm.Timeout as e:
    print("Request timed out:", e)
```

Full list of exceptions: [Exception mapping](./exception_mapping.md).

---

## 8. Use structured logging

Track latency, cost, and errors in production by attaching a custom logger.

```python
import litellm

def log_success(kwargs, response_obj, start_time, end_time):
    cost = litellm.completion_cost(completion_response=response_obj)
    print(f"model={kwargs['model']} cost=${cost:.6f} latency={end_time - start_time:.2f}s")

def log_failure(kwargs, exception, start_time, end_time):
    print(f"FAILED model={kwargs['model']} error={exception}")

litellm.success_callback = [log_success]
litellm.failure_callback = [log_failure]
```

LiteLLM also ships integrations for [Langfuse, Helicone, Datadog, and many more](./observability/).

---

## 9. Checklist

| Item | Why |
|---|---|
| `acompletion` / `aembedding` | Non-blocking async calls |
| `timeout=` on every call | No hung requests |
| `Router` with `num_retries` | Automatic retry on transient errors |
| `fallbacks` configured | Survive provider outages |
| `BudgetManager` or `litellm.max_budget` | Prevent runaway spend |
| Caching enabled | Reduce cost on repeated prompts |
| Exception handling | Graceful degradation |
| Success/failure callbacks | Observability in production |
