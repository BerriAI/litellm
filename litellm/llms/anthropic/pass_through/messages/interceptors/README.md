# Messages Interceptors

Interceptors are short-circuit handlers for the `/v1/messages` path. They run **before** the normal backend call and can fully replace it with their own response.

## When to add an interceptor

Use an interceptor when you need to **replace the backend call entirely** with your own logic — for example, running an orchestration loop, synthesizing a response from multiple sub-calls, or short-circuiting to a non-LLM backend.

Use a **pre-request hook** (`_execute_pre_request_hooks` / `CustomLogger.async_pre_request_hook`) instead when you only need to **mutate request parameters** (tools, stream flag, metadata) before the normal call proceeds.

| Scenario | Use |
|---|---|
| Replace the backend call with a loop or synthetic response | Interceptor |
| Translate or strip tools before the call | Pre-request hook |
| Feature that is always active (built-in LiteLLM behavior) | Interceptor |
| Optional integration that operators register | `CustomLogger` callback |

## How to add a new interceptor

1. Create `your_feature.py` in this directory.
2. Implement `MessagesInterceptor` from `base.py`:
   - `can_handle(tools, custom_llm_provider) -> bool` — return True when your interceptor owns this request.
   - `async handle(...) -> Union[AnthropicMessagesResponse, AsyncIterator]` — do your work and return the response.
3. Register it in `__init__.py` by appending to `_interceptors`.

```python
# your_feature.py
from .base import MessagesInterceptor

class MyFeatureHandler(MessagesInterceptor):
    def can_handle(self, tools, custom_llm_provider):
        return some_condition(tools, custom_llm_provider)

    async def handle(self, *, model, messages, tools, stream, max_tokens,
                     custom_llm_provider, **kwargs):
        ...
        return response
```

```python
# __init__.py
from .your_feature import MyFeatureHandler

_interceptors = [
    AdvisorOrchestrationHandler(),
    MyFeatureHandler(),  # add here
]
```

## Existing interceptors

### `AdvisorOrchestrationHandler`

Handles `advisor_20260301` tool for providers that don't support it natively (all non-Anthropic providers for now).

**Triggers when:** `advisor_20260301` is in `tools` AND `custom_llm_provider` is not in `ADVISOR_NATIVE_PROVIDERS`.

**What it does:**
- Translates the advisor tool to a regular function tool the provider understands.
- Runs the executor model; when it calls the `advisor` tool, runs the advisor model and injects the result as a `tool_result`.
- Loops until the executor produces a final text response or `max_uses` is exceeded.
- Wraps the final response in `FakeAnthropicMessagesStreamIterator` if the caller requested streaming.
