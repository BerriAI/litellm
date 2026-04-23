# Agentic Loop Hook

Build a `CustomLogger` callback that intercepts a model response, fulfills tool calls server-side, and reruns the model — transparently to the caller.

:::info Supported call types
- `async` only (sync calls do not trigger the hook)
- Non-streaming only (streaming responses cannot be inspected for tool calls)
- Works on both `/v1/messages` and `/v1/chat/completions`
:::

## Implement the callback

Override two methods on `CustomLogger`:

```python
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch

MY_TOOL = "my_tool"

class MyToolCallback(CustomLogger):

    async def async_should_run_agentic_loop(
        self, response, model, messages, tools, stream, custom_llm_provider, kwargs
    ):
        # Return (True, context_dict) if there are tool calls to handle
        content = getattr(response, "content", None) or []
        calls = [b for b in content if isinstance(b, dict)
                 and b.get("type") == "tool_use" and b.get("name") == MY_TOOL]
        if not calls:
            return False, {}
        return True, {"tool_calls": calls}

    async def async_build_agentic_loop_plan(
        self, tools, model, messages, response,
        anthropic_messages_provider_config,
        anthropic_messages_optional_request_params,
        logging_obj, stream, kwargs,
    ):
        calls = tools["tool_calls"]
        results = [f"result for {c['input']}" for c in calls]  # your logic here

        follow_up = messages + [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]}
                for c in calls
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c["id"], "content": results[i]}
                for i, c in enumerate(calls)
            ]},
        ]
        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(messages=follow_up),
        )
```

For `/v1/chat/completions`, override `async_build_chat_completion_agentic_loop_plan` instead — same idea, `optional_params` replaces `anthropic_messages_optional_request_params`.

## Register it

```python
import litellm
litellm.callbacks = [MyToolCallback()]
```

Or in `config.yaml`:

```yaml
litellm_settings:
  callbacks: ["my_module.MyToolCallback"]
```

## `AgenticLoopPlan` fields

| Field | Effect |
|---|---|
| `run_agentic_loop=True` + `request_patch` | Reruns the model with the patched request |
| `response_override` | Returns this value directly to the caller (no rerun) |
| `terminate=True` | Stops the loop, returns the current response |
| `run_agentic_loop=False` (default) | Skips; next callback is checked |

`AgenticLoopRequestPatch` accepts: `model`, `messages`, `tools`, `max_tokens`, `optional_params`, `kwargs`.

## Loop safety

- Default max reruns: `3` — override per-request with `kwargs["max_agentic_loops"]`
- Identical tool-call fingerprints abort the loop automatically
- Current depth is in `kwargs["_agentic_loop_depth"]`

## Examples in this repo

- `litellm/integrations/compression_interception/handler.py`
- `litellm/integrations/websearch_interception/handler.py`
