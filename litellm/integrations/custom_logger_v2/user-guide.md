# CustomLoggerV2 user guide

> [!WARNING]
> **Feature in progress.** CustomLoggerV2 is still being built and cannot yet be used for everything a `CustomLogger` does. These parts are missing or not wired up on `main`:
>
> | Item | State on `main` | What a callback author should do |
> | --- | --- | --- |
> | `Respond` | Type exists and hooks allow it, but the proxy raises an uncaught `V2Outcome` | Do not return it |
> | `Reroute` | Stored on the request scope, read by nobody | Do not return it |
> | `pre_call` patches to `metadata`, `optional_params`, `request_headers`, `guardrail_info` | Allowed by the declaration, `InvalidPatchError` when applied on the proxy | Patch only `model`, `messages`, `input`, `tools` |
> | `must_deliver` | Not implemented. Off-path delivery is in-process and best effort. | `awaited=True` plus your own durable write |
> | `StreamTransformer` and `stream_chunk` | Not implemented, not subscribable | Keep chunk logic in a legacy logger |
> | `call_ended` | Subscribable, never dispatched | Use `attempt_ended` |
> | `prompt_resolve`, `pre_route`, `audit_event` | In the enum, no V2 call site | Keep on `CustomLogger` |
> | `opt_in` activation and admin opt-out | The proxy never fills in the names | Use `default_on` or `required` |
> | Config registration | `litellm_settings.callbacks` should accept a V2 instance; not run in a live proxy | Verify before relying on it |

## What it is

CustomLoggerV2 is the callback API where each handler declares the hook it runs on, the fields it reads, the fields it writes and whether it may stop the request. LiteLLM uses those declarations to order callbacks, to build only the data someone asked for, and to keep read-only callbacks off the request path.

This guide is for people writing callbacks in Python. It describes `main` as of 2026-11-16. Both contract PRs are merged: [#51214](https://github.com/BerriAI/litellm/pull/51214) (types, dispatcher, call sites) and [#51526](https://github.com/BerriAI/litellm/pull/51526) (typed fields, Rust callbacks, proxy request scope). The package is `litellm/integrations/custom_logger_v2/`.

## Quick start

Subclass `CustomLoggerV2`, mark each handler with `@on`, and put an instance in `litellm.callbacks`. There is nothing to switch on: V2 dispatch is active as soon as one V2 logger is in the list.

```python
import litellm
from litellm.integrations.custom_logger_v2 import CallContext, CustomLoggerV2, Field, Hook, on


class UsageLog(CustomLoggerV2):
    @on(Hook.attempt_ended, reads={Field.model, Field.outcome, Field.usage, Field.cost})
    def record(self, view, ctx: CallContext[None]) -> None:
        print(ctx.call.ids.call_id, view.model, view.outcome, view.cost.response_cost)


litellm.callbacks = [UsageLog()]
```

Every handler takes the same two arguments.

- `view` exposes the declared fields as attributes (`view.model`, `view.usage`). A field the hook point has no value for raises `AttributeError`, so use `getattr(view, "usage", None)` for optional data.
- `ctx` is a `CallContext` with `ctx.hook`, `ctx.call` (ids, call type, target model and provider, timing, cache hit) and `ctx.state` (see request state below). `ctx.call` is `None` on hooks that have no call, such as `list_models`.

A handler that only reads may be a plain function, and then it runs in a worker thread. A handler that writes, may abort, or is `awaited` must be `async def`, or the class definition raises `InvalidDeclarationError`.

On the proxy, list the instance under `litellm_settings.callbacks` as `my_module.my_instance`, the same way as a legacy custom logger. That path goes through the callback manager, which accepts V2 instances, but it has not been run in a live proxy.

### Request state

To carry data between two hooks of one request, declare a `request_state` class on the logger. LiteLLM creates one instance per request, lazily, and passes it as `ctx.state`.

```python
from dataclasses import dataclass, field

class PiiMask(CustomLoggerV2):
    @dataclass
    class request_state:
        tokens: dict[str, str] = field(default_factory=dict)

    @on(Hook.pre_call, reads={Field.messages}, writes={Field.messages})
    async def mask(self, view, ctx):
        ctx.state.tokens["<EMAIL_1>"] = "a@b.com"
        ...
```

Request state only spans hooks that run inside the proxy's request scope. Outside the proxy the SDK opens a new scope for each hook, so `request_state` does not carry from one hook to the next there; key your own state by `ctx.call.ids.call_id` instead. Do not keep per-request data on `self`.

## Hooks

A hook is a named point in a request's life. Request-path hooks run before the caller gets a response and can change or stop the request. Logging hooks run on the logging plane and can never fail the request.

| Hook | When it runs | May write | Outcomes allowed |
| --- | --- | --- | --- |
| `pre_call` | Proxy, after auth, before routing | `messages`, `input`, `tools`, `model`, `optional_params`, `metadata`, `request_headers`, `guardrail_info`, `budget_reservation`, `rate_limit_statuses` | Abort, Respond, Reroute |
| `during_call` | Proxy, concurrently with the provider call | none | Abort, Respond |
| `filter_deployments` | Router, when candidates are chosen | `candidate_deployments` (narrow only) | Abort |
| `pre_deployment_call` | SDK, just before the provider request | `provider_request_body`, `request_headers` | Abort, Respond |
| `post_deployment_success` | SDK, after a deployment answers | `response` | Abort, Respond |
| `post_deployment_failure` | SDK, after a deployment fails | `error` | Abort |
| `post_call_success` | Proxy, before the response is returned | `response`, `guardrail_info` | Abort, Respond |
| `post_call_failure` | Proxy, after the request failed | `error` | Abort |
| `response_headers` | Proxy, when response headers are built | `response_headers` | none |
| `list_models` | Proxy, on `/models` | `model_list` | Abort |
| `mcp_tool_result` | After an MCP tool call | `tool_result` | Abort, Respond |
| `log_prepare` | Logging, before success loggers, success only | `standard_logging_payload` | none |
| `provider_request_sent` | Logging, when the provider request goes out | none | none |
| `provider_response_received` | Logging, when the provider answers | none | none |
| `attempt_ended` | Logging, once per attempt, success or failure | none | none |
| `router_event` | Logging, on fallback and group rate limit events | none | none |

The table is `HOOK_SPECS` in `hooks.py`. Declaring a write the hook does not allow, or `may_abort=True` on a hook that allows no outcome, fails when the class is defined. Returning an outcome type the hook does not allow is caught only at runtime, by a dispatcher built with `enforce=True`.

Five hooks are in the `Hook` enum but should not be used yet. `stream_chunk` rejects subscriptions. `call_ended` accepts them but is never dispatched, so settle on `attempt_ended`. `prompt_resolve`, `pre_route` and `audit_event` have no V2 call site on `main`.

For a logger that only records what happened, `attempt_ended` is the hook to use. It replaces both the success and the failure event, and `view.outcome` tells them apart.

## Declarations

`@on(hook, reads=..., writes=..., may_abort=..., parallel=..., awaited=...)` is the whole declaration. What you declare decides where the handler runs.

| You declare | The handler | Must be async |
| --- | --- | --- |
| only `reads` | runs off the request path, after the hook has returned to the caller; errors are logged and counted, never raised | no |
| `reads` and `awaited=True` | runs after the writers and is awaited before the hook returns; cannot write or abort | yes |
| `writes` | blocks the hook, runs in list order relative to anything touching the same fields, returns a `Patch` or `None` | yes |
| `may_abort=True` | blocks the hook, runs in its own serial step, may return an outcome | yes |
| `may_abort=True, parallel=True` | runs concurrently with the other parallel handlers as the last step; cannot write | yes |

Rules the declaration is checked against:

- One handler per hook per class. A subclass inherits its parents' handlers and may override one by redeclaring that hook.
- `writes` must be a subset of what the hook allows, and never a derived field such as `usage` or `key_identity`.
- A logger that writes or may abort cannot be removable by the caller, so it cannot set `honors_no_log=True` or allow caller opt-out.
- `during_call` handlers with `may_abort=True` are parallel by default.

**Reads are not enforced in production today.** The process-wide dispatcher is built with `enforce=False`, so `view` will hand over an undeclared field if the hook point happens to have it. Declare every field you read anyway: the declaration drives ordering, decides which derived fields get built, and a dispatcher built with `enforce=True` raises `UndeclaredReadError` (use that in tests).

## Fields

A field is one named piece of request data, a member of the `Field` enum. Which fields have a value depends on the hook point.

| Where | Fields with a value |
| --- | --- |
| Proxy `pre_call` | `model`, `messages`, `input`, `tools` when the request has them, `key_identity`, `key_limits` |
| Proxy `post_call_success` | the request fields above, `response`, `key_identity` |
| Proxy `post_call_failure` | the request fields, `error`, `key_identity`, `usage` |
| Proxy `response_headers` | `response_headers`, `key_identity` |
| Logging hooks | `model`, `messages`, `input`, `optional_params`, `standard_logging_payload`, plus the typed records below |
| `attempt_ended` only | `outcome`, `input_sequence_length` |

Typed records on the logging hooks are small frozen pydantic models from `litellm/types/integrations/custom_logger_v2.py`: `usage`, `cost`, `deployment`, `request_tags`, `guardrail_info`, `attempt_timing`, `rate_limit_state`, `logged_request`, `logged_key_identity`, `logged_model_group`, `metadata_labels`, `mcp_tool_call`, `log_id`. They are built lazily and only when some selected logger declared them.

Prefer typed records over `standard_logging_payload`. For example `view.usage.logged.total_tokens` and `view.cost.response_cost` need no payload at all. Members of a record are optional: `view.usage.logged` and `view.cost.response_cost` can be `None`, so check before using them.

A record whose source data had an unexpected type carries the offending member names in `.rejected` and leaves those members as `None`.

Two escape hatches exist for migration: `deprecated_kwargs` is the legacy kwargs dict and `deprecated_response_obj` is the legacy response object. Reading `deprecated_kwargs` counts as reading every field, which puts the handler behind every writer.

## Changing or stopping a request

A handler never mutates what it is given. It returns a value that says what should change.

### Patch

`Patch().set(field, value)` replaces a field. `Patch().merge(field, {...})` adds keys to a mapping field (`metadata`, `optional_params`, `request_headers`, `response_headers`, `guardrail_info`, `dynamic_params`, `provider_request_body`, `standard_logging_payload`). Calls chain, and every patched field must be in the handler's `writes`.

```python
import re

from litellm.integrations.custom_logger_v2 import Abort, CustomLoggerV2, Field, Hook, Patch, on


def redact(content):
    return re.sub(r"\S+@\S+", "[masked]", str(content))


class MaskAndGate(CustomLoggerV2):
    @on(Hook.pre_call, reads={Field.messages}, writes={Field.messages}, may_abort=True)
    async def check(self, view, ctx):
        if any("sk-" in str(m.get("content")) for m in view.messages):
            return Abort(status=400, message="request contains an API key", headers={"x-blocked-by": "mask-and-gate"})
        return Patch().set(Field.messages, [{**m, "content": redact(m["content"])} for m in view.messages])

    @on(Hook.response_headers, writes={Field.response_headers})
    async def headers(self, view, ctx):
        return Patch().merge(Field.response_headers, {"x-masked": "1"})
```

A patch applied at `log_prepare` is seen by every later logger, legacy ones included. This was checked with a V2 logger adding `request_tags` ahead of a legacy `async_log_success_event`.

### Outcomes

| Outcome | Meaning | What happens on `main` |
| --- | --- | --- |
| `Abort(status, message, headers=, code=, body=, exception=)` | Stop the request with an error | Works. The proxy raises `HTTPException(status, {"error": message}, headers)`, or your own `exception` if you pass one. Later callbacks on that hook do not run. |
| `Respond(response, status=200)` | Stop and return this response instead | **Not wired.** At proxy `pre_call` it raises an internal `V2Outcome` that nothing catches. |
| `Reroute(model)` | Continue, but send the call to another model | **Not wired.** It is recorded on the request scope and nothing reads it, so the request goes to the original model. |

An outcome requires `may_abort=True` and must be one the hook allows. `Abort` and `Respond` may carry a `patch`, limited to `guardrail_info` and `metadata`.

On the three failure-side hooks (`post_call_failure`, `post_deployment_failure`, `response_headers`) an exception raised by a handler is swallowed and counted. On every other request-path hook it propagates to the caller, so raise only when you mean to fail the request.

### What the proxy cannot apply yet

At proxy `pre_call`, a patch can set `model`, `messages`, `input` and `tools`. Patching `metadata`, `optional_params`, `request_headers` or `guardrail_info` is accepted by the declaration but raises `InvalidPatchError` ("metadata has no place in the legacy kwargs of this hook") when it is applied. Only `metadata` was run; the other three follow from the same key table. A V2 handler cannot write `deprecated_kwargs` either, so until this is fixed keep that logic in a legacy logger.

## Ordering

Callbacks run in the order they appear in `litellm.callbacks`. Legacy loggers, V2 loggers and Rust callbacks share that one line. Each hook is planned from the declarations of the loggers selected for the request:

1. **Serial steps.** Handlers that write or may abort, excluding parallel ones. A handler that may abort starts a new step, after everything before it in the list. A later writer can still join that step when it writes a field the aborting handler reads, since it must not run before that reader. A writer that cannot abort is placed after any earlier handler that writes a field it touches and not before an earlier handler that reads a field it writes. Handlers that touch unrelated fields share a step and run concurrently. Patches from one step are applied in list order before the next step starts.
2. **Parallel step.** Handlers declared `parallel=True` run together, last. On `during_call` the first exception is raised at once, without waiting for the others. The other handlers are not cancelled and run to completion.
3. **Legacy read-only loggers** run inline.
4. **Awaited readers** run concurrently and are awaited.
5. **Off-path readers** are handed to a background task after the hook returns.

If a serial or parallel handler returns `Abort`, nothing after it runs, including steps 3 to 5.

On hooks that allow writes, a legacy logger counts as writing everything, so it always gets its own step, and a V2 reader listed after it sees its changes. To make one logger see another's patch, list it later and declare the field in `reads`.

## Delivery and opt-out

A read-only handler is delivered **best effort**. It runs as a background task in the same process after the hook returns. If it raises, the error is logged as a warning (`custom logger v2 <name> failed on <hook>`) and counted in `Dispatcher.failures`. Nothing retries it, and a process that exits first loses it. `InProcessSink.drain()` waits for pending deliveries, which is what tests should call.

`awaited=True` moves a read-only handler onto the hook itself: the hook does not return until it finishes. Use it when a later step depends on the handler having run, and expect it to add its latency to that hook. Its errors are still swallowed and counted.

`must_deliver`, a guaranteed-delivery lane, was agreed in design but is not in the code. If losing an event is unacceptable, use `awaited=True` and do your own durable write inside the handler.

Who may switch a logger off is set by a `Selection` on the class:

```python
from litellm.integrations.custom_logger_v2 import Activation, OptOutBy, Selection

class UsageLog(CustomLoggerV2):
    selection = Selection(
        name="usage_log",
        activation=Activation.default_on,
        opt_out_by=frozenset({OptOutBy.admin, OptOutBy.caller}),
        honors_no_log=True,
    )
```

- `Activation.required` always runs. `default_on` runs unless opted out. `opt_in` runs only when the request asks for it by name.
- `honors_no_log=True` skips the logger when the request sets `no-log`. The default is `False`, so a V2 logger records `no-log` requests unless you set it.
- `OptOutBy.caller` lets a key or request header disable the logger through the proxy's dynamic callback disabling.

On the proxy only `no-log` and caller opt-outs are filled in from the request today. Nothing sets the `requested` or admin opt-out names, so `opt_in` loggers never run there yet.

## Streaming

V2 has no per-chunk hook yet. `Hook.stream_chunk` is marked not subscribable, and `@on(Hook.stream_chunk, ...)` fails with "this hook point has no V2 subscription support yet". The `StreamTransformer` from the design doc is not in the code.

What works today for a streamed call:

- `ctx.call.kind.stream` tells you the call is streamed, and `ctx.call.timing.first_token` is the slot for time to first token. It was empty in a mocked stream, so check it for `None`.
- `attempt_ended` fires once after the stream completes, with usage and cost for the whole stream.
- To inspect or rewrite chunks, keep a legacy `CustomLogger` with `async_post_call_streaming_iterator_hook` or `async_post_call_streaming_hook` next to your V2 logger. Both kinds can sit in the same callback list.

## Migrating from CustomLogger

You do not have to migrate. A legacy `CustomLogger` is wrapped automatically, keeps its position in the list and behaves as before. Migrate a logger when you want it off the request path, want typed fields instead of `kwargs`, or want its ordering to follow from what it touches.

| Legacy method | V2 hook |
| --- | --- |
| `log_success_event`, `async_log_success_event`, `log_failure_event`, `async_log_failure_event` | `attempt_ended` (check `view.outcome`) |
| `logging_hook`, `async_logging_hook` | `log_prepare` |
| `log_pre_api_call` | `provider_request_sent` |
| `log_post_api_call` | `provider_response_received` |
| `async_pre_call_hook` | `pre_call` |
| `async_moderation_hook` | `during_call` |
| `async_post_call_success_hook` | `post_call_success` |
| `async_post_call_failure_hook` | `post_call_failure` |
| `async_post_call_response_headers_hook` | `response_headers` |
| `async_filter_deployments` | `filter_deployments` |
| `async_pre_call_deployment_hook`, `async_pre_request_hook` | `pre_deployment_call` |
| `async_post_call_success_deployment_hook` | `post_deployment_success` |
| `async_post_call_failure_deployment_hook` | `post_deployment_failure` |
| `async_post_mcp_tool_call_hook` | `mcp_tool_result` |
| `log_success_fallback_event`, `log_failure_fallback_event`, `log_model_group_rate_limit_error` | `router_event` |
| `async_filter_listed_models` | `list_models` |
| streaming hooks, `async_pre_routing_hook`, prompt management hooks, audit log hooks | stay on `CustomLogger` for now |

Before and after for the common case:

```python
# before
class SpendLog(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        payload = kwargs["standard_logging_object"]
        await self.write(payload["model"], payload["total_tokens"], payload["response_cost"])

# after
class SpendLog(CustomLoggerV2):
    @on(Hook.attempt_ended, reads={Field.model, Field.outcome, Field.usage, Field.cost})
    async def record(self, view, ctx):
        if view.outcome.value != "success" or view.usage.logged is None:
            return
        await self.write(view.model, view.usage.logged.total_tokens, view.cost.response_cost)
```

Behaviour that changes when you migrate:

- **Mutation becomes a return value.** `data["messages"] = ...` becomes `return Patch().set(Field.messages, ...)`.
- **Blocking becomes a return value.** `raise HTTPException(...)` becomes `return Abort(...)` with `may_abort=True`.
- **Read-only handlers leave the request path.** A legacy `async_log_success_event` ran in the logging line in list order. A V2 `attempt_ended` reader runs as a background task and may finish after the response. Add `awaited=True` if something depends on it.
- **One event for success and failure.** `log_prepare` runs on success only.
- **`no-log` is opt-in per logger.** Set `honors_no_log=True` if the logger should skip those requests.
- **A logger class is either legacy or V2.** `CustomLoggerV2` does not inherit from `CustomLogger`. Split a mixed logger into two classes if part of it has to stay legacy.
- **Exceptions on request-path hooks fail the request**, except on the three failure-side hooks.

## Testing

Build a private dispatcher with `enforce=True` and call `run_hook` with the fields you want the handler to see. Enforcement makes the test fail on an undeclared read, an undeclared write or an outcome without `may_abort`. Off-path handlers have not run when `run_hook` returns, so drain the sink first.

```python
import pytest
from litellm.integrations.custom_logger_v2 import (
    Abort, CallbackRegistry, Dispatcher, Field, Hook, InProcessSink, RequestSelection, build_manifest,
)


class RecordingLog(CustomLoggerV2):
    def __init__(self):
        self.rows = []

    @on(Hook.attempt_ended, reads={Field.model, Field.outcome})
    def record(self, view, ctx):
        self.rows.append((view.model, view.outcome))


def dispatcher(*loggers):
    registry = CallbackRegistry()
    for logger in loggers:
        registry.register(build_manifest(logger))
    sink = InProcessSink()
    return Dispatcher(registry, sink=sink, enforce=True), sink


@pytest.mark.asyncio
async def test_masks_and_blocks():
    d, _ = dispatcher(MaskAndGate())
    with d.open_scope(RequestSelection()) as scope:
        ok = await scope.run_hook(Hook.pre_call, {Field.messages: [{"role": "user", "content": "mail a@b.com"}]}, None)
        assert ok.outcome is None
        assert ok.fields[Field.messages] == [{"role": "user", "content": "mail [masked]"}]
        blocked = await scope.run_hook(Hook.pre_call, {Field.messages: [{"role": "user", "content": "sk-1"}]}, None)
        assert isinstance(blocked.outcome, Abort) and blocked.outcome.status == 400


@pytest.mark.asyncio
async def test_off_path_reader_is_delivered():
    log = RecordingLog()
    d, sink = dispatcher(log)
    with d.open_scope(RequestSelection()) as scope:
        await scope.run_hook(Hook.attempt_ended, {Field.model: "gpt-4o-mini", Field.outcome: "success"}, None)
        assert log.rows == []
        await sink.drain()
        assert log.rows == [("gpt-4o-mini", "success")]
```

`build_manifest(logger)` alone is a cheap test that the declarations are valid. `RequestSelection(no_log=True)` checks that `honors_no_log` does what you expect.

For an end-to-end check through the SDK, set `litellm.callbacks = [logger]` and call `litellm.acompletion(..., mock_response="hello")`. `provider_request_sent`, `log_prepare` and `attempt_ended` all fire without a network call. Give off-path handlers a moment to run before asserting.
