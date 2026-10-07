> Hand-off copy of `~/repos/notes/rust-migration/router-poc.md` for cloud sessions that cannot see the local notes vault. Delete this file before opening a PR

# Rust router POC

Agreed 2026-10-06 in an interview session. Branch `litellm_rust_router_poc` (off `main` at `2cee61626d`). Builds on [[router-structure]] and [[router-rust-mapping]] (this is option 2 from there)

## What the POC has to prove

- The Python/Rust boundary design works: who owns state, how kwargs, errors, streams and metadata cross, nothing counted twice
- Behavior parity: the Rust backend makes the same picks, retries and fallbacks as Python and writes the same Redis keys, values and TTLs
- The crate design from [[router-rust-mapping]] (snapshot + `ArcSwap`, filter pipeline, `Operation` enum, execute loop) holds up as real code

Performance is not a goal of the POC

## Python side

`litellm.Router` becomes a thin facade over a backend interface. Two equal-behavior backends sit behind it:
- the current Python implementation, moved behind the interface unchanged
- a Rust backend exposed through `python-bridge`

Backend choice: a new `Route.ROUTER` rule in `litellm/rust_bridge/catalog.py`, toggled by `LITELLM_RUST` like the other routes, resolved once at `Router(...)` construction

Facade surface: **provisional decision, expected to change.** For now the facade exposes the full current `Router` surface, raw internals included (`cache`, `cooldown_cache`, `pattern_router`, `model_list`, index maps, ...), so no caller changes. The Rust backend's stand-ins for those internals live behind one seam, and every place that depends on this choice carries a marker pointing back here. The likely replacement is a typed protocol with real methods, with the ~164 raw-attribute call sites in the proxy migrated to it

## Rust side

- The Rust backend owns the registry: `ArcSwap<Snapshot>` with deployments, indices, aliases, patterns and settings. `upsert_deployment`, `delete_deployment` and `update_settings` go through Rust. Python attributes like `model_list` and `model_names` are read-only views projected from the snapshot
- Runtime state (cooldowns, usage counters, strategy stats) lives in memory plus Redis, with key formats, values and TTLs identical to Python so both backends can share one Redis

## Request flow

Rust drives the loop: resolve, filter, pick, retry, fallback, cooldown

Each attempt goes out as an `Invoke { deployment, operation, kwargs }` host op on the existing `coroutine` protocol. Python answers it inline in the caller's task by calling `litellm.<op>(**kwargs)`. kwargs cross as opaque Python objects; the router only reads the model, the stream flag and the metadata it writes

### Scope boundary: Rust replaces routing, never inference

Every router operation today has three layers. For embeddings:
1. `PythonRouter.aembedding`: the public entry point, which hands off to the retry and fallback loop (`async_function_with_fallbacks`)
2. `PythonRouter._aembedding`: one attempt, which picks a deployment, merges its `litellm_params` into the kwargs, counts the call and calls layer 3
3. `litellm.aembedding` (`litellm/main.py`): the inference code, meaning provider request formatting, the HTTP call, response parsing, logging and cost tracking. It does not know a router exists

The Rust backend takes over layers 1 and 2 only. Layer 3 stays exactly as it is, and replacing or reimplementing any inference code (`litellm.<op>`, provider transformations, `litellm/responses/streaming_iterator.py`, the Anthropic adapters, ...) is out of scope for this POC. Rust reaches layer 3 only through the approach this branch established: an `Invoke` host op, answered by the Python helper in `litellm/router_backends/rust_call.py`, which calls the same `litellm.<op>(**kwargs)` with the same kwargs `PythonRouter` builds. `Dispatch::Native` stays in the `Operation` design as a seam but is never used in the POC

The coroutine is driven by an async Python driver and a sync Python driver, so sync methods (`router.completion`, ...) are supported too

Python-bound features (auto, complexity, adaptive and quality routers, custom strategies, routing plugins, pre-routing hooks) are host ops that call the existing Python implementations

## Errors

When an Invoke raises, the Python driver classifies the exception into a small typed record (kind, `status_code`, `retry_after`, provider) and sends that to Rust. The original exception object rides along opaquely, so the facade re-raises it unchanged

## Streaming

The Invoke result is a Python stream. The Python driver buffers chunks with a per-format "is this content" check. If the stream fails before the first content chunk, it resumes Rust with `StreamFailedBeforeContent { error }`, and Rust returns the next attempt. Chunks never cross the boundary, so it's one crossing per attempt

## Side effects

Rust returns a typed per-call outcome (deployment, model id, attempted targets, fallback info, routing decision). The Python driver writes it into kwargs and metadata exactly where today's readers look for it. Rust records usage, latency and cooldowns itself, and the Python router callbacks aren't registered for Rust-backed instances, so nothing is counted twice

## Unsupported config

Decided at construction. The Rust backend checks the config up front. If anything unsupported is configured, `RUST_OPT_IN` declines to the Python backend for the whole instance (logged) and `RUST_REQUIRED` raises. Raw members that aren't emulated yet raise `NotImplementedError` naming the member

## POC scope (revised 2026-10-07)

The POC is the four increments below. Everything else is in the backlog and is not part of the POC

### Increment 1: the basic load balancer for chat, Responses API and Anthropic messages

- Core loop: multiple deployments per group, simple-shuffle, retries with backoff, generic fallbacks, cooldowns (memory + Redis), usage counters, non-streaming and streaming calls on both drivers
- Typed fallbacks and streams: context-window and content-policy fallbacks, retry policies, mid-stream retry before the first content chunk
- Operations: `completion`/`acompletion`, `aresponses` and `aanthropic_messages`

The Responses API and Anthropic messages reuse the same loop, but each has its own rules for a stream that fails partway (spec section 9):
- Responses API: no same-group retry. Fallback is allowed even after content was sent, by building a continuation input from the text generated so far. Opening lifecycle events (`response.created` and friends, at most 200) are held and replayed only if no fallback takes over
- Anthropic messages: the stream commits at the first `content_block_delta` or after 200 buffered frames, and pings pass through live. A recoverable error frame first gets a same-group retry loop with its own budget, then the fallback chain. Metadata lives under `litellm_metadata` instead of `metadata`

What that work touches:
- the Rust crate: the engine learns the two extra stream-failure rules (fall back after content with a continuation, and retry in the group before falling back on a recoverable frame) and the per-operation retry budget
- `litellm/router_backends/rust_call.py`: the attempt bodies that mirror `_ageneric_api_call_with_fallbacks_helper` and its `_responses_attempt` / `_anthropic_messages_attempt` variants, and per-format stream buffering and classification
- `litellm/router_backends/rust_router.py`: the `aresponses` and `aanthropic_messages` methods
- maybe `python-bridge`, if the classified record needs new fields

What it reuses without changes: the `PythonRouter` helpers for continuation input, partial usage and fallback headers (`_build_responses_continuation_input`, `_combine_responses_fallback_usage`, `_adopt_fallback_response_headers`, ...) and `litellm/router_utils`. Neither the facade (`litellm/router.py`) nor any inference code changes. If an Anthropic stream rule turns out to be tangled inside a `PythonRouter` method, it gets extracted into a shared function both backends call rather than copied

### Increment 2: changing the model list while running

The proxy builds one `Router` at startup and then syncs it with the database on a timer (`ProxyConfig._update_llm_router`): `delete_deployment(id)` for models that are gone, `upsert_deployment(deployment)` for new or edited ones, and `update_settings(**router_settings)` for settings stored in the database. Model management endpoints call `delete_deployment` directly

The Rust backend serves these by building a new `Snapshot` and swapping it in with `Registry::replace`. Calls already running keep the snapshot they started with:
- `upsert_deployment`: Python normalizes the deployment and sets its id as today, then Rust replaces the snapshot. It returns `None` when the deployment is unchanged, like Python. An edited deployment is removed and appended at the end, as Python does, because list order decides seeded picks
- `delete_deployment`: returns the removed deployment or `None`
- `update_settings`: only the keys in `RUNTIME_UPDATABLE_ROUTER_SETTINGS`, with Python's int casting and `RetryPolicy` validation
- the Python side (the discarded `PythonRouter` used for normalization and attempt kwargs, and its OpenAI client cache) is updated in the same call, so both sides stay in step

Decided 2026-10-07 (deferred, revisit only if it matters in practice): the backend is chosen once, at construction, but an upsert or settings update can add something the Rust backend declines (a deployment with `tpm`, a wildcard model, a non-shuffle `routing_strategy`, tag filtering, ...). When that happens the instance permanently falls back to `PythonRouter`, built from the updated config. Redis state carries over; in-memory state (local counters, in-memory cooldowns) is lost, and that is accepted

### Increment 3: Python-only features, called from Rust

Rust keeps running the loop and asks Python at fixed points, with one host op per point. Python answers by calling the existing implementation on the `PythonRouter` instance it already keeps. Rust only sends a host op when the config uses the feature, so a plain call still crosses once per attempt:
- before picking: the pre-routing hook (`async_pre_routing_hook`), which drives the auto, complexity, adaptive and quality routers and the session router. It can change the model, the messages and the tier's `litellm_params`
- narrowing the candidates: routing plugins (the `plugins` argument) and CustomLogger `async_filter_deployments` / `async_pre_call_check`. Rust sends the candidate ids and gets back the subset
- picking: a custom routing strategy (`set_custom_routing_strategy`). Rust sends the candidates and gets back the chosen deployment instead of shuffling
- before each cross-group fallback: the proxy's `fallback_access_check` and `fallback_budget_check`. Rust asks whether the target is allowed and skips it if not
- notifications: the CustomLogger fallback events and `router_cooldown_event_callback`. Rust tells Python after a hop or a cooldown and does not wait for an answer

Some of these features read the model list (the auto router looks up its tier groups), so they depend on increment 2 keeping the Python side in step with the snapshot

#### Approaches (sketch, 2026-10-07)

Where each feature sits in Python's flow today (spec sections 2.2 and 2.4):
- the pre-routing hook runs at the start of every pick, retries included, before candidates are resolved. It can rewrite the model, the messages and the tier's `litellm_params`, and stamps `routing_decision`, consumed tags and the session affinity TTL into the metadata bucket
- CustomLogger `async_filter_deployments` (which is also how the optional pre-call checks plug in: prompt caching, encrypted content affinity, deployment affinity, model rate limits) is filter 8 of 15, after cooldowns, and routing plugins are filter 11
- a custom routing strategy replaces `async_get_available_deployment` as a whole
- the proxy's fallback access and budget checks run before each cross-group fallback target, inside the fallback loop Rust already owns
- fallback events and `router_cooldown_event_callback` are notifications with no answer

Two facts hold whichever way we go. First, the features that learn from outcomes (the adaptive router, the affinity and rate-limit checks) do it through their own `CustomLogger`s in `litellm.callbacks`, and every attempt still calls `litellm.<op>` with logging, so they keep getting those events with no forwarding from Rust. That only works if the Python-side router stops removing them: `discard()` today drops the router's own callbacks and its `optional_callbacks` together, and only the first should go. Second, their state lives on the Python-side router, which increment 2 already keeps in step with the snapshot

Option A, one host op per hook point. New `RouterHostCall` variants (`PreRoute`, `Narrow`, `Pick`, `AllowFallback`, `Notify`) that the engine sends only when the snapshot says the feature is on. Python answers each by calling the existing method on the Python-side router, rebuilding the deployment dicts and request kwargs those methods expect from ids
- good: Rust keeps the whole pipeline and its order, every decision shows up in the differential harness, and features can move to native code one op at a time
- costs: up to four extra crossings per attempt with features on, and an adapter per hook. The pre-routing hook in particular writes into the request's metadata and can move the request to another group, which the engine then has to follow for fallback lookups (Python looks the chain up by the tier group first)

Option B, hand selection to Python for configs that use these features. The attempt's target becomes `Select { group, excluded, retry_skipped }` instead of a deployment id: Python runs `async_get_available_deployment` on the Python-side router (pre-routing hook, filters, plugins, custom strategy, all of it), with Rust's active cooldowns passed in as `_excluded_deployment_ids`, calls `litellm.<op>` in the same op, and reports back the deployment and group it used. Rust keeps retries, backoff, fallback chains, cooldowns and usage
- good: still one crossing per attempt, exact Python behavior for every selection feature at once, and the same path could later carry tags, `order`, team models, aliases and wildcards, which is most of what the proxy needs
- costs: for these configs Rust no longer makes the pick, so the POC's claim that Rust drives filter and pick only holds for plain configs. Python's "no deployments" error reads Python's own cooldown cache for its message and cooldown time, so a rejection from Python selection needs mapping onto Rust's state

Option C, hybrid (recommended). Option B for everything on the selection side, and Option A only for the hooks that sit inside the loop Rust owns: an `AllowFallback` op before each fallback target, and notifications folded into the ops Python already applies at the end of a call. That gets every selection feature to parity cheaply, keeps the fallback checks where the proxy expects them, and leaves room to port simple filters (tags, `order`, rate limits) to native Rust later, one at a time

Option D, keep declining and port features natively. Pure-config filters could move to Rust as native code, but the model-driven routers (auto, complexity, adaptive, quality) and custom strategies or plugins are user or ML code that has to stay in Python, so they need A or B eventually anyway

Before prototyping, the choice that matters is whether giving up Rust-made picks for feature configs (B and C) is acceptable for what the POC has to prove, or whether Rust must make every pick (A)

Decision (2026-10-07): Option C. Rust drives the loop, and the strategy routers stay in Python and are reached through `Select`. What the strategies depend on, checked before prototyping:
- the contract is `PreRoutingStrategy.async_pre_routing_hook(model, request_kwargs, messages, input)` returning `PreRoutingHookResponse` (model, messages, routing_decision, session_affinity_ttl_seconds, litellm_params), but every strategy also writes into `request_kwargs` metadata, and the router-level wrapper stamps more keys that later filters read (consumed tags for the tag filter, the affinity TTL for the deployment affinity check, forwarded marker params for the attempt kwargs)
- they call back into the router: the complexity router reads and writes session pins in `router.cache`, reads `get_model_list`, `get_router_model_info` and `async_get_healthy_deployments` for its modality, health and prompt-cache gates, and runs its LLM classifier through `router.acompletion` / `aresponses`; the auto router embeds through `router.aembedding`
- context compaction and routing compression are armed through contextvars in the hook and read by the inference call, so the hook and `litellm.<op>` have to run in the same Python task, which `Select` does and per-hook ops would not
- under `Select`, the health and prompt-cache gates still read the Python-side cooldown cache, which is empty, so the Python-side router needs Rust's cooldowns as its cooldown view, not only as `_excluded_deployment_ids` for the routed group

### Increment 4: what the proxy needs to switch over

Today the proxy always gets `PythonRouter`, because it builds `Router(...)` with arguments the Rust backend declines (`router_general_settings`, `search_tools`, `ignore_invalid_deployments`, `fallback_access_check`, `fallback_budget_check`, `auto_router_capability_limit`). Switching over needs three things:
- accept the proxy's constructor arguments. The two fallback checks come from increment 3. The others are settings the Rust backend has to honor or can safely pass to the Python side
- serve the proxy's reads. 116 proxy files use about 109 distinct router members, about 386 times. Most uses read the model list (`get_model_list`, `get_deployment`, `model_list`, `get_model_names`, `get_model_group_info`, `get_model_access_groups`, `get_deployment_credentials_with_provider`, ...), and those become views of the snapshot. The rest reach into live internals (`cache`, `pattern_router`, `adaptive_routers`, `deployment_latency_map`, `health_state_cache`, `model_name_to_deployment_indices`, ...). Each of those gets a stand-in, or becomes a typed method on the facade, which is where the provisional full-surface decision gets settled
- serve the request path. `route_llm_request` calls `getattr(llm_router, route_type)(**data)` for every operation, so one instance serves chat, embeddings, files, batches, passthrough and the rest

Decided 2026-10-07 (deferred, same rule): a Rust-backed instance that is asked for an operation it does not serve yet permanently falls back to `PythonRouter` for the whole instance, losing in-memory state. So the proxy can switch over now and runs on Rust until the first backlogged operation arrives

## Backlog (not part of the POC)

- the one-shot operations: embeddings, text completion, image generation and editing, transcription, speech, rerank, moderation
- operations on stored objects: files, batches, fine-tuning jobs, assistants, vector stores, realtime sessions, passthrough routes
- fan-out operations: `abatch_completion` and the fastest-response variants
- the other routing strategies (least busy, latency, usage, cost) and per-deployment limits (`tpm`, `rpm`, `max_parallel_requests`, `order`)
- deployment filters: aliases, wildcard patterns, access groups, tags, team models, routing groups, pre-call checks, health checks
- request options that raise `NotImplementedError` today: `priority`, `specific_deployment`, request-level `model_group_retry_policy`, `include_fallback_errors`, `_router_weights`, client-side credentials

Out of scope entirely: Rust calling providers itself (see Scope boundary)

## Parity testing

- Existing behavior tests that go through `Router(...)` get parametrized over both backends. Rust-side failures are the to-do list
- A differential harness: same config, scripted fake Invoke outcomes (errors, latencies), seeded randomness. Assert both backends make the same decisions and write the same Redis keys, values and TTLs

## Order of work

1. Facade refactor: move today's `router.py` behind the backend interface with no behavior change. Its own commit
2. `Route.ROUTER` catalog rule and backend selection at construction
3. Rust crate: snapshot/registry, `Operation`, execute loop, host-op types
4. `python-bridge` export + async/sync Python drivers for the router coroutine
5. Differential harness and parametrized tests, then fill in the first increment until they pass

## Implementation notes

- Each Invoke awaits a small Python helper, not `litellm.<op>` directly. The helper catches the exception and returns a typed outcome (classified record plus the original exception). For streams it also buffers until the first content chunk before returning, so classification and buffering both stay in Python and each attempt is still one host op
- `host-python::run_call` already gives both drivers: async goes through `lifecycle._settle` in the caller's task, sync answers host ops inline. No new driver. The router does not go through `run_public_call`, because the `litellm.<op>` behind each Invoke already runs legacy Logging and preflight
- Deployment normalization and `generate_model_id` stay in Python (they depend on Python `str()` / `json.dumps` output). Rust receives normalized deployments with ids already set and owns the snapshot built from them
- Redis parity risk: Python's sync `RedisCache.set_cache` writes `str(value)` (a Python repr), async paths write `json.dumps`, counters use `INCRBYFLOAT` with a TTL set only when absent, and keys pass through `check_and_fix_namespace`. Parity means matching whichever Python path writes each key. Rust state sits behind a store trait: in-memory first, then Redis
- Keys in scope for the first increment: `deployment:{id}:cooldown` (CooldownCacheValue: exception_received masked to 50 chars, status_code as str, timestamp, cooldown_time; TTL = cooldown time), `deployment:{id}:allowed_fails[:{suffix}]` (fleet-wide counter, TTL = cooldown time), `{id}:successes` / `{id}:fails` (local only, TTL 60)

## Log

- 2026-10-06: design agreed, branch created
- 2026-10-07: facade verified, steps 2, 3 and 4 landed, step 5 started (parametrized suites, Redis parity)
- 2026-10-07: scope revised: Rust replaces routing only, never inference. The POC is increments 1 to 4 (basic load balancer for chat, Responses API and Anthropic messages, runtime model list changes, Python-only features as host ops, proxy switchover); the rest is backlog

## Checkpoint 2026-10-07

Done:
- Step 1 verification: done. tests/unit, tests/router_unit_tests and tests/local_testing/test_router_fallbacks.py were run on the branch and on its base `cb76270b`; the branch-only failures were the facade identity issue (fixed in `71120e1f`) and two router_unit_tests that patched the facade instead of the backend (fixed). `make check` found an isort issue and an unformatted `python_router.py`, LIT010/LIT008 and four ANN204 in `router.py`, and a moved LIT013 suppression, all fixed. In this container `make check`'s dependency sync and the basedpyright gate cannot provision (a private git dependency returns 403) and the dashboard has no node_modules, so those steps were run per file instead
- Step 2: `Route.ROUTER` (`RUST_OPT_IN`) in `litellm/rust_bridge/catalog.py`. `litellm/router_backends/selection.py::select_backend` runs in `Router.__init__`; `rust_support.py` decides what the Rust backend can serve (declines to `PythonRouter` under opt-in, raises under `RUST_REQUIRED`).
- Step 3: the Rust crate. `engine::Engine::route` runs fallbacks -> retries -> attempt against a `host::RouterHost` (`invoke`, `sleep`). Modules: `snapshot` (registry), `settings`, `failure` (`Classified` record, `Rejection` for router-raised errors), `selection` (resolve, cooldown filter, retry skip, `simple_shuffle`), `retry` (`should_retry_this_error`, backoff), `fallback` (chain lookup), `cooldown` (failure callbacks), `store` (memory + Redis with Python's encodings), `pyrepr` (Python `str()` of dicts), `random` (CPython MT19937), `operation` (`Operation`, `Dispatch`, all `Invoke` today)

Design notes made while implementing:
- Python-bound facts cross inside `Classified`: exception classes in MRO order, `str(e)`, status, stamped `num_retries`, the two Retry-After readings (sleep and cooldown use different header precedence), guardrail/exempt flags, `callbacks_ran` (false for errors the attempt raises after litellm returned, e.g. blocked output) and `exact_litellm_type`
- Metadata and exception writes the router makes (log_retry breadcrumbs, hop bucket copies, exhausted-retry stamps, the fallback debug text) are `host::Op`s the host applies in order before the next attempt or on the final result. Buckets are numbered: 0 is the request's own, each fallback hop opens a copy of its parent's, matching `run_async_fallback`
- Router-raised errors (`RouterRateLimitError`, no-healthy `BadRequestError`) are `Raised::Router { id, rejection }`; the host materializes each id once
- Sleeps are a host op so tests can observe them and the sync driver can block
- Python's shared-logging behavior is kept: with `litellm_logging_obj` passed, only the first failure runs the failure callbacks and fallback-hop failures cool down through `_trigger_cooldown_for_failed_deployment`

- Step 4: `python-bridge` exports `Router` (`src/routes/router/`): `Router(deployments, settings, providers, redis_url, seed)` holds the `Engine`; `route(call, driver, asynchronous)` runs one call through `host-python::run_call` with `HookChain::new()`. `Invoke` and `Sleep` are host ops on `RouterHostCall`; async answers await the driver's coroutine inline, sync answers call `invoke_sync`/`sleep_sync`. `map_error` asks the driver for the exception to raise (ops applied, rejections materialized once per id), `encode_response` asks it for the final response (ops applied, retry/fallback headers added)
- Python: `RustRouter` (`litellm/router_backends/rust_router.py`) builds a `PythonRouter` from the same arguments minus Redis and `discard()`s it, so normalization, `generate_model_id` and attempt kwargs building stay Python and no router callbacks are registered. It projects deployments and resolved settings to the native router. `acompletion`/`completion` are served, including async streaming; sync streaming, `priority`, `specific_deployment`, request-level `model_group_retry_policy`, `include_fallback_errors`, `_router_weights`, and client-side credentials raise `NotImplementedError`. Other members raise `NotImplementedError` naming the member, except the read views `model_list`, `model_names`, `get_model_names`
- `litellm/router_backends/rust_call.py::RoutedCall` is the per-call driver: it mirrors `_acompletion`/`_completion`'s body after selection (same `PythonRouter` helpers through a typed `AttemptRouter` protocol), classifies exceptions, and applies the router's ops to numbered metadata buckets
- Tests: `tests/unit/router_backends/test_rust_call.py` (classification, buckets, rejections, debug text), `tests/test_litellm_rust/router/test_rust_router.py` (native; same config and seed through both backends: identical pick sequences, headers, error text, cooldown-then-reject sequence, sync path, facade selection)

- Step 5 (in progress): `litellm/router_backends/selection.py::pinned_backend` fixes the backend decision; `tests/unit/router_backends/backends.py::router_backend` runs an opted-in module on both backends (Rust pinned to `RUST_REQUIRED`), with known Rust-side failures as strict xfails in `RUST_GAPS`. Opted in so far: `tests/unit/test_router_per_deployment_num_retries.py`, `tests/unit/test_router_exception_redaction.py`. `tests/test_litellm_rust/router/test_redis_parity.py` runs both backends against a live `redis-server` and compares every key, raw value (timestamp masked) and TTL: cooldowns, `allowed_fails` counters and failure RPM keys match byte for byte, and a Rust router honors cooldowns a Python router wrote

Streaming (chat, both drivers): `RoutedCall` reads the stream until the first chunk with content (Python's `_stream_chunks_have_generated_content`) and returns a `ReplayStream` (a `FallbackAwareStreamWrapper`, sync and async) that replays the buffered chunks. A `MidStreamFallbackError` before content is classified `stream_failure: before_content` and goes to that hop's own fallback chain without in-group retries; any other pre-content stream error is `terminal`. Either way the result settles the call: Python's outer retry and fallback layers had already returned the stream, so the engine's `Stop::Settled` and `Success::settled` keep them from retrying, trying the next target or rewriting the outcome. A stream error reaching the caller is unwrapped to its `original_exception`, as Python's wrappers do. The sync driver uses the async rule on purpose, because Python's sync wrapper reruns the same group and can recurse until `RecursionError` (spec section 11 item 26). Partial usage is not added to the fallback's, matching Python, whose check never passes (item 27). Like Python, the sync path no longer touches `total_calls`/`success_calls`/`fail_calls` (item 28)

Responses API (`RustRouter.aresponses`): `RoutedCall._ageneric` mirrors `_ageneric_api_call_with_fallbacks_helper` after selection (metadata in `litellm_metadata`, failures counted by model group, the Anthropic refusal check). `rust_streams.responses_until_output` holds the opening lifecycle events until the first output, as Python's wrapper does; a mid-stream fallback error before then goes to the hop's chain with a content-policy trigger unwrapped. After output the stream still falls back: the `ResponsesReplay` (Python's `FallbackResponsesStreamWrapper`, moved to `litellm/router_utils/responses_fallback_stream.py` so both backends use it) calls `Engine::resume` through a second `route` with the hop's group, depth, original group and attempted targets, and the input continued from the generated text. Partial usage is added to the fallback's terminal events with Python's helpers (not covered by a parity test yet). Headers after a fallback that falls back again differ on purpose (spec section 11 item 29). Anthropic messages (`RustRouter.aanthropic_messages`): `rust_streams.anthropic_until_committed` holds lifecycle frames until the first content delta or 200 frames, and fails the attempt on a recoverable error frame or a safeguard refusal, deciding with `PythonRouter`'s own gates (`_anthropic_messages_stream_can_retry`, `_anthropic_messages_stream_can_fall_back`, `_anthropic_messages_recoverable_frame_error`, `_refusal_fallback_available`) over the attempt's request. When nothing could take over it commits at once and forwards every frame. Pings, which Python forwards live while frames are held, replay ahead of the held frames, so the caller sees the same order but only once the attempt commits. A failure before the commit goes to the engine's `retry_stream` (`_aanthropic_messages_retry_same_group`): the committed, policy or plain budget, `should_retry_this_error` unless a policy governs, backoff, then the hop's fallback chain; a retry whose stream fails again carries on the same count. Retry decisions and cooldowns read the provider error inside the envelope (`Classified.original`), the fallback chain reads the envelope with content-policy and context-window errors unwrapped. A retry failing before its stream opens comes back in Python's envelope (`stream_retry` on the attempt), with `retries_pre_stream` saying whether the loop goes on

Increment 2 (runtime model list changes): `RustRouter.upsert_deployment`, `add_deployment`, `delete_deployment` and `update_settings` apply the change to the `PythonRouter` that normalizes deployments (so no-op detection, ids, the remove-then-append order and setting validation are Python's own), then swap the native snapshot with `Router.replace` (`Registry::replace`); a running call keeps the snapshot it started with. Before applying, the prospective config (the original arguments, every runtime-updatable setting applied so far and the new deployment) goes through `unsupported_reason`. If the Rust backend cannot serve it, the instance hands over to a `PythonRouter` built from the original arguments, the current model list and the accumulated settings, through the hook the facade installs (`on_fall_back`), and the change is applied there; in-memory state is lost, Redis state carries over. Under `RUST_REQUIRED`, or with no facade, it raises `RustRouterUnsupportedError` instead. `update_settings` ignores keys outside `RUNTIME_UPDATABLE_ROUTER_SETTINGS`, as Python does, and `max_retries` (which only the Python side reads) counts as supported. `retry_policy` and `get_settings()` read through to the Python side, and a name that is not a Router member raises `AttributeError` rather than `NotImplementedError`, so `hasattr` behaves. Tests: `tests/test_litellm_rust/router/test_runtime_changes.py` (added, edited and deleted deployments and runtime fallbacks route the same on both backends; the hand-over and the required-mode error), an engine test for the running-call snapshot, and `tests/unit/test_router_retry_policy_update.py` opted in (its 11 Rust gaps are optional pre-call checks)

Suite survey (2026-10-07, every test run on both backends with Rust pinned to `RUST_REQUIRED`): no Rust-only failure was a wrong routing decision. In `tests/local_testing/test_router_{fallbacks,retries,timeout}.py`, 23 tests pass on Python and fail on Rust: 11 use configs the backend declines (rpm/tpm, wildcards, routing strategies, dict fallback entries, `allowed_fails_policy`) and 12 patch a `PythonRouter` internal (`should_retry_this_error`, `_time_to_sleep_before_retry`, `make_call`, `log_retry`, `_get_stream_timeout`); those modules also hold live tests that need provider keys, so they stay Python-only. In `tests/unit/test_router/test_router.py` 777 of 992 fail on Rust: 248 declined configs, and the rest call `PythonRouter` internals (`async_function_with_fallbacks_common_utils`, the Anthropic stream helpers, ...) or read views increment 4 has to provide (`get_model_group_info`, `get_deployment_by_model_group_name`, `get_deployment_model_info`, `get_model_list`, `get_model_access_groups`, `get_deployment_credentials_with_provider`, ...). Suites that test Python helpers directly (`router_utils/test_fallback_event_handlers.py`, `test_cooldown_handlers.py`, `litellm_core_utils/test_fallback_generalizations.py`) never call the router's own methods, so running them twice adds nothing. Opted in: `tests/unit/test_router_streaming_fallback_metadata.py`, and `tests/unit/test_router_retry_policy_update.py` with increment 2. The parity suites under `tests/test_litellm_rust/router` carry the behavior coverage instead

Next: prototype increment 3 with Option C (`Select` target first, then `AllowFallback`), then increment 4

Known gaps to close or decline: CustomLogger fallback-event hooks and `router_cooldown_event_callback` (increment 3), the 200-key `InMemoryCache` eviction, fallback keys containing `/` (provider-prefixed matching needs the cost map)
