# The life of a request

This page follows one `await router.acompletion(model="gpt-4o", messages=...)` from the facade, through the Rust core, to the provider and back. The other endpoint methods (`aembedding`, `aresponses`, `aanthropic_messages` and the rest) take the same path with a different operation name

## From the facade to the core

Every endpoint method is a thin wrapper over one core call, `call(op, request)`. The operation names the endpoint (`acompletion`, `aembedding`, `aanthropic_messages`, ...) and the request is the caller's keyword payload. In the core they become an `Op` and a `RouterRequest`, which carries the requested name and the few request fields routing reads, such as the caller's team id, `specific_deployment`, a per-request `num_retries`, `disable_fallbacks` and per-team weights. The rest of the payload travels with the request untouched, for the attempt to use

```mermaid
flowchart TB
    A["Router.acompletion<br/>facade"] --> C["RouterCore.call('acompletion', request)"]
    C --> D["RustCore driver<br/>starts a RouterCall"]
    D --> F["call loop: fallbacks<br/>moves to another model group when retries are spent"]
    F --> R["call loop: retries<br/>loops in the same model group, with backoff"]
    R --> T["attempt<br/>select, resolve executor, run, record the outcome"]
    T --> X{"executor"}
    X -->|"native route"| N["in-process inference route"]
    X -->|"anything else"| H["Host::invoke<br/>litellm.acompletion with the deployment's params"]
    N & H --> P["provider HTTP call"]
```

Fallbacks wrap retries, and retries wrap a single attempt. Each attempt selects a deployment and calls it, so a retry can land on a different deployment. The loop is in `litellm-rust/crates/router-core/src/call.rs`

## One attempt, in order

```mermaid
sequenceDiagram
    autonumber
    participant F as Router facade
    participant K as Rust core
    participant H as PythonHost
    participant S as RouterCache
    participant P as Provider
    F->>K: call(op, request)
    K->>K: snapshot of registry, settings, dynamic config
    K->>K: resolve the name, team filter
    K->>S: read cooldowns for the candidates
    S-->>K: cooled-down ids
    K->>K: drop cooled-down and blocked deployments
    K->>H: select(Plugins): pre-call checks, filters, sub-routers
    H-->>K: remaining candidates
    K->>K: order filter, skip ids that failed this request
    K->>K: strategy picks one (simple-shuffle natively)
    K->>H: invoke(attempt)
    H->>P: litellm.acompletion(**deployment params, **request)
    P-->>H: response or error
    H-->>K: output with total tokens, or a typed failure
    K->>S: usage counters, or fail counts and maybe a cooldown
    K-->>F: response with routing headers
```

Some details worth knowing

The snapshot is taken once, at the start of the call. A concurrent `apply()` swaps in a new registry for later calls, and this call keeps the one it started with

The core turns an attempt's error into a typed `AttemptError` with a failure kind (rate limit, context window, content policy, auth, timeout, ...), the status code and any `Retry-After`. Retry, fallback and cooldown rules read those facts, never the exception text

The provider SDK and HTTP client are told not to retry (`max_retries=0`). The router's loop is the only retry loop, which is what lets each retry pick a different deployment

`litellm.acompletion` is the same function SDK users call. Under `PythonHost` the router does not talk to providers itself. It chooses the deployment, and the provider transformation layer does the rest. Under the gateway's `NativeHost`, or for a native route, the attempt runs in Rust

The response carries the chosen deployment's id in `response._hidden_params["model_id"]`, and the proxy reports it in the `x-litellm-model-id` header together with the model group, the retries and the fallbacks the call used

## How results flow back into routing

The core records each outcome itself, right after the attempt, so nothing depends on a logging callback firing

```mermaid
flowchart LR
    att["attempt outcome"] -->|"success"| ok["Health::record_success<br/>usage this minute, success count"]
    att -->|"failure"| bad["Health::record_failure<br/>fail count, allowed fails, cooldown decision"]
    ok & bad --> cache[("RouterCache")]
    att --> ev["RouterEvent to Host::observe<br/>Selected, Succeeded, AttemptFailed,<br/>CooldownSet, Retry, Fallback, RetriesExhausted"]
    ev --> cb["logs, callbacks, alerts, metrics"]
    cache --> next["next selection"]
```

Strategies that keep their own statistics, such as latency or in-flight counts, still learn from the logging callbacks of each call. Until a strategy is ported to Rust it runs in Python through the select bridge and keeps its counters there. `router.observe(request_scoped=True)` binds the router's observers to each request's logging object instead of registering them globally

## The sync path

`router.completion()` and the other sync methods run the same core call to completion on the calling thread. The proxy uses the async methods for every request
