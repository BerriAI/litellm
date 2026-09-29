# Inference orchestration ownership

`litellm-core` owns call entrypoints, call envelopes, provider dispatch, credential fallback, preparation sequencing, and coordination of transport, hooks, caching, and streams

## Crate boundaries

Shared API payloads belong in `llms-types::formats`, and shared provider wire contracts belong in `llms-types::providers`. Reuse those contracts inside execution envelopes rather than defining another payload schema here

Adapter contracts and shared transformation machinery belong in `llms/src/base_llm/<format>`. Provider policy belongs in `llms/src/<provider>/<format>`. Core selects concrete adapters and invokes their contracts. Authentication policy, header selection, capability checks, payload rewriting, and response normalization stay with the adapters. Calling one provider's helper for every provider is still a policy dependency

Pure provider resolution, prompt normalization, argument processing, and settings lookup/merge belong in `core-utils`. Route-neutral clients, TLS, proxies, media fetching, header application, and transport errors belong in `http`. Provider code must never import from core

Hosts supply execution dependencies and implement host operations, callback execution, and integration-specific observation. Keep inference handlers in core or llms. Keep HTTP serving, request decoding at the external boundary, config file reading, rollout state, databases, and Python bindings in their owning crates

Use shared host call machinery for stream delivery, backpressure, cancellation, and detachment rather than implementing them in each route. Route code supplies execution dependencies and route-specific capabilities. WebSocket session orchestration remains separate from single HTTP-call orchestration

Python paths identify counterparts, not Rust ownership. Import items from their canonical crate. Never re-export another crate's items or give an item a second public path. A private submodule may expose its own items at its module root

## Error ownership

Declare a failure once in the lowest crate that raises it. Core wraps that error with its source intact or maps it once at the route boundary. Route errors must not redeclare lower-layer failures

Scope follows the concept, not the first caller. Provider-local errors cannot be imported by other providers or shared adapter machinery. Framing failures belong in `framer`, and transformation failures belong in `llms`. Follow the workspace error-definition rules

## Caching and accounting

Core owns effective request identity, typed response reconstruction, and stream capture/replay. `cache-response` owns cache policy, namespacing, scope encoding, versioned envelopes, and freshness. Hosts choose cache scope, including identity isolation for the gateway. Attaching a cache must preserve the route's execution contract

Cache identity must follow provider preparation, credential resolution, and request interception. Keys describe the effective URL, authenticated headers, and rewritten body. Signed requests bypass caching until signing identity has a stable cache representation

Core delivers execution facts for provider and cached results through the awaited host operation before public response processing or stream opening. Usage belongs in the response or delivered stream. Passive observation must not replace accounting delivery

Pricing, budget charging, rate-limit counters, and reservations belong to accounting owners, never core or response-cache services. Python owns accounting on the legacy callback path. Native gateway accounting belongs to gateway dependencies independently of `host-python`

## Test ownership

Test route dispatch, credential precedence, sequencing, error propagation, cancellation, and cache/stream execution in core. Test payload serialization in `llms-types` and provider transformations in `llms`. Follow the workspace test rules and exercise behavior through injected dependencies rather than inspecting source structure
