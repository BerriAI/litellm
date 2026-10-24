# Response caching

This crate owns response-cache semantics shared by Rust execution in the Python SDK and in the Rust gateway: what a cache entry is, when it is fresh, how a call's cache policy is expressed, and the contract a cache service offers to core. Keep it independent of Python, HTTP serving and any proxy's configuration format

## Who decides what

Core decides when. It looks up before calling the provider and stores provider output once that output is final, a complete response after the hooks that run once per provider call and a stream once the caller has consumed it. Core never decides whether a call may use the cache, what the key is, or how an entry is laid out

The cache service decides what. It decides whether a call may read or write, derives the key, reads and writes storage, and owns the stored layout. When the service delegates to another cache implementation, that implementation owns all four, and Rust only carries values across

Values crossing that boundary are route-neutral, either a complete response or a stream's bytes. Rust never names a route's stored shape. Each route's stored layout is defined once, in the layer that owns storage

## One key for everyone who shares a cache

Every path that reads or writes the same entries derives the key from the same inputs with the same rule. Two paths that each normalize the call on their own drift apart silently, and nothing fails when they do

Key inputs are the call as the caller expressed it, including arguments passed by position. A missing input is a correctness bug, since unrelated requests then share an entry and one caller receives another caller's answer. An extra input, such as an injected default the other paths leave out, splits entries that should be shared. Treat both as key-compatibility breaks

Decide whether the cache applies before deriving the key. Disabled caching, opt-in modes, request-size limits and read/write controls must not pay for hashing the request. Derive the key once per call and use that same key for lookup, store and the key reported on a hit

Preserve existing keys and stored formats unless a change is an explicit migration decision

## Entries

A hit carries the key that found it. Report that key, never a separately derived one

An entry with no usable output is neither served nor stored. An entry that cannot be decoded, has the wrong surface or version, or does not match what the request asked for is a miss, not an error

A stream request is answered with a stream and a complete request with a complete response. Returning the other kind is a miss unless the route defines how to replay one as the other

Native envelopes are versioned per API surface. They are distinct from any format shared with another cache implementation, which must stay readable and writable by that implementation

## Streams

Store a stream only after the caller has consumed it to the end, the protocol reached its terminal event and no error event appeared. Early close, cancellation, errors and incomplete streams store nothing. A replayed stream yields the same bytes the provider sent

## Failures

A cache failure is a cache miss, never a request failure. That covers deciding eligibility, deriving the key, lookup and store. Cancellation is not a cache failure and must propagate. A failed store never repeats the provider call

Keep misses and storage failures distinguishable in return values, and leave the decision to continue with the provider to core. A freshness check may reject an entry the backend still holds. Writes record when the response was produced, not when the write lands. An asynchronous caller gets its response before a background write completes

## Responses and hooks

Response hooks sit on two sides of the cache. A hook that runs once per provider call, such as a deployment success hook, sits inside it. It runs before the store, its result is what gets cached, and it does not run again on a hit, since it may execute tools, call providers again or redact what must not be stored. A hook that runs once per request, such as per-caller policy, sits outside it and runs on hits and misses alike

A hit therefore returns what the first caller received from the inside hooks. Hits still produce success callbacks and cache-hit accounting, and callback execution stays in the host

## Services, scope and policy

Construct the cache service at the boundary, from the boundary's configuration, and inject it like the HTTP client. Reuse it across calls. Core and provider code never discover cache configuration from globals or backend factories

Configured resources and per-call policy stay separate. Backend, namespace, default expiry and entry limits belong to the service. Read and write controls, expiry and freshness overrides and caller scope belong to the call. Call policy never replaces the attached service or its scope, and every scope is explicit, shared or isolated, with no default

A service's lookup and store need nothing beyond lookup and store. Connection checks, flushing, deletion, counters and scripts live on separate interfaces. Exact and semantic matching are different lookup contracts, and supporting semantic matching needs an embedding execution contract, not a renamed trait

Implementations that must run in the caller's task, such as a Python cache, run through host operations, never as direct calls from a Rust worker

Response caching is not budget or rate-limit coordination. Counters, reservations and atomic admission stay out of the cache service, even when both use the same store

## Layering

Storage traits, codecs and backend capabilities belong to the storage crates, and stay reusable beyond LLM responses. This crate owns entries, freshness, envelopes, cache policy and the service contract. Core owns route identity, response reconstruction, stream capture and replay, and embedding partial hits. Boundaries own configuration, resource construction and caller identity

API workloads, lookup behavior and storage are separate axes. Chat Completions, Messages, Responses and embeddings are workloads. Exact and semantic are lookups. Memory, Redis, disk and object stores are storage

## Tests

Test each contract where it is owned. Storage capabilities belong in backend tests, entries and freshness here, reuse and replay in core, delegation and callbacks at the bridge, and HTTP behavior at the gateway. Any key or format shared with another implementation needs a test where each side reads what the other wrote
