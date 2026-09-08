# core — the runtime and the single core route

`litellm-core` is the LiteLLM SDK in Rust: one language-neutral route per
endpoint (`messages::messages`, `ocr`, `audio_transcription`, `realtime`,
`responses`). Every host (the Rust SDK, the PyO3 bridge, the gateway) calls the
same route program; they differ only in the service implementations they
supply. Core owns route sequencing, provider policy, transformation,
authentication, protocol operations, provider I/O and callback phase placement.
Hosts provide capabilities; they never orchestrate the route.

## What core owns

Core owns all decisions that must stay identical across hosts. A host adapter
that performs any of the following around a core entrypoint is a bug:

- private admission, and the point after which execution cannot be replayed;
- deployment and provider interpretation supplied in the typed route inputs;
- callback and guardrail phase placement and result interpretation;
- provider transformation, credential acquisition, final authorization and
  encoding;
- explicit multi-operation protocols such as OCR upload, submission and
  polling;
- provider I/O, response normalization and terminal success or failure;
- transfer of stream, connection and deferred-completion ownership.

A host adapter performs only work specific to its public boundary: construct
the runtime composition, convert public inputs into `request` + `options` +
`CallContext`, invoke exactly one core route entrypoint, and map the outcome to
the host's response or error contract. Never a second provider pipeline.

## Route shape

A top-level call is a module under `src/<route>/`, shaped like `messages`:

```
core/src/messages/
  mod.rs             # pub async fn messages(..) -> Result<.., Error>  (+ _stream for SSE)
  types.rs           # request/response types
  transformation.rs  # the provider template trait
  request.rs         # provider resolution, auth headers, URL and body construction
  handler.rs         # the provider call
```

`ocr` prepares callback-visible headers and body, settles those authoritative
roots into a native request after callbacks, and sends it through the shared
`http_utils::http_request` function. Reducto upload, Azure Document Intelligence polling,
and HTTP document URL conversion are declined at admission until they have an
implementation on this settled-request path. Audio transcription provider I/O,
realtime WebSocket dialing and splicing, and Responses WebSocket dialing and
splicing are also core-owned.

The invariant is one function body owns the route lifecycle. The conceptual
shape is:

```rust
pub async fn ocr<S>(
    services: &S,
    request: OcrRequest,
    options: OcrOptions,
    context: CallContext,
) -> NativeResult<OcrResponse>
where
    S: OcrServices,
```

The exact spelling may be a method on `LiteLlm<S>`. Public adapters may wrap
that function but can never reimplement admission, callbacks, provider
preparation or transport around it.

## Provider and route ownership

`providers/dispatch.rs` selects a typed adapter for each supported provider and
route pair, including model-specific OCR variants. Routes own their contracts
and lifecycle sequencing; provider adapters own admission policy, URLs,
transformation and authorization. Keep provider selection out of handlers

Provider modules share credential and protocol helpers across their route
adapters. Routes supply the exact settled bytes to the adapter's authorization
operation at the existing lifecycle phase. Shared helpers must never invoke
another public route or repeat its callbacks

WebSocket execution carries its selected adapter through dialing and event
transformation. The existing Responses WebSocket entrypoint defaults to OpenAI;
realtime resolves its existing optional provider prefix through dispatch

## Services, not a context

Capabilities are supplied through focused trait implementations. Route-specific
requirements traits (an `OcrServices` bundle) declare exactly what a route needs
(calls, clock, ...); they are not host contexts, and core is never
passed a catch-all gateway environment. The ordinary Rust client supplies native
defaults; callers override implementations at construction.

`CallServices` opens one request-scoped session per call, owning per-call state
(timing, logging state, retained host objects, deferred completion, correlation
state). No-op call services are the default, and their presence must not move
provider behavior into a host or force callback payload materialization on an
unaffected fast path.

Your `Callback`/`CallServices` operations must stay distinct where their
contracts differ (argument identity, replacement adoption, exception policy,
scheduling, direct vs awaited execution). A universal `emit(Event, Json)` is
not enough. core still decides which operation runs next and how its result
affects execution; the adapter only dispatches it (Python callback, native
logger call, or nothing).

## Language neutrality

Core types and service contracts must not contain `Py<PyAny>`, `Py<PyDict>`,
Axum requests, gateway state or Python logging objects. Host implementations may
retain those values privately; core sees only the trait operations and the typed
values they return. `CallContext`, `*Request`, `*Options`, `*Response` and core
errors are language-neutral.

## Dependency rules

```
core must not depend on PyO3, Axum or gateway integration types
litellm-auth owns shared credential values, capability contracts and future static helpers
cloud-auth crates own reusable native credential and signing mechanisms
core owns provider precedence, header policy and authorization timing
Tower/Axum types stop at the gateway adapter boundary
provider transformation, auth policy and I/O remain in core
request-scoped host state belongs to a call session
service construction dependencies do not leak into service interfaces
```

`litellm-auth` has only standard-library production dependencies and does not
depend on core or cloud crates. Add it as a dependency only where consumed.
Cloud crates retain their own credential types and errors. `AuthorizationFuture`
lives in `core::providers` because its result uses the core error type. The former
`core::auth` module is removed without compatibility re-exports

## Not the target

Do not introduce a dynamic `TypeId` service map, a shared gateway callback
environment reused from core or the bridge, per-callback JSON serialization, or
a full Effect layer API. Services traits plus constructors and a scoped call
owner are the minimum design; add more machinery only when concrete consumers
require it.

## Streaming ownership

Core owns provider sessions through completion. Streaming HTTP calls transfer a
`StreamingCall` whose completion registration keeps terminal dispatch alive
until the host finishes or drops the stream. Realtime and Responses WebSocket
entrypoints retain the provider connection while they splice events, then emit
exactly one terminal record after the committed session completes or fails.

Realtime pool warmup is not a user call and must emit zero terminal records on
both success and failure. A warmed connection transfers into `realtime`; only
that serving session owns completion and terminal dispatch.

## Shared HTTP execution

`http_utils` owns cached reqwest clients and the traced HTTP send function for
chat completions, messages, transcription and OCR. Client profiles preserve
connection, timeout, redirect and decompression settings. Provider adapters own
URLs, credentials, authorization, payloads and response transformations. Routes
own lifecycle sequencing and response body or stream consumption

HTTP execution receives settled bytes and never reserializes or reauthorizes
them. OCR uses reqwest requests and responses directly; `OcrTransport`,
`OcrTransportRequest`, `OcrTransportResponse` and the public `buffered_post` module
have been removed without compatibility aliases. `OcrServices` now requires
only `Clock` and `TerminalDispatcher`; its route signature is unchanged
