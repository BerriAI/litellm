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
  prepare.rs         # provider resolution, auth headers, URL
  handler.rs         # the provider call
  client.rs          # the shared reqwest client
```

`ocr` prepares callback-visible headers and body in `prepare.rs`, settles those
authoritative roots into a native request after callbacks, and sends it through
`http_utils::buffered_post`. Reducto upload, Azure Document Intelligence polling,
and HTTP document URL conversion are declined at admission until they have an
implementation on this settled-request path. `audio_transcription` and
`realtime` remain in flight.

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

## Services, not a context

Capabilities are supplied through focused trait implementations. Route-specific
requirements traits (an `OcrServices` bundle) declare exactly what a route needs
(transport, calls, clock, ...); they are not host contexts, and core is never
passed a catch-all gateway environment. The ordinary Rust client supplies native
defaults; callers override implementations at construction.

`CallServices` opens one request-scoped sessions per call, owning per-call state
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
Tower/Axum types stop at the gateway adapter boundary
provider transformation, auth and I/O remain in core
request-scoped host state belongs to a call session
service construction dependencies do not leak into service interfaces
```

## Not the target

The current `CallLifecycleHooks` shape is not the final public service API: it
folds lifecycle sequencing into a stateless generic transformation interface,
needs `Send` futures, cannot express the full replacement and error contracts,
and does not model request-scoped retained ownership or Python caller-task
driving. It is a stepping stone, not the contract to build new routes against.

Do not introduce a dynamic `TypeId` service map, a shared gateway callback
environment reused from core or the bridge, per-callback JSON serialization, or
a full Effect layer API. Services traits plus constructors and a scoped call
owner are the minimum design; add more machinery only when concrete consumers
require it.

## Gateway migration

Existing gateway-hosted OCR, transcription and WebSocket provider execution
predates this boundary and must move here as those routes migrate. Keeping a
gateway callback as a `CallServices` implementation does not justify keeping
provider orchestration beside it in the gateway.
