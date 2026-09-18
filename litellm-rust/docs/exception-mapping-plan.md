# Exception mapping: plan to align the Rust path with the five-stage design

Scope: branch `litellm_ocr_rust_parity_gaps`. The types and the trait are generic, but only the OCR route is moved in this work. Chat, messages, responses and audio transcription are a later change (section 4.5) and nothing here depends on them. Nothing in this plan has been built or run. The target design is the five-stage chain (capture, normalize, classify, construct, finalize) where each stage has one owner and the pyo3 layer owns none of them

## 1. What this branch changed

- `fd18a6c9d70`, `75cbb689b8a`: tests only
  - Failing contract tests for the OCR Rust path gaps
  - A differential parity sweep in `tests/test_litellm_rust/ocr/test_parity.py` with `support/parity.py` (`Outcome`, `Divergence`, `assert_parity`)
- `9a0c45f6e3e`: `core/src/litellm_core_utils/secret_redaction.rs`, a port of the Python redaction patterns
- `0eb60785854`: the mapping itself
  - `core/src/litellm_core_utils/exception_mapping_utils.rs` (1021 lines, one file): `ProviderException`, `ExceptionContext`, `PublicFailure`, `exception_type`, with `map_openai_exception`, `map_vertex_exception`, `map_cohere_exception` and `map_exception_by_status` as hand-written functions, plus 13 rstest groups
  - `python-bridge/src/failures.rs`: `provider_exception` (parses a `RustUpstreamError` back into a `ProviderException`), `exception_context` (reads `litellm.*` globals and kwargs), `public_exception` (serde dict to `litellm.rust_bridge.failures.public_exception`), `map_upstream_failure`
  - `python-bridge/src/routes/ocr/host.rs`: `map_failure` tries `map_upstream_failure` first, then falls back to the Python `route_host.map_failure`
  - `litellm/rust_bridge/failures.py`: Pydantic `PublicFailure` with `extra="forbid"` and the constructor `public_exception`. The old `map_failure` that calls `litellm.exception_type` is still there
  - `litellm/rust_bridge/dispatch.py`: `finalize_failure`, `_finalized`, `_afinalized` wrap the native call in `run` and `arun`

## 2. Where the branch stands against each stage

| Stage | State on the branch | Gap |
|---|---|---|
| 1. Capture | `ocr::Error` carries `Provider{status, body, headers}`, `Transport`, validation, auth and file variants | None in core. The bridge turns it into a `PyErr` too early (`routes/ocr/errors.rs::to_pyerr`) |
| 2. Normalize | `OcrConfigKind::get_error_class(message, status, headers)` exists but returns `ocr::Error`. The real normalization happens in the bridge: `provider_exception` rebuilds `ProviderException` from `RustUpstreamError.args` and a `headers` attribute set with `setattr` | Wrong owner and a lossy round trip through a Python exception |
| 3. Classify | `exception_type(&ExceptionContext, &ProviderException)` in core | Only `Provider` and `Transport::Http` errors reach it. Everything else (`ValueError` with a patched `status_code`, `FileNotFoundError`, `OSError`, `RuntimeError`) goes to the Python `litellm.exception_type`, which is Python-path-only code. `RequestFormat` is classified in Python through the `ocr_request_format_error` attribute. Provider dispatch is on strings (`"mistral" \| "azure_ai"`, `"vertex_ai"`, `"cohere"`) |
| 4. Construct | `failures.public_exception` | Matches the design. The payload shape does not: flat `response` and `request` fields that both have `Omitted`, an `upstream_content` override applied with `setattr` after construction, `stdout` carried as a string, no `provider_specific_fields` |
| 5. Finalize | `dispatch.finalize_failure`, tested for sync and async | Done. Only `str(error)` coverage is missing |
| Driver | `machine_failed` keeps the interrupted host `PyErr` apart from the native error, then calls `H::native_error` and `map_failure(&PyErr)` | `map_failure(py, &error).unwrap_or(error)` at `driver.rs:390` swallows a mapper failure |

Two things the design text does not say but the code shows:

- Only OCR runs on the `RouteHost` driver. Chat, messages, responses and audio transcription use `run_sync` / `run_async` with `*_error_to_pyerr`. Step 5 of the design is therefore larger than moving a call site. See 4.5
- `runtime._raise_upstream` and the `map_failure` functions in `chat_completions/route_host.py`, `messages/route_host.py` and `responses/route_host.py` are dead stand-ins
  - No Rust code calls those three `map_failure` functions. Only `routes/ocr/host.rs` imports a `route_host.map_failure`
  - `catalog.RULES` enables only OCR and bedrock transcription. Transcription errors go through `core_error_to_pyerr` and never raise `RustUpstreamError`. OCR's `RustUpstreamError` is mapped inside the driver. The only way one reaches `_raise_upstream` today is the `unwrap_or(error)` swallow in the driver
- OCR invents an HTTP response for timeouts: `ocr/client.rs::transport_error` returns `Transport::Http{status: 408, body: "OCR request timed out"}` and `ocr/document.rs:207` does the same for document downloads. The Python path has no response at all there, it maps an httpx exception. The sweep does not cover transport failures yet
- `RequestFormat` maps to `litellm.UnsupportedParamsError`, which is not in the design's `PublicFailure` list. It fits as one more `Status` class (it subclasses `BadRequestError`), so no new kind is needed

## 3. Target shapes

### Core (`core-utils/src/exception_mapping_utils/`)

- `original.rs`
  - `enum OriginalException { Http{status, body, headers}, Connection{message}, Timeout{message}, Response{message}, Local{class: LocalClass, message, status: Option<u16>} }`
  - `enum LocalClass { ValueError, FileNotFound, OsError }`
  - `ProviderException`, `status_code_is_synthesized` and the status 0 sentinel are deleted
- `public.rs`
  - `struct PublicFailure { kind, message, model, llm_provider, litellm_debug_info, litellm_response_headers, provider_specific_fields, print_banner }`
  - `enum PublicKind { Status{class: StatusClass, response: Option<ResponseArg>}, Timeout{status: Option<u16>, response: Option<ResponseArg>}, ApiConnection{request: RequestStub}, Api{status: u16, request: RequestStub}, Passthrough{class: LocalClass, message} }`
  - `StatusClass` is today's simple kinds minus `ApiConnection`, plus `UnsupportedParams`
  - `ResponseArg` stays `Upstream | Stub`, without `Omitted`. `Option` says whether the constructor gets one
  - `upstream_content` goes away: the vertex case becomes `Status{response: Some(Upstream{status: <python status>, body, headers})}`, decided in the rule, so Python never patches `error.response` after construction
  - `stdout: String` becomes `print_banner: bool`. The banner text moves to `litellm/rust_bridge/failures.py`, because printing is part of stage 4 and the text today lives in Python-path-only code
- `mod.rs`: `ExceptionContext`, the message prefixes (`exception_provider`, `python_capitalize`, `extra_information`), redaction of `error_str`, the timeout marker check, and the dispatch `match provider: Provider`. `ExceptionContext.custom_llm_provider` becomes core's provider enum plus the raw string for messages, so dispatch never compares strings
- `status.rs`: the fallback table, today's `map_exception_by_status`
- `rules.rs`: `struct Rule { when: When, kind, message: MessageTemplate, response: ResponseChoice, debug: bool }` and the one interpreter `fn apply(rules: &[Rule], mapping: &Mapping) -> Option<PublicFailure>`, first match wins
- `openai.rs`, `vertex_ai.rs`, `cohere.rs`: one ordered `&[Rule]` each
- Dead branches: drop every branch that only matches a string the Python SDKs produce, for example the `BadRequestError.__init__() missing 1 required positional argument: 'param'` check in `map`. Each removal is a `Divergence` row in the sweep if the sweep can reach it, and a line in the PR body either way

### Provider config (stage 2)

- `OcrConfigKind::error_class(&ocr::Error) -> OriginalException` replaces `get_error_class`
  - `Provider{..}` -> `Http` with headers
  - `Transport::Http` -> `Http` without headers
  - `Transport::Network` -> `Timeout` when the reqwest error was a timeout, else `Connection`. `transport::Error::Network(String)` loses that bit today, so add `transport::Error::Timeout(String)` and set it in `from_reqwest_before_dispatch` and `From<reqwest::Error>`
  - Delete the invented 408 in `ocr/client.rs::transport_error` and `ocr/document.rs`. Both return `transport::Error::Timeout`. `Transport::Connect` (refused, DNS) -> `Connection`
  - The message text cannot match: Python embeds httpx's exception text and Rust has reqwest's. Parity is on class, `status_code`, the absence of a response, and the litellm-owned prefix of the message. The httpx part of the message is one recorded `Divergence` on `message` for the transport rows only
  - Invalid and oversized responses (`InvalidResponse`, `ResponseField`, `EmptyContent`, `TooLarge`, `NumericRange`) -> a new variant `OriginalException::Response{message}`. The Python path raises inside response transformation and `exception_type` turns that into whatever the sweep shows (expected `APIConnectionError`). This replaces open question 1
  - `FileRead` with `NotFound` -> `Local{FileNotFound, "File not found: <path>", None}`. Other `FileRead` -> `Local{OsError, <io message>, None}`
  - `is_request()` errors, auth errors and the other value-error cases from `core_error_to_pyerr` -> `Local{ValueError, message, Some(400)}`
  - `RequestFormat` needs its own arm, because its public class is `UnsupportedParams` and its message includes the rejected `req_format` value. Carry the value in the error (`RequestFormat{value: String}`) so the message is built in core
  - Response errors (`TooLarge`, `ResponseField`, ...) are `RuntimeError` today. `LocalClass` has no `RuntimeError`. Open question 1 below

### Bridge (`python-bridge`) and driver (`host-python`)

- `RouteHost` loses `native_error` and `map_failure(&PyErr)` and gains `fn classify(&self, py, error: <Route as Route>::Error) -> PyResult<PublicFailure>`
  - The OCR impl builds `ExceptionContext` with `failures::exception_context`, calls `config.error_class(&error)`, then `exception_type`
- `struct BridgeFailure(PublicFailure)` with `impl From<BridgeFailure> for PyErr` in `python-bridge/src/failures.rs`. It is the only caller of `public_exception`. If the Python constructor itself fails (import error, Pydantic validation), that error is what the `From` impl returns
  - `host-python` must not depend on `python-bridge`, so the driver cannot name `BridgeFailure`. The trait method returns `PyResult<PyErr>` built through the `From` impl inside the route's `classify`, or the trait gets an associated `type Failure: Into<PyErr>`. Recommendation: the associated type, because it keeps the `PublicFailure` value visible to the driver's fake route in tests
- Driver: `machine_failed` calls `classify` exactly once, only when `self.interrupted` is `None`. `failure()` loses the `FailureOrigin::Call => map_failure(...).unwrap_or(error)` arm. A `classify` error is raised with the classifier error as the exception and the native error's message attached through `__context__` (a `RuntimeError` holding `error.to_string()`), so nothing is swallowed
- Deleted: `RustUpstreamError` (class, `.pyi` entry, `bindings.native_exception_types` second slot), `provider_exception`, `map_upstream_failure`, `routes/ocr/errors.rs` (`to_pyerr`, `upstream_error`, `attach_status`), `litellm/rust_bridge/ocr/route_host.map_failure`, and at the end `failures.map_failure` with its `ExceptionMapper` protocol
- Kept: `create_exception!` for `RustBridgeDeclined` only
- Not used, per the design: `import_exception!`, lazy `PyErrArguments`, `#[pyclass(extends=PyException)]`, a `#[pyclass]` mirror of `PublicFailure`

### Shared Python (`litellm/rust_bridge/failures.py`)

- The Pydantic models mirror the new `PublicKind`. `extra="forbid"` stays
- `_simple`'s eleven near-identical arms collapse into a `Mapping[StatusClass, type[Exception]]` lookup plus one call, because every `Status` class now takes the same keyword arguments
- `Passthrough` builds the builtin (`ValueError`, `FileNotFoundError`, `OSError`) and sets `status_code` when present
- `public_exception` prints the banner when `print_banner` is true, sets `litellm_response_headers` and `provider_specific_fields`

## 4. Steps, each with its unit tests

The parity sweep must be green after every step except for recorded divergences. Every step lands with the tests listed under it. Rust tests use rstest tables and assert the whole `PublicFailure` value with `assert_eq!`, never a single field, so a mutation in any field fails a test

### 4.1 `OriginalException` and `error_class` in core

- Sweep first: add a transport axis to `test_parity.py` before touching the code, so the 408 shows up as a red row. Cases: read timeout, connection refused, body that is not JSON, JSON missing required fields, body over the size limit, each for sync and async and per provider. The local server in the sweep's support code needs a "sleep past the timeout" handler, a closed port, and a large-body handler
- Code: add `original.rs`, `transport::Error::Timeout`, remove both invented 408s, `OcrConfigKind::error_class`. `exception_type` takes `&OriginalException`. Keep a private `From<ProviderException>` shim only inside this step if needed to keep the diff reviewable, and delete it before the step's commit
- Tests (Rust, permanent)
  - `error_class`: one case per `ocr::Error` variant group listed in section 3, per provider config where the config overrides anything. The `FileRead` cases assert the exact message, including the path
  - A test that iterates every `ocr::Error` variant that `is_request()` and asserts `Local{ValueError, _, Some(400)}`, so a new request variant cannot be forgotten
  - `transport`: a timeout reqwest error becomes `Timeout`, a connect error before dispatch stays `Connect`
  - `ocr/client.rs`: replace `request_timeout_has_an_http_408_status` with a test that a timed-out request and a timed-out document download both yield `transport::Error::Timeout` and that no `Http` variant is ever produced without a real response
  - `exception_type` on `Response{..}`: whole-value case per provider family
  - `exception_type` on `Connection` and `Timeout` inputs: replaces today's `synthesized_status_is_a_connection_error`

### 4.2 `classify` on the trait, one `From` impl, OCR moved over

- Code: trait change in `host-python/src/adapter.rs`, driver change in `driver.rs`, `BridgeFailure`, OCR `classify`, delete `routes/ocr/errors.rs` and the `RustUpstreamError` re-parsing
- Dead stand-ins go in this step, because fixing the `unwrap_or(error)` swallow removes the last way a `RustUpstreamError` can escape
  - Delete `runtime._raise_upstream`, the `except upstream` arms in `attempt` and `aattempt`, and the second slot of `bindings.native_exception_types`
  - Delete `map_failure` from `chat_completions/route_host.py`, `messages/route_host.py` and `responses/route_host.py`, with their tests under `tests/test_litellm/rust_bridge/<route>/`
  - `chat_completions_error_to_pyerr` still names `RustUpstreamError`. Chat has no rule in `catalog.RULES`, so map those two arms to `PyRuntimeError` for now and let 4.5 replace the function
  - `test_runtime.py`: drop the `_raise_upstream` cases, keep and extend the declined-falls-back cases
- Tests (`host-python` driver, existing fake route with its `log`)
  - A native error reaches `classify` exactly once and the `Failed` event carries the classified error
  - A host-raised error during an op skips `classify` and is emitted unchanged (`self.interrupted` path)
  - A cancellation is returned as is, with no `classify` call and no `Failed` event
  - A `classify` that returns `Err` surfaces that error, and the native error's text is reachable from `__context__`. This is the regression test for `unwrap_or(error)`
  - Failures in `begin` and `after_success` stay unmapped (already covered, keep)
- Tests (`python-bridge`)
  - `exception_context`: reads `suppress_debug_info` and `redact_messages_in_exceptions` from `litellm`, reads `vertex_project`, `vertex_location` and the four metadata keys, tolerates `metadata=None` and a non-mapping `metadata`, stringifies non-string values the way Python's `str()` does
  - `BridgeFailure -> PyErr`: a valid payload yields the litellm class. A payload the Python side rejects yields the Pydantic error, not a panic and not a silent fallback

### 4.3 Reshape `PublicFailure`, the constructor, and add the golden fixtures

- Code: `public.rs`, the Pydantic models, the collapsed constructor, banner text moved to Python, `provider_specific_fields`
- Tests
  - Rust: a test builds one `PublicFailure` per kind (`Status` once per `StatusClass`, `Timeout` with and without status, `ApiConnection`, `Api`, `Passthrough` once per `LocalClass`), serializes them and compares against `tests/test_litellm/rust_bridge/fixtures/public_failures/*.json`. An env switch regenerates the files. The comparison is what runs in CI
  - Python, permanent, in `tests/test_litellm/rust_bridge/test_failures.py`: parametrized over the fixture files. For each one assert the exact class, `isinstance` of the matching `openai.*` type, `status_code`, `message`, `llm_provider`, `model`, response body and headers, `litellm_response_headers`, `provider_specific_fields`, and that the banner is written to stdout only when `print_banner` is true
  - Python: an unknown field and an unknown `StatusClass` are both rejected. A test asserts every `StatusClass` literal has a constructor entry and a fixture, so adding a class on one side fails on the other
  - Delete the three `map_failure` tests at the top of `test_failures.py` in step 4.4, not here

### 4.4 Cover `Local`, then delete the Python fallback

- Code: classifier handles `Local`. `ValueError` with status 400 follows what the Python `exception_type` does with it today for OCR (the sweep is the oracle for whether that is `BadRequestError`, `APIConnectionError` or a passthrough, per provider). `FileNotFound` and `OsError` become `Passthrough`. `RequestFormat` becomes `Status{UnsupportedParams}`. Then delete `ocr/route_host.map_failure`, the Python call in `host.rs` and `failures.map_failure` with its `ExceptionMapper` protocol. After this step nothing under `litellm/rust_bridge/` imports `litellm.exception_type`. Add a test in `tests/test_litellm/rust_bridge/` that fails if any module there references it, since that is the invariant the step exists for
- Tests
  - Rust: one table row per `LocalClass` x status x provider family, whole-value assertions
  - Rust: `RequestFormat{value}` message equals the Python text, including `repr` quoting of the value. Cover a string with a quote in it
  - Sweep: extend `test_parity.py` with a validation axis, written before the port so the rows decide the table above. Cases: bad `req_format`, empty file, missing file, unreadable file, missing document URL, invalid MIME type, invalid data URI, Cohere with a PDF, Reducto with a plain URL, invalid `pages`, invalid provider, missing Azure AI credentials, missing Document Intelligence credentials, missing Reducto key. Each for sync and async. Every row asserts no request reached the local server
  - `tests/test_litellm/rust_bridge/ocr/test_route_host.py`: drop the `map_failure` cases with the function

### 4.5 Chat, messages, responses, audio transcription (later, not part of the OCR work)

- These routes are not on the driver. They also have decline semantics that OCR does not: an error before dispatch must stay `RustBridgeDeclined` so `runtime.attempt` can fall back to Python
- Code
  - Per route in core: `fn disposition(&Error) -> Disposition` with `enum Disposition { Declined, Failed(OriginalException) }`. This moves the pre-dispatch list out of `chat_completions_error_to_pyerr` into core, where it can be tested without Python
  - Per route provider config: `error_class`
  - Bridge: build `ExceptionContext` at entry while attached, before `run_sync` / `run_async` detach. The `map_err` closure becomes `Declined -> RustBridgeDeclined`, `Failed(original) -> BridgeFailure(exception_type(&context, &original)).into()`
  - `RustUpstreamError` itself is deleted here, with its `.pyi` entry
- Tests
  - Rust core: `disposition` table per route, one row per error variant. Assert that every variant is covered by matching exhaustively in the test helper, so a new variant fails to compile until it has a row
  - Rust core: `error_class` per provider config
  - Python `test_runtime.py`: a declined call still falls back, a classified litellm exception passes through `attempt` untouched (no wrapping, no prefix), sync and async
  - Sweep: add `tests/test_litellm_rust/chat/` and `messages/` parity files using the same `support/parity.py`
  - Proxy and Router: one smoke sweep per route, same shape as 4.7

### 4.6 Split the classifier into rule tables

- Done last on purpose: by then the sweep and the whole-value tests pin the behavior, so the split is a pure refactor
- Code: `rules.rs`, `status.rs`, `openai.rs`, `vertex_ai.rs`, `cohere.rs`. Existing test cases move next to their rule file unchanged
- Tests
  - One case per rule, named after the rule
  - One case per pair of rules that can both match, asserting the earlier one wins (for example a 400 whose body has both a context-window marker and a content-policy marker)
  - Separate test modules for `status.rs` (every mapped status plus one unmapped), `extra_information` (each optional context field present and absent), redaction (message redacted, markers still matched on the redacted string), and `print_banner` against `suppress_debug_info`
  - Run `cargo mutants -p litellm-core-utils --file 'src/exception_mapping_utils/**'` and hold the 90% kill rate. Survivors become table rows

### 4.7 Router and proxy smoke sweep

- `tests/test_litellm_rust/ocr/test_router_and_proxy.py` has five hand-written tests today and they pass. Shrink it to one parametrized sweep: `(upstream status, body, headers) -> (proxy HTTP status, error JSON type and code, retried or not, deployment cooled down or not)`, run against both backends through `support/parity.py`
- Keep one success row. Add one row each from the transport and validation axes, so the Router's retry and cooldown decisions are checked for a `Timeout`, an `APIConnectionError` and a `Passthrough`
- Router and proxy only consume the exception, so nothing beyond this smoke sweep belongs here. A failure in it is fixed in the classifier or the constructor, never in the Router
- Can be done at any point. Doing it right after 4.1 gives the later steps a cheaper end-to-end signal

### 4.8 Known gaps in the `exception_type` port

All three are unreachable for today's OCR models, so they cannot be `Divergence` rows (`assert_parity` fails a registered divergence that does not diverge). They are recorded as a `KNOWN_GAPS` doc comment at the top of `exception_mapping_utils/mod.rs` and in the PR body. Each has a trigger that says when it stops being acceptable

- The Vertex partner URL for "claude" models in `extra_information`. Trigger: a Vertex route whose models include Anthropic partner models. Then `api_base` gets the branch and a table row
- A stripped model name that happens to resolve in `get_llm_provider`, which changes the provider Python reports. Trigger: a route whose model names overlap the model cost map. Needs the provider resolution port, not a classifier change
- Python tracebacks embedded in fallback messages (`traceback.format_exc()` in the `APIConnectionError` fallback). Rust cannot reproduce this and should not. Permanent, deliberate divergence. If a sweep row ever reaches it, register a `Divergence` on `message` and compare the prefix before the traceback

## 5. Test ownership summary

| Contract | Owner | Location | Lifetime |
|---|---|---|---|
| `Error -> OriginalException` | core provider config | `core/src/<route>/provider_config.rs` tests | permanent |
| `Error -> Declined \| Failed` | core route | `core/src/<route>/error.rs` tests | permanent |
| `(OriginalException, ExceptionContext) -> PublicFailure` | core classifier | next to each rule file | permanent |
| `PublicFailure` JSON shape | core + shared Python | golden fixtures, written by Rust, read by `test_failures.py` | permanent |
| classify once, host errors skip, cancellation, classifier failure surfaces | `host-python` driver | `driver.rs` tests with the fake route | permanent |
| `litellm.*` globals and kwargs -> `ExceptionContext` | `python-bridge` | `failures.rs` tests | permanent |
| `num_retries`, `timeout`, `str(error)` | Python dispatch | `test_dispatch.py` | permanent |
| Python path vs Rust path | nobody, it is an oracle | `tests/test_litellm_rust/` | deleted with the Python path |

## 6. Open questions

1. Polling errors (`PollTimeout`, `PollLocation`, `PollOrigin`, `OperationStatus`) are `RuntimeError` today. They likely fit `Timeout` and `Response`, but the Python polling code decides. Add sweep rows for Document Intelligence polling before choosing
2. The mapped error's `__context__` is the `RustUpstreamError` today. After 4.2 there is no native `PyErr`. If the sweep compares `__context__` or `__cause__`, record one divergence. Otherwise nothing to do
3. `ExceptionContext` needs core's provider enum. OCR has its own `OcrProvider`, and chat has another. If there is no shared enum yet, 4.1 uses a small `enum ExceptionFamily { OpenAiCompatible, VertexAi, Cohere, Other }` derived by each route, and the shared provider enum is a separate change
