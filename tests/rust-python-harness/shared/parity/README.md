# Implementation parity testing through the SDK interface

> Given the same SDK call and identical provider behavior, do two implementations expose the same SDK contract?

## What the harness compares

- A fixture contains a LiteLLM SDK input and a recorded upstream provider response
- The same LiteLLM input is transformed by isolated baseline and candidate implementations
- The resulting provider requests must match in method, path, headers, and body, excluding runtime-specific HTTP metadata
- The recorded provider response is then replayed unchanged to both workers
- The harness compares the values returned through the Python SDK interface
- Non-streaming responses are compared directly, including their concrete return type and public model fields
- Streaming responses are consumed and compared chunk by chunk, including wrapper type, chunk type and order, termination, and public exception behavior
- Failed SDK calls are compared by exception class, stable message, status, code, model, provider, and parameter fields
- Traceback paths and line numbers are excluded because they are runtime-specific
- Route-specific comparators and chunk normalizers handle differences in each public SDK contract

## Process isolation

- SDK object and stream parity runs both implementations sequentially in the same process so tests can retain returned objects
- Every test saves and restores the original bridge state
- A small subprocess smoke test verifies environment-based startup configuration and detects fallback to the Python HTTP implementation

## Streaming execution

The invocation callback passed to `run_in_process` must consume the stream before returning its `StreamOutcome`.
Use `consume_sync_stream` inside that callback, or await `consume_async_stream` inside the callback passed to
`run_in_process_async`. Provider requests are collected only after the callback completes. Streaming is explicit:
an iterable return value alone does not select stream consumption

The consumers retain the wrapper type, iteration capabilities, chunk types and order, and any partial output before
an error. Errors retain their creation or iteration phase and the full public `SDKError` fields, with traceback text
removed. `capture_sync_stream` and `capture_async_stream` consume through the same helpers and then serialize the
outcome for subprocess reports. A serialization failure raises as a harness failure rather than becoming an SDK error

Response models and stream chunks share a recursive comparator. It compares concrete model, container, and scalar
types, public fields and extras, and exact values while ignoring Pydantic private attributes at every nesting level.
An API may supply an explicit chunk normalizer for its public contract

Shared tests exercise a local SSE provider through recording, VCR cassette storage, replay, and typed event comparison
in sync and async modes. They cover fragmented events, split UTF-8 characters, CRLF framing, coalesced events, and
application errors within a normally completed HTTP stream. HTTP byte boundaries and decoded SDK event boundaries
are checked separately

OCR remains the only integrated LiteLLM route. These tests validate shared streaming machinery, not another route's
SDK parity. Connection interruption, early cancellation, and lifecycle timeout enforcement remain outside this coverage

## Hypothesis and property-based testing

- Hypothesis is a Python library for property-based testing
- Example-based tests use inputs selected by the test author
- Property-based tests define strategies for valid inputs and properties that must hold for every generated example
- Hypothesis generates combinations from those strategies and normally shrinks a failing example to a smaller reproducible case
- In this harness, Hypothesis is used only during fixture generation to expand the LiteLLM input corpus
- Each API owns the strategies that vary its supported inputs
- Fixture generation is deterministic, and each generated input is recorded with the raw provider response it received
- The parity tests use committed fixtures and do not call the provider or generate new Hypothesis examples
- Provider responses are replayed unchanged, so the parity test does not fuzz or validate provider behavior
- Because Hypothesis does not run the parity assertion directly, parity failures are not automatically shrunk

## API-owned fixtures

The shared package owns recording, replay, persistence, execution, comparison, and route-neutral media constructors.
Each API package owns its input models, explicit strategies, provider targets, route-specific assets, fixture directory,
and regeneration command. See the API package documentation for its configured contracts and recording command

## VCR cassettes

Fixtures use VCR's YAML `version: 1` format with ordered request/response `interactions`. VCR handles text and binary
body serialization. Each cassette also contains `recorded_at`, `ttl_seconds: 0` (committed fixtures never expire), and
`x-litellm` metadata holding the SDK input and request provenance. Streaming responses carry
`x-litellm-chunk-lengths` so local replay preserves the original byte boundaries

The recording server captures requests before forwarding their responses. Saved requests use the stable
`http://parity-provider.invalid` origin and strip authentication headers and credential query parameters. The upstream
request keeps its credentials. Provider response bytes and non-success statuses are preserved

Standard VCR can load these files and replay their interactions. Parity tests keep using the local HTTP server because
Rust HTTP calls do not pass through VCR's Python patches. The harness still compares the two implementations' requests
against each other; the saved request is available for inspection and VCR playback, not a new parity assertion

Refresh parity cassettes through the API's recording command. Generic VCR writers do not preserve the SDK metadata

Legacy JSON fixtures remain readable. Migrated cassettes mark reconstructed requests as `python_replay`; fresh
recordings use `recorded`. The metadata extensions follow the filesystem cassette layout proposed in
[PR #39338](https://github.com/BerriAI/litellm/pull/39338), without depending on its unmerged persistence backend

## References

- [Hypothesis documentation](https://hypothesis.readthedocs.io/en/latest/)
- [Hypothesis documentation source](https://github.com/HypothesisWorks/hypothesis/tree/master/hypothesis/docs)
