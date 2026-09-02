# Python/Rust parity testing in the Python SDK interface

> Given the same SDK call and identical provider behavior, does the PyO3 implementation behave same as Python?

## What the harness compares

- A fixture contains a LiteLLM SDK input and a recorded upstream provider response
- The same LiteLLM input is transformed by isolated Python and Rust workers
- The resulting provider requests must match in method, path, headers, and body, excluding runtime-specific HTTP metadata
- The recorded provider response is then replayed unchanged to both workers
- The harness compares the values returned through the Python SDK interface
- Non-streaming responses are compared directly, including their concrete return type and public model fields
- Streaming responses are consumed and compared chunk by chunk, including wrapper type, chunk type and order, termination, and public exception behavior
- Failed SDK calls are compared by exception class, stable message, status, code, model, provider, and parameter fields
- Traceback paths and line numbers are excluded because they are runtime-specific
- Route-specific comparators and chunk normalizers handle differences in each public SDK contract

## Process isolation

- SDK object and stream parity runs Python and Rust sequentially in the same process so tests can retain the returned objects
- Every test saves and restores the original bridge state
- A small subprocess smoke test verifies environment-based startup configuration and detects fallback to the Python HTTP implementation

## Hypothesis and property-based testing

- Hypothesis is a Python library for property-based testing
- Example-based tests use inputs selected by the test author
- Property-based tests define strategies for valid inputs and properties that must hold for every generated example
- Hypothesis generates combinations from those strategies and normally shrinks a failing example to a smaller reproducible case
- In this harness, Hypothesis is used only during fixture generation to expand the LiteLLM input corpus
- The current OCR strategy varies supported optional parameters while keeping inputs valid
- Fixture generation is deterministic, and each generated input is recorded with the raw provider response it received
- The parity tests use committed fixtures and do not call the provider or generate new Hypothesis examples
- Provider responses are replayed unchanged, so the parity test does not fuzz or validate provider behavior
- Because Hypothesis does not run the parity assertion directly, parity failures are not automatically shrunk

## Recording fixtures

The recording command runs four explicit stages:

1. Generate deterministic SDK inputs for every configured OCR target
2. Build target-scoped, deduplicated recording jobs
3. Record upstream responses through one globally bounded worker pool
4. Persist each fixture and report whether it was recorded, cached, or failed

Run it with:

```shell
uv run python -m tests.test_litellm.ocr.fixtures.record --examples 4 --concurrency 4
```

`--concurrency` caps all provider calls across all targets. Each completed job is reported immediately, and the final
summary reports recorded, cached, and failed totals. Independent jobs finish after a failure, then the command exits
nonzero if any job failed

## OCR input boundaries

OCR strategies generate only public `litellm.ocr()` and `litellm.aocr()` inputs. Every case contains the normalized
`model`, Mistral-shaped `document`, optional `custom_llm_provider`, and LiteLLM keyword arguments. The fixture-only
`boundary` tag selects the valid keyword-argument set and is removed before calling the SDK. Strategies never build
provider wire payloads.

Each boundary has a required corpus containing a baseline and one case for every supported top-level LiteLLM OCR
parameter for every active registered model that uses that transformation. Models whose registry deprecation date has
passed are excluded. `--examples` controls additional Hypothesis-generated cases; it does not replace the required
corpus.

The explicit boundaries are Mistral, Azure-hosted Mistral, Vertex-hosted Mistral, Azure Document Intelligence,
Vertex DeepSeek, Reducto v3, and Reducto legacy. Provider credentials and endpoints only control target discovery, so
a machine records the boundaries it has configured and skips the rest. Azure-hosted Mistral enumerates its active
registered models rather than requiring a separately configured deployment model. Reducto fixtures record both upload
and parse responses. Their parity cases are non-strict expected failures until the Rust OCR bridge supports Reducto, so
both expected failures and unexpected passes keep CI green during the rollout.

The committed corpus does not need to contain live recordings for every configured target. In particular, Azure and
Vertex generation paths are covered by unit tests without requiring their credentials in CI. Recordings can be added
later without changing the fixture schema or runner.

Invalid OCR inputs do not use recorded provider responses. The parity suite checks unsupported providers and models,
malformed documents, invalid request formats, invalid Azure Document Intelligence parameters, and invalid headers in
both sync and async SDK calls. These cases must return the same public exception fields without sending a provider
request. A malformed Azure Document Intelligence document reaches the Rust bridge so its native validation error is
also compared against Python

## References

- [Hypothesis documentation](https://hypothesis.readthedocs.io/en/latest/)
- [Hypothesis documentation source](https://github.com/HypothesisWorks/hypothesis/tree/master/hypothesis/docs)
