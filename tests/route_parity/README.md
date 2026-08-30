# Python/Python parity testing in Python SDK interface

> Given the same SDK call and identical provider behavior, does the PyO3 implementation behave same as Python?

## What the harness compares

- A fixture contains a LiteLLM SDK input and a recorded upstream provider response
- The same LiteLLM input is transformed by isolated Python and Rust workers
- The resulting provider requests must match in method, path, headers, and body, excluding runtime-specific HTTP metadata
- The recorded provider response is then replayed unchanged to both workers
- Each worker serializes its normalized LiteLLM SDK response to JSON, and the results must match

## Hypothesis and property-based testing

- Hypothesis is Python libary for property-based testing
- Example-based tests use inputs selected by the test author
- Property-based tests define strategies for valid inputs and properties that must hold for every generated example
- Hypothesis generates combinations from those strategies and normally shrinks a failing example to a smaller reproducible case
- In this harness, Hypothesis is used only during fixture generation to expand the LiteLLM input corpus
- The current OCR strategy varies supported optional parameters while keeping inputs valid
- Fixture generation is deterministic, and each generated input is recorded with the raw provider response it received
- The parity tests use committed fixtures and do not call the provider or generate new Hypothesis examples
- Provider responses are replayed unchanged, so the parity test does not fuzz or validate provider behavior
- Because Hypothesis does not run the parity assertion directly, parity failures are not automatically shrunk

## References

- [Hypothesis documentation](https://hypothesis.readthedocs.io/en/latest/)
- [Hypothesis quick start](https://hypothesis.readthedocs.io/en/latest/quickstart.html)
