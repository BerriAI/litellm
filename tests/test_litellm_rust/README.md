# Rust bridge tests

Put API-specific tests under `ocr/`, `messages/`, or `chat/`. Keep request construction, recording servers, callback recorders, and state isolation in `support/`. Under `integrations/`, keep registry inventory in `test_catalogue.py`, Python/Rust comparisons in `test_backend_parity.py`, callback ownership and completion in `test_callback_lifecycle.py`, individual exporters in `test_exporters.py`, and only multi-integration interactions in `test_composition.py`

Use `backend` with `"python"` and `"rust"` for parity tests. A test name must identify the observer and expected result. Keep multiple assertions together only when they are consequences of one mutation, failure, scheduling, or lifecycle event. Record callback observations and assert them after the callback returns because production logging can swallow callback exceptions. Shared route parametrization belongs in `support/routes.py` only when its observable contract is the same for every route

The catalogue stores labels for selected behavioral cases. Those labels are inventory metadata and do not prove that matching pytest cases exist or execute

Run `make test-rust-extension` as the acceptance command. It builds a fresh wheel, installs that wheel into a temporary environment, requires `LITELLM_RUST=1`, and runs this suite with isolated Python imports

Collection fails when `LITELLM_RUST=1` is set but the compiled `_native` module cannot be imported. The autouse fixture selects Rust unless a parity or fallback case explicitly selects Python. Backend selection alone does not prove native execution because a public route can fall back. Tests that claim native request coverage must also assert either the `x-litellm-rust` response marker or a wire property that distinguishes the native client. `test_public_ocr_executes_the_compiled_extension_without_python_fallback` is strict and its server rejects Python HTTPX requests

The retained callback contract modules are marked as non-strict expected failures until the implementation from #40070 lands. Harness tests and the compiled-extension OCR smoke test remain strict. Passing contract cases appear as XPASS so staging coverage stays visible

`isolated_backend` restores the previous backend override and callback state on exit, including after exceptions or cancellation. Scopes can nest in the same task. Run backend comparisons sequentially: overlapping scopes in different tasks raise before changing process-global state. Use separate worker processes for parallel backend comparisons
