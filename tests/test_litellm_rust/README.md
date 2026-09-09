# Rust bridge tests

Put API-specific tests under `ocr/`, `messages/`, or `chat/`. Keep request construction, recording servers, callback recorders, and state isolation in `support/`. Integration tests and their fixtures belong in `integrations/`.

Use `backend` with `"python"` and `"rust"` for parity tests. Keep callback mutation and lifecycle assertions next to the API surface that owns them. Shared route parametrization belongs in `integrations/routes.py` only when its observable contract is the same for every route.

Run the suite with `LITELLM_RUST=1 uv run pytest tests/test_litellm_rust`.

`isolated_backend` restores the previous backend override and callback state on exit, including after exceptions or cancellation. Scopes can nest in the same task. Run backend comparisons sequentially: overlapping scopes in different tasks raise before changing process-global state. Use separate worker processes for parallel backend comparisons
