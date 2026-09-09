# Rust bridge tests

Put API-specific tests under `ocr/`, `messages/`, or `chat/`. Keep request construction, recording servers, callback recorders, and state isolation in `support/`. Integration tests and their fixtures belong in `integrations/`.

Use `backend` with `"python"` and `"rust"` for parity tests. Keep callback mutation and lifecycle assertions next to the API surface that owns them. Shared route parametrization belongs in `integrations/routes.py` only when its observable contract is the same for every route.

Run the suite with `LITELLM_RUST=1 uv run pytest tests/test_litellm_rust`.
