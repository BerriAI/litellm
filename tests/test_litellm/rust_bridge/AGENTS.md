# Rust bridge tests

Test what each side of the bridge does, not the rollout policy that picks a side. `LITELLM_RUST` and `catalog.RULES` change every time a route or backend rolls forward, so a test that sets the env var or patches the catalog to reach a path goes red on a policy change even when the code under test is fine

Call each path directly with an explicit decision instead. The Python path is the implementation the dispatcher falls back to, e.g. `litellm.ocr.main.ocr`. The Rust path is the native binding, e.g. `NATIVE_OCR.load()` from `litellm/rust_bridge/ocr/entrypoints.py`, called with the request, args and kwargs that dispatch would hand it. When the native side reads a policy-derived setting such as `settings.secret_manager().native`, pin that field in the test instead of deriving it from the catalog. `ocr/test_secrets.py` shows the pattern

Rollout policy itself, meaning which rule matches and what `LITELLM_RUST` changes, belongs in `test_catalog.py`, `test_configuration.py` and `test_dispatch.py`, tested against rules the test builds rather than the shipped `catalog.RULES`
