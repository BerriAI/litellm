litellm-core owns provider transformations and shared types. Existing non-OCR top-level calls may still own their full call path.

A route module owns everything the call needs: types, the provider template trait, provider transforms (under `providers/`), provider/auth/URL resolution, and the handler that performs the HTTP call. Handlers belong here, never in a host crate.

Not here: serving HTTP (axum routes, extractors), config file reading, rollout state, databases, or callback dispatch. Env reads are limited to credential fallback in a route's `prepare.rs`.

Routes (messages, ocr, realtime) and providers (anthropic, mistral, openai) are modules, not crates.

OCR is transformation-only in core: supported parameter metadata, parameter mapping, request body/data transformation, and decoded provider response normalization. OCR resolution, auth, environment, URL/headers, document materialization, HTTP, polling, and lifecycle ordering belong in `litellm-runtime`.
