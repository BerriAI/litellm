# gateway-inference folder architecture

This crate is the reusable, framework-independent inference domain. It is used
by `litellm-gateway-server` and `litellm-python-bridge`

Axum routes, auth extractors, application state, startup, and Tower dependencies
belong in `litellm-gateway-server`. This crate must not depend on the server

Transport-neutral inference orchestration lives under `src/runtime/`. Existing OCR,
audio transcription, realtime I/O, and callback integrations remain here as
migration seams until their provider call paths move into `litellm-core`
