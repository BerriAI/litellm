# ai-gateway folder architecture

This crate is a reusable, framework-independent gateway runtime and integration
library. It is used by `litellm-gateway-server` and `litellm-python-bridge`

Axum routes, auth extractors, application state, startup, and Tower dependencies
belong in `litellm-gateway-server`. This crate must not depend on the server

Transport-neutral route orchestration lives under `src/runtime/`. Existing OCR,
audio transcription, realtime I/O, and callback integrations remain here as
migration seams until their provider call paths move into `litellm-core`
