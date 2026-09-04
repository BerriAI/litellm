# LiteLLM AI Gateway Runtime

`litellm-ai-gateway` is the reusable, framework-independent runtime used by the
Rust gateway server and Python bridge

It owns gateway integrations, transport-neutral route orchestration, and legacy
OCR, audio transcription, and realtime I/O that have not yet moved into
`litellm-core`. It has no Axum or Tower dependency

The executable server, HTTP/WebSocket routes, auth extractors, application
state, config startup, Docker image, and deployment files live in
[`../gateway-server`](../gateway-server)
