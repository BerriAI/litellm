# LiteLLM Gateway Inference

`litellm-gateway-inference` is the reusable, framework-independent inference
domain used by the Rust gateway server and Python bridge

It owns inference integrations, transport-neutral orchestration, and legacy
OCR, audio transcription, and realtime I/O that have not yet moved into
`litellm-core`. It has no Axum or Tower dependency

The executable server, HTTP/WebSocket routes, auth extractors, application
state, config startup, Docker image, and deployment files live in
[`../gateway-server`](../gateway-server)
