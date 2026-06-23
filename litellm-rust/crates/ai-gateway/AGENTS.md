litellm-ai-gateway is the host/routes layer — the only crate allowed to touch the network. It performs HTTP/WebSocket I/O, attaches auth headers, and runs the end-to-end route functions (`run_ocr`, `realtime`) using the pure transforms from litellm-core.

No transform/business logic lives here — that belongs in litellm-core.
