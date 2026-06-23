litellm-core is the PURE translation layer: request/response/event types, route contracts (traits), and provider transforms (modules under `providers/`). No network, no I/O, no env reads.

Routes (ocr, realtime) are modules, not crates.
