# gateway-server architecture

The Rust gateway server is the root composition crate. It owns the Axum binary,
HTTP/WebSocket routes, auth extractors, application state, and startup config.
It delegates inference behavior to `litellm-gateway-inference`; future domain
crates plug into this composition root the same way

```mermaid
flowchart LR
  C[client] <--> S[litellm-gateway-server<br/>Axum host]
  S --> G[litellm-gateway-inference<br/>runtime and integrations]
  G --> K[litellm-core]
  G <--> O[OpenAI realtime]
  G -. spend tracking callback .-> P[litellm proxy]
  F[litellm-config<br/>load-time only] --> S
  F -. Python backend .-> P
```
