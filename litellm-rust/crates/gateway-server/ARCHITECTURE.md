# gateway-server architecture

The Rust gateway server owns the Axum binary, HTTP/WebSocket routes, auth
extractors, application state, and startup config. It delegates transport-neutral
orchestration and callback integrations to `litellm-ai-gateway`

```mermaid
flowchart LR
  C[client] <--> S[litellm-gateway-server<br/>Axum host]
  S --> G[litellm-ai-gateway<br/>runtime and integrations]
  G --> K[litellm-core]
  G <--> O[OpenAI realtime]
  G -. spend tracking callback .-> P[litellm proxy]
  F[litellm-config<br/>load-time only] --> S
  F -. Python backend .-> P
```
