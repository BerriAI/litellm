# Forwarding gateway

`Forward` is a reverse proxy in front of the Python proxy on loopback. Python must see each request as if the client had connected to it directly, because every Python route (inference, MCP, pass-through, UI, WebSocket) is served behind it without Rust knowing which one it is

- Never branch on a route or path. A behavior one endpoint needs is a protocol behavior every endpoint gets
- Forward the method, request target, headers, and body byte for byte. Only drop hop-by-hop headers (RFC 9110 section 7.6.1) and set `x-litellm-rust-peer`, which replaces any client copy
- Stream both bodies. Never buffer, cap, or time out a body, since uploads and provider streams have no upper bound
- A client disconnect closes the upstream connection at every stage, because that is how Python learns to stop a provider stream
- An upstream that dies mid-response fails the client body. Never end it cleanly, or SSE clients treat a truncated stream as complete
- Pooled upstream connections are retired well before Python's keep-alive timeout, so a request never lands on a connection Python is closing
- WebSocket is the one protocol switch Python serves. Upgrades are relayed as raw bytes after the 101 and drained on shutdown

## Tests

Test through real sockets with `tests/support`, never `oneshot`. `Gateway` runs the forwarder on a listener like production, `recording` is a well-behaved upstream that captures what Python would receive, and `raw_upstream` and `Raw` drive bytes by hand for framing, timing and disconnects a real server would not produce. `tests/roundtrip.rs` checks the forwarding invariant for generated requests and responses. Add a protocol edge case there or as a focused test next to its neighbors
