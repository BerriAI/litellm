# Forwarding gateway

`Forward` is a reverse proxy in front of the Python proxy on loopback. Python must see each request as if the client had connected to it directly, because it serves every route behind Rust without Rust knowing which one

## Rules

### What Python and the client receive

- Forward the method, target, headers and body byte for byte within the [URL contract](FORWARDING.md#url-fragments-and-literal-hashes). Only drop hop-by-hop headers (RFC 9110 section 7.6.1) and set `x-litellm-rust-peer`, which replaces any client copy
- Never invent a Host. Python builds absolute URLs from it, so a missing one must never become the private upstream address
- A request Python could not have received unchanged (HTTP/1.1 without Host, a transfer coding other than chunked, a target without a path) gets the status uvicorn would have given, from Rust
- Stream both bodies. Never buffer, cap or time out a body, since uploads and provider streams have no upper bound
- An upstream that dies mid-response fails the client body. A clean end would make SSE clients treat a truncated stream as complete
- A client disconnect closes the upstream connection at every stage, which is how Python learns to stop a provider stream

### Connections

- Serve HTTP/1.1 only, as uvicorn does. WebSocket is the one protocol switch, relayed as raw bytes after the 101 and drained on shutdown
- Retire pooled upstream connections at half of Python's keep-alive timeout, so no request lands on one Python is closing
- Close idle client connections after Python's keep-alive timeout, as uvicorn did. Time only the wait for the next request, never an exchange in progress
- Drain on EOF of stdin exactly as on SIGTERM, since the supervisor holds the only write end. `input_closed` reads on a plain thread, because a read blocked on tokio's blocking pool would hold up the runtime's exit

### The native app and `admit`

- Take the native app as a tower `Service`, never an axum `Router`. Never branch on a route or path or attach a route layer. A behavior one endpoint needs is a protocol behavior every endpoint gets
- Never forward to Python from an axum fallback, on a router or a method router. The route table would then decide part of what Python answers, and native layers would wrap those requests. `admit` is the only place that decides
- Take `admit` as a plain function beside it, never a `Layer`. Only `admit::in_front_of` composes the two, and only `serve` calls it
- `admit` runs before every native layer. Only the connection and `ClientSocket` sit above it, for both answerers
- Only `admit` sends a request to Python. A refused request touches no native layer, and an admitted one never reaches Python, even on a native 404
- A refused request comes back from `admit` as the client sent it. The types cannot hold this, so `gateway/tests/forward.rs` does
- Keep `Forward` crate-private and a leaf, so no caller can wrap it, layer it or fall back to it
- Take the reason as `Display + Into<&'static str>`, and never depend on the gateway crate
- Set `x-litellm-gateway` with `overriding`, outside the whole native app and outside `Forward`. It reads `rust` or `python; reason=<why>`, and a `NotForwarded` status counts as `python`

## Layout

Every decision has one home

| File | Owns |
|---|---|
| `lib.rs` | The only wiring, and the only place a server starts. `serve` makes the one `Connections` that `Forward` and `server::serve` share |
| `server.rs` | Accepting client connections and their timeouts, which `axum::serve` does not expose |
| `connections.rs` | Every connection and relay, and the shutdown sequence |
| `client.rs` | `ClientSocket` in each request's extensions, and the one function that reads it back |
| `forward.rs`, `websocket.rs` | Moving requests, bodies and upgraded connections, never touching a header |
| `rewrite.rs` | What Python and the client receive, as pure functions |
| `error.rs` | `NotForwarded`, the only place that picks a status the forwarder answers itself |
| `admit.rs` | `in_front_of` and the `x-litellm-gateway` values |
| `connector.rs` | Keeping hyper reading Python's answer after Python stops reading the request |
| `settings.rs` | Parsing the environment, so an `Upstream` is a loopback origin by construction |

## Environment

Only the Python supervisor in `litellm/proxy/rust_forward/` starts forwarding mode, from `litellm` and from `python -m gateway.launch`. `supervisor/children.py` is the only writer of these variables, so adding, renaming or removing one changes `settings.rs` and `children.rust_spec` in the same change. Every variable is required, and a missing or invalid one fails startup naming it. Without `LITELLM_FORWARD_UPSTREAM` the binary runs as the standalone Rust gateway and this crate does nothing

Never read Python's own settings (`HOST`, `PORT`, `KEEPALIVE_TIMEOUT`, `GRACEFUL_SHUTDOWN_TIMEOUT`) or keep a copy of its defaults, since Python can take those from CLI flags the environment never shows

| Variable | Meaning |
|---|---|
| `LITELLM_FORWARD_UPSTREAM` | Python's loopback origin, e.g. `http://127.0.0.1:53211` |
| `LITELLM_FORWARD_HOST`, `LITELLM_FORWARD_PORT` | public bind address |
| `LITELLM_FORWARD_UPSTREAM_KEEPALIVE` | seconds uvicorn keeps an idle connection open |
| `LITELLM_FORWARD_SHUTDOWN_DRAIN` | seconds to drain on shutdown, already including Python's graceful timeout and teardown headroom |
| `LITELLM_FORWARD_CONFIG` | absolute path of the local YAML file Python loads its config from, empty when it loads none. This crate only parses it, and the gateway loads the file |

## Tests

Test through real sockets with `tests/support`, never `oneshot`. `Gateway` runs the forwarder on a listener like production, `recording` captures what Python would receive, and `raw_upstream` and `Raw` drive bytes by hand for framing, timing and disconnects a real server would not produce. Waits measured in minutes use paused tokio time

There is one test file per traffic shape uvicorn can emit and never one per route. Add a new shape file only for a wire behavior none of them covers

| File | Shape |
|---|---|
| `exchange` | plain request and response |
| `framing` | requests whose framing a parser could read two ways |
| `slow_head` | minutes before response headers |
| `stream`, `idle_stream` | incremental response bodies, and long-lived streams that go quiet |
| `upload`, `download`, `websocket`, `client_identity` | as named |
| `lifecycle` | disconnects, pooling, idle timeouts, shutdown |
| `roundtrip` | the forwarding invariant for generated messages |
