# Rust forwarding, Python half

With `LITELLM_RUST` on, a supervisor runs the Rust gateway on the public port and Python on a private loopback listener. Rust forwards every request to Python unchanged, so Python still serves every route. This package is the Python half of forwarding and pairs with the crate `litellm-rust/crates/gateway-forward`. Once Rust serves every route, both get deleted together, along with one dispatch branch in each entry point

The same path runs everywhere. `litellm` (the main images and a local CLI run) and `python -m gateway.launch` (the gateway image) both call `supervisor.supervise`, and nothing else starts the Rust gateway in forwarding mode. In a container the supervisor is PID 1, so its signal and exit rules are deployment behavior and not CLI polish

## Boundary

A supervised Python server differs from Python alone in exactly two ways, and nothing else in the proxy may change because Rust is in front

- It listens on the private loopback listener the supervisor gives it, never on the public address
- It restores the client address Rust saw, because its own TCP peer is always Rust

`proxy_server.py`, routes, auth and middleware never know Rust is there. A change that needs one of them to branch on forwarding is in the wrong place

## Modules

- `server.py` describes what an entry point would serve with on the public port, as a `PythonServer`. The supervisor reads nothing else about the entry point
- `mode.py` is the Python child's side. `Supervised` is the whole contract the supervisor hands the child, and the `uvicorn_*` and `gunicorn_*` functions translate it into each server's own options
- `peer.py` restores the client address
- `supervisor/` starts, watches and stops both children. Its rules are in `supervisor/AGENTS.md`
- `platform/` holds every OS-specific operation. Its rules are in `platform/AGENTS.md`

## Entry points

Entry points (`litellm/proxy/proxy_cli.py`, `gateway/launch.py`, `litellm_cli/main.py`) only translate between their own flags and this package

- In the supervisor they describe what they would serve with as a `server.PythonServer` and hand it to `supervisor.supervise`, which decides whether forwarding can start and what Rust and the Python child receive. A refusal (no gateway binary, or a server or listener Rust cannot front) is reported and returned, and the entry point then serves from Python alone exactly as with `LITELLM_RUST` off
- In the supervised child they merge `mode.uvicorn_options` or `mode.uvicorn_flags` into their server settings, or hand `mode.private_listener` to `run_gunicorn_server`, and wrap the app with `peer.wrap` in their worker factory. These are identities when forwarding is off, so entry points never branch on it
- A server is started three ways here, through uvicorn keyword arguments, uvicorn's own CLI and gunicorn, and each one states the same two facts, the private listener and that the server itself trusts no peer's forwarding headers. The uvicorn translations are in `mode.py` and turn proxy headers off. `run_gunicorn_server` applies `server_listener.gunicorn_overrides` for the `PrivateListener` it is given, which sets gunicorn's `bind` and an empty `forwarded_allow_ips`. A new way to start a server gets its own translation, never a branch in the entry point

## Client address

Python must see each request as if the client had connected directly. `x-litellm-rust-peer` is the only Rust-specific input, and the proxy app itself never knows Rust is in front of it

- Rust sends the client's TCP peer in `x-litellm-rust-peer` and leaves every forwarding header untouched. It never appends itself to `X-Forwarded-For`, because the proxy parses that header on its own (`use_x_forwarded_for`, `mcp_xff_num_trusted_hops`) and an extra hop would shift what those settings resolve
- Server-level proxy headers are off in a supervised server, through `proxy_headers: False` for uvicorn, `--no-proxy-headers` for its CLI and an empty `forwarded_allow_ips` for gunicorn. The server's TCP peer is always loopback, which uvicorn and gunicorn trust by default, so leaving them on would let any client set its own address and scheme
- `peer.RustPeerMiddleware` puts the real peer back into the scope and then runs uvicorn's own `ProxyHeadersMiddleware` with the operator's trust. It wraps the app from outside, so it runs before anything that reads the client address
- The operator's trust is one string, `trusted_proxies`. It is resolved by the server library the entry point uses (`server.uvicorn_trusted_proxies` or gunicorn's config), never defaulted in this package. `server.NO_TRUSTED_PROXIES` stands for proxy headers the operator turned off, since a trust list that matches nobody behaves the same
- `peer.py` honors the peer header only from the loopback hop and always removes it. Any process on the host can connect to the private listener and send that header, so loopback is the trust boundary of the private listener. That widens what Python alone allows only when the operator's trust list leaves loopback out

## Environment contracts

Each environment contract is written and read in one place

- The Python child's contract is the `mode.Supervised` type itself. `mode.supervised_environment` serializes it into one variable and `mode.rust_forward_mode` validates it back, so a new field needs no parsing code
- `supervisor.children.rust_spec` writes the `LITELLM_FORWARD_*` variables that `gateway-forward`'s `settings.rs` reads. Python owns every value Rust needs, so a new one gets a new variable there and a matching field in `settings.rs` in the same change
- `mode.rust_forward_mode` reads `LITELLM_RUST` strictly, so an invalid value stops the CLI at startup with one line naming it. Per-call decisions in `litellm/rust_bridge/configuration.py` treat the same invalid value as unset, because they run inside SDK calls
