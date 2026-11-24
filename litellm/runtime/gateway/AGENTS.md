# Gateway image entry point

`python -m gateway.launch` is how the gateway image starts. `launch.py` hands its arguments to uvicorn's own CLI and serves `main.py`, which trims the proxy app to the LLM data plane

- Importing `launch.py` stays cheap. Every uvicorn worker imports it again before it can answer its first health ping, so LiteLLM imports wait until `main` or `worker_app` runs
- PgBouncer starts exactly once per pod, before the workers start, and the workers inherit the loopback URL through `DATABASE_URL`. `configure_database` is the only place that does this

## Rust forwarding

`launch.py` is an entry point of Rust forwarding, and every forwarding decision lives in `litellm/proxy/rust_forward/`, whose `AGENTS.md` states the contract

- `main` dispatches on `rust_forward_mode` and does nothing else about forwarding. With forwarding off or refused it serves exactly as before
- `python_server` reads the flags and environment through uvicorn's own CLI parser, so Rust gets the values uvicorn would run with. Never reparse a uvicorn flag by hand
- Under forwarding the supervisor process runs `configure_database`, as `before_start`, once forwarding is known to be possible. The supervised child never runs it and inherits the exported URL
- The supervised child appends `mode.uvicorn_flags` after the operator's flags, so the private listener and `--no-proxy-headers` win
- `worker_app` wraps the app with `peer.wrap`, which is the identity when forwarding is off
