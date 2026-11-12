# Proxy

## Rust forwarding

`proxy_server.py`, routes, auth and middleware never know whether Rust is in front of the proxy, and none of them may branch on it. `proxy_cli.py` only describes what it would serve with as a `rust_forward.server.PythonServer`, calls `rust_forward.supervisor.supervise`, and merges the options `rust_forward.mode` returns, which are identities when forwarding is off. Every forwarding decision lives in `rust_forward/`, and `rust_forward/AGENTS.md` states the contract
