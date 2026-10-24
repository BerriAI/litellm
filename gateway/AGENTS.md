# Gateway

## Rust gateway

- `LITELLM_RUST_GATEWAY=true` enables the public Rust listener. Unset or exact `false` preserves Python-only startup
- Keep `LITELLM_RUST` independent. It controls the in-process Rust bridge
- Rust forwards every request to Python, which owns authentication, routing, policy and provider calls. Do not add native request selection or database initialization here
- `launch.py` owns startup and pooling. `supervisor.py` coordinates container processes, not application requests. A Python or Rust child exit fails the container rather than restarting that child
- Start Python on loopback `127.0.0.1:4002` before Rust on public port 4000. Reserve 4001 for metrics and never expose the private Python port in container or Service ports
- With a writer `DATABASE_URL`, start one managed PgBouncer even when `LITELLM_PGBOUNCER_ENABLED=false`. Both children receive the same pooled URL and marker, and Python must not start another pooler
- Keep database-free mode database-free. Read replicas remain outside the shared writer connection cap
- Drain Rust, stop Python, then stop PgBouncer and reap remaining children within the shutdown deadline. Defaults allow 15 seconds for Rust, 5 for Python and a 10-second cleanup margin
- Docker stop timeout and Helm termination grace must cover `LITELLM_RUST_GATEWAY_TERMINATION_GRACE_SECONDS`, which defaults to 30
- Configure `LITELLM_RUST_GATEWAY_TRUSTED_PROXIES` with explicit ingress IPs or CIDRs. Empty trusts no external proxy, wildcard trust is rejected, and `Host` and `Origin` are preserved
- Terminate TLS at the edge. Enabled mode rejects Uvicorn TLS, reload, Unix-socket and file-descriptor listeners, env-file overrides and disabled proxy headers
- The image packages `/usr/local/bin/litellm-gateway`. Local runs may override `LITELLM_RUST_GATEWAY_BINARY` and the private port through `LITELLM_RUST_GATEWAY_PYTHON_PORT`
- Roll back by removing the Rust gateway opt-in or setting it to `false` and redeploying
