# Opt-in Rust gateway

The Rust gateway is an optional public HTTP listener. The existing Python gateway remains the default. Set
`LITELLM_RUST_GATEWAY=true` to start the Rust gateway on the public listener and supervise the existing Python
gateway on loopback port 4002. Set the flag to exactly `false`, or leave it unset, to keep the existing startup
path. No separate enable flag is required for the Python upstream.

The Rust binary defaults to `/usr/local/bin/litellm-gateway`; local deployments can set
`LITELLM_RUST_GATEWAY_BINARY` to an executable binary. `LITELLM_RUST_GATEWAY_PYTHON_PORT` changes the private
Python port. The private listener must remain on loopback and must not be added to a container or Kubernetes
Service port.

```sh
docker run --stop-timeout 30 -p 4000:4000 \
  -e LITELLM_RUST_GATEWAY=true \
  -e LITELLM_RUST_GATEWAY_TRUSTED_PROXIES=10.0.0.0/8 \
  -e LITELLM_RUST_GATEWAY_TERMINATION_GRACE_SECONDS=30 \
  -e LITELLM_GATEWAY_METRICS_PORT=4001 \
  -e CONFIG_FILE_PATH=/app/config/config.yaml \
  litellm/litellm:main
```

The supervisor waits up to 120 seconds for startup, drains Rust for up to 15 seconds, then gives Python 5 seconds
to stop. It also allows a 10-second cleanup margin. `LITELLM_RUST_GATEWAY_STARTUP_TIMEOUT_SECONDS`,
`LITELLM_RUST_GATEWAY_DRAIN_TIMEOUT_SECONDS`, and `LITELLM_RUST_GATEWAY_PYTHON_GRACE_SECONDS` configure those
timeouts. `LITELLM_RUST_GATEWAY_TERMINATION_GRACE_SECONDS` must be at least the drain timeout plus Python grace
plus the cleanup margin. Container orchestrators must grant at least this value before sending SIGKILL. For
Docker, set `--stop-timeout` to the same or a larger number; Docker's default stop timeout is shorter.

The trusted-proxy setting is an optional comma-separated list of IP addresses or CIDRs. An unset or empty list
trusts no external proxy. Wildcard trust is rejected. The Rust gateway canonicalizes forwarded client and scheme
headers using the configured trust list. It preserves the original `Host` and `Origin` headers. Configure only
proxy addresses that are controlled by your deployment.

The Rust gateway does not support TLS termination or Uvicorn TLS settings. Terminate TLS at the load balancer or
another trusted edge, then forward HTTP to the gateway. Uvicorn reload, Unix-domain and file-descriptor listeners,
environment-file overrides, and explicit `--no-proxy-headers` are rejected in enabled mode.

The Helm chart keeps the Service on port 4000. Configure `LITELLM_RUST_GATEWAY` through the gateway container's
environment or `envFrom`, and set `gateway.terminationGracePeriodSeconds` to match the supervisor grace setting.
When the metrics sidecar is enabled, the chart exports its reserved port to the supervisor even when the feature
flag comes from `envFrom`, so port collisions are rejected during enabled startup.

```yaml
gateway:
  envConfigMaps:
    - gateway-runtime
  metricsServer:
    enabled: true
    port: 4001
  terminationGracePeriodSeconds: 30
```

The `gateway-runtime` ConfigMap can contain `LITELLM_RUST_GATEWAY: "true"` and
`LITELLM_RUST_GATEWAY_TRUSTED_PROXIES: "10.0.0.0/8"`. The chart does not add a private Python port to the pod
ports or Service.

To roll back, set `LITELLM_RUST_GATEWAY=false` in the same environment source and redeploy, or roll back the
Deployment revision. The feature flag is strict and disabled by default, so removing the opt-in also restores the
existing Python startup path.
