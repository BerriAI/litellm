# Python callback and guardrail extension host (POC)

This process runs customer Python callbacks and guardrails outside the LiteLLM proxy or Rust gateway. It is a POC and is disabled unless an extension-host endpoint is configured.

## Why RPC instead of PyO3

PyO3 embeds CPython in the Rust process. That is useful for LiteLLM's temporary config reader, but it would keep customer code, the GIL, Python crashes, imports, and mutable plugin state inside the gateway. It would also give the Python proxy and Rust gateway different plugin boundaries.

The gRPC boundary gives both gateways the same versioned API and the same sidecar executable. Customer modules are imported only by this host. The Rust POC still uses PyO3 once at startup to read the existing YAML configuration; Stage 3 can replace that reader without changing this protocol.

## Run the host

Install LiteLLM with its gRPC dependencies, set a shared token, and bind to loopback:

```shell
export LITELLM_EXTENSION_HOST_TOKEN='replace-me'
litellm-python-extension-host --listen 127.0.0.1:50051
```

Enable the Python proxy without changing callback or guardrail entries:

```yaml
general_settings:
  python_extension_host:
    endpoint: http://127.0.0.1:50051
    token: os.environ/LITELLM_EXTENSION_HOST_TOKEN
    connect_timeout_seconds: 5
    hook_timeout_seconds: 30

litellm_settings:
  callbacks:
    - customer_plugins.logging.proxy_handler_instance
  success_callback:
    - customer_plugins.logging.on_success
  failure_callback:
    - customer_plugins.logging.on_failure

guardrails:
  - guardrail_name: customer-policy
    litellm_params:
      guardrail: customer_plugins.guardrails.CustomerGuardrail
      mode: pre_call
```

For invocation-scoped cache access, also configure a reverse service and point the host at it:

```yaml
general_settings:
  python_extension_host:
    endpoint: http://127.0.0.1:50051
    token: os.environ/LITELLM_EXTENSION_HOST_TOKEN
    gateway_listen: http://127.0.0.1:50052
```

```shell
export LITELLM_GATEWAY_SERVICES_ENDPOINT=http://127.0.0.1:50052
litellm-python-extension-host --listen 127.0.0.1:50051
```

The host receives an opaque `CacheRef`, not `DualCache`, and exposes only `async_get_cache` and `async_set_cache`. The proxy revokes the reference after the invocation.

## Rust gateway

Build with `python-config`, point `LITELLM_CONFIG_PATH` at the existing proxy YAML, and set:

```shell
export LITELLM_PYTHON_EXTENSION_HOST_ENDPOINT=http://127.0.0.1:50051
export LITELLM_PYTHON_EXTENSION_HOST_TOKEN='replace-me'
```

The gateway creates one long-lived tonic channel, activates the same manifest, installs remote guardrail/logger adapters, and retains the client in `AppState`. With no endpoint, it creates no channel and performs no extension RPCs.

## Runtime contract

Startup is `GetCapabilities` → `PrepareRevision` → `CommitRevision`. Pre-call, during-call, and post-call guardrails are awaited. Terminal callback events use a bounded batch queue. Streaming uses one bidirectional RPC and bounded channels for backpressure; cancellation drops the RPC and upstream producer. After a transient disconnect, the client fails open and re-prepares its retained manifest.

V1 accepts importable or mounted local modules and supports:

- `async_pre_call_hook`
- `async_moderation_hook`
- `async_post_call_success_hook`
- `async_log_success_event`
- `async_log_failure_event`
- `async_log_stream_event`
- `async_post_call_streaming_hook`
- `async_post_call_streaming_iterator_hook`
- configured success/failure callback functions

A plugin that overrides another hook is rejected as a whole. S3/GCS loading and arbitrary Python-object transport are not supported.

## Failure and security limits

Transport failures and timeouts fail open: guardrails pass through the original value, streams pass through original chunks, and callback events may be dropped. Intentional plugin blocks remain blocks. This policy is not suitable for mandatory compliance guardrails; fail behavior must become configurable per plugin before GA.

The token is sent as `x-litellm-extension-token`. Plaintext API keys are never included in protobuf messages. TCP plaintext is for loopback or trusted private networks only; UDS and mTLS are follow-ups.

## Protocol generation

The canonical contract is `proto/litellm/python_extension/v1/extension_host.proto`. Python bindings and stubs are checked in under `litellm/python_extension/generated/v1`. Rust bindings are generated into Cargo `OUT_DIR` by `litellm-python-extension-protocol/build.rs` and are never checked into the source tree.
