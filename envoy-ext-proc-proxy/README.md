# ext_proc_proxy - HTTPS/HTTP Proxy & Envoy ext_proc Server

An asynchronous service in Python featuring:
1. **HTTPS-terminating HTTP/HTTPS Proxy** using [`aiohttp`](https://docs.aiohttp.org/).
2. **Envoy External Processor (`ext_proc`) Server** using [`grpc.aio`](https://grpc.github.io/grpc/python/grpc_asyncio.html) with TLS protection and `aiohttp`.

---

## Features

### 1. HTTPS Proxy
- **HTTPS Termination**: Accepts incoming client connections over TLS/HTTPS on a configurable port.
- **Protocol Routing via `X-Forwarded-Proto`**:
  - `X-Forwarded-Proto: http` -> forwards outgoing request via **HTTP**.
  - `X-Forwarded-Proto: https` -> forwards outgoing request via **HTTPS**.
  - No `X-Forwarded-Proto` header -> defaults to **HTTPS**.
- **Client Keep-Alive**: Supports persistent HTTP/1.1 keep-alive connections on the client side.
- **Full Streaming**: Asynchronously streams request payloads and response bodies without buffering in memory.
- **Upstream TLS Verification**: Verifies target server TLS certificates when connecting over HTTPS.
- **Certificate Options**:
  - Load server certificate and private key from PEM files (`--cert` and `--key`).
  - Automatically generate a self-signed certificate on the fly for testing (`--self-signed`).
- **Clean Header Management**: Strips hop-by-hop headers and prevents propagation of `X-Forwarded-For` or `X-Forwarded-Host`.

### 2. Envoy ext_proc gRPC Server
- **TLS Protection**: The gRPC ext_proc server listens on a secure TLS port using the same certificate and private key as the HTTPS proxy.
- **Bidirectional gRPC Streaming**: Implements `envoy.service.ext_proc.v3.ExternalProcessor` using `grpc.aio`.
- **Target URL Dispatching**: Forwards incoming requests to the preconfigured backend HTTP/HTTPS server (`--ext-proc-target`).
- **End-to-End Streaming**: Streams request body chunks incrementally to the target HTTP server without buffering, and streams the target server's response headers and body chunks back to Envoy via `StreamedImmediateResponse`.
- **Error Handling**: Converts upstream connection failures and timeouts to `502 Bad Gateway` / `504 Gateway Timeout` responses with descriptive error details.
- **Observability Mode**: Respects Envoy `observability_mode` configuration.

---

## Installation & Requirements

- Python 3.9+ (Python 3.14 compatible)
- OpenSSL (for self-signed certificate generation)
- `aiohttp`, `grpcio`, `grpcio-tools`

Dependencies are installed in the local virtual environment `.venv`. Protobuf stubs are generated into `.gen/`.

---

## Usage

### Prep
```bash
uv sync
```

### Run Both Servers (Testing with Self-Signed Certificate)

```bash
uv run ext-proc-proxy \
  --self-signed \
  --port 8443 \
  --ext-proc-port 50051 \
  --ext-proc-target http://127.0.0.1:8080
```

### Run with Custom Certificate and Key

```bash
uv run ext-proc-proxy \
  --cert /path/to/cert.pem \
  --key /path/to/key.pem \
  --port 8443 \
  --ext-proc-port 50051 \
  --ext-proc-target https://backend.internal:8443
```

### Command-Line Arguments

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Host/IP address to bind the HTTPS proxy server to |
| `-p`, `--port` | `8443` | Port to listen for incoming HTTPS proxy connections |
| `--cert` | `None` | Path to server certificate PEM file |
| `--key` | `None` | Path to server private key PEM file |
| `--self-signed` | `False` | Generate self-signed certificate for testing |
| `--keepalive-timeout` | `75.0` | Keep-alive timeout for client connections in seconds |
| `--upstream-timeout` | `60.0` | Timeout for upstream requests in seconds |
| `--log-level` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--ext-proc-host` | `0.0.0.0` | Host/IP address to bind the Envoy ext_proc gRPC server to |
| `--ext-proc-port` | `50051` | Port to listen for Envoy ext_proc gRPC connections (TLS) |
| `--ext-proc-target` | `http://127.0.0.1:8080` | Target URL for the ext_proc HTTP server |

---

## Testing

Run all unit and integration tests:

```bash
uv run python -m unittest
```
