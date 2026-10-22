## Scope

- This crate is the one place a `reqwest` client is built. `clippy.toml` disallows `reqwest::Client::new`, `Client::builder` and the TLS builder methods everywhere else, so every outbound connection honors the host's TLS, proxy, keepalive and timeout settings by construction
- Input is `HttpClientConfig`, output is a pooled `Client`. `Resolution::from(&HttpSettings)` is the only way to turn user settings into an `HttpClientConfig`, and it is where names the user wrote (`ssl_ecdh_curve`, `ssl_security_level`) meet what rustls supports. A value rustls cannot honor keeps the default and is reported in `Resolution::unsupported`, never silently dropped and never an error
- Settings as the user writes them (`HttpSettings`, `HttpSettingsLayer`, `SslVerify`, `TcpKeepalive`, `EnvironmentProxies`) live in `../http-types`, which links neither reqwest nor rustls. Crates that produce settings depend on that crate, not on this one
- Route-neutral and Rust-only. Nothing here names a route, a provider, an API format, a credential or a config.yaml key. Provider credentials, OAuth, beta headers and model behavior are decided above and arrive as plain headers
- `gateway` and `python-bridge` are both hosts of this crate. Each lowers its own settings source into an `HttpSettingsLayer` and takes clients from one process-wide `HttpClientPool`

## What lives here

- `pool.rs`: `HttpClientPool`, keyed by `(HttpClientConfig, ClientVariant)` with a TTL. `ClientVariant` is the full list of ways a client may differ from the configured one (redirect policy, pinned DNS, dropped client certificate and proxies). Add a variant only for a difference that must be fixed when the client is built, otherwise set it on the request
- `config.rs` and `tls.rs`: `HttpClientConfig` to `reqwest::ClientBuilder` and to `rustls::ClientConfig`
- `proxy.rs`: `EnvironmentProxies` to reqwest proxies, and the URL matcher the media fetcher uses to decide whether a URL leaves through the proxy
- `media.rs`: the SSRF-safe `MediaFetcher`. It is here because address pinning is a build-time property of the client (`dns_resolver2`) and because it needs the proxy matcher. `UrlPolicy` and `DownloadPolicy` are the host's inputs, the host never decides a URL is safe on its own
- `request.rs`: case-insensitive header helpers over `Vec<(String, String)>` and `HeaderPolicy` for forwarding. They keep existing precedence, duplicate handling and unrelated forwarded headers. `string_headers` and `HeaderError` validate the shape of a caller's `extra_headers` and are the only place a request field is named
- `outbound.rs`: `OutboundRequest` serializes the body once and a `RequestSigner` sees those exact bytes. A route builds it after its hooks ran and cannot change it afterwards
- `transport.rs`: `Error::{Http, Network, Connect}` is the one transport error every crate above nests. `Connect` means the request never reached the provider, so a caller may reroute. Messages never carry URLs
- `websocket.rs` and `mcp.rs`: the two connections the reqwest 0.12 builder cannot make. `mcp.rs` repeats the builder chain for the reqwest 0.13 that rmcp needs, delete it when the two versions converge

## Boundary with consumers

- Consumers take `litellm_http::Client`, which derefs to `reqwest::Client`, and use reqwest's request and response types directly. This crate owns how a client is built and configured, it is not a wrapper over reqwest's request API
- A consumer builds its request from the `Client` it was given. It never reads the environment, a config file or a settings snapshot to decide TLS, proxy or timeouts
- Tests take `Client::for_test`, `plain_for_test` or `no_redirect_for_test` behind the `test-support` feature. Production code never calls them

## Known gaps

- `websocket.rs` builds its own rustls config from native roots and ignores `HttpClientConfig` (`ssl_verify`, CA bundle, client certificate, proxies). Make it take an `HttpClientConfig` like the pool does before adding another WebSocket caller
- `HttpSettings::connect_timeout` has no layer field, so it is always the default

## Tests

- Verify observable request behavior (which headers were sent, which proxy was picked, whether a client was reused), never helper structure
- TLS tests generate certificates with `rcgen` and run a local listener. They never depend on a public host

# references

- https://github.com/BerriAI/litellm-docs/blob/main/docs/guides/security_settings.md owns the user-facing meaning of `ssl_verify`, `ssl_certificate`, `ssl_security_level` and `ssl_ecdh_curve`
