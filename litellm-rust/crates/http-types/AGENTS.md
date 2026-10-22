## Scope

- This crate is the HTTP client settings as the user writes them, before anything decides what a client can do with them. `HttpSettingsLayer` is one source's values, where `None` means that source did not set the key. `HttpSettings::from_layers` merges layers highest precedence first and fills defaults. `HttpSettings` is the merged result
- It knows the key names the user writes (`SSL_VERIFY`, `AIOHTTP_TRUST_ENV`, `HTTP_PROXY`, ...) and how Python reads them (`SslVerify::parse` is bool-or-path, lowercase proxy names win, `HTTP_PROXY` is ignored under CGI). It knows nothing about reqwest, rustls or which cipher a security level selects. That is `../http`, through `Resolution::from(&HttpSettings)`
- No I/O. Every environment read goes through an injected `Lookup`, and `without_missing_files` takes the existence check as a closure
- Depends on `litellm-core-utils` (`Lookup`, `Layer`, `merge`) and `veil` only. It must stay buildable without reqwest, so a settings producer can emit a layer without linking a TLS stack

## Producers and the consumer

- A host produces one `HttpSettingsLayer` per source it trusts and lets `from_layers` decide. It never pre-merges two sources into one layer
- `config-load` produces the gateway's layers from config.yaml and the environment, `python-bridge` produces the SDK's from the Python settings snapshot. `../http` is the only consumer. Anything that wants a client goes through `HttpClientConfig`, not these types
- Defaults live in `HttpSettings::default` and are applied only in `from_layers`. A layer never carries a default
- Python flags that only exist to be combined (`aiohttp_trust_env`, `disable_aiohttp_trust_env`, `disable_aiohttp_transport`) stay as the user's flags on the layer. `from_layers` collapses them into `trust_proxy_env` once, so no consumer re-derives it

## Redaction

- Proxy URLs can embed credentials, so `EnvironmentProxies` redacts `all`, `http` and `https` and exposes them only through accessors. Keep `Debug` derived through `veil`, never hand-written

## Tests

- Assert the merged `HttpSettings` for a given stack of layers, and the parsed layer for a given injected environment, never the shape of the helpers
