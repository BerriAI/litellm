# Gateway UI

Serves the exported dashboard under `/ui/`, with the `_next` asset aliases used by the Python proxy

The router takes an asset directory, an authentication backend, a session store, and the cookie security setting. `gateway` owns environment configuration and server startup. `gateway-auth` implements the local administrator backend and the `UiSession` extractor

Run from `litellm-rust` with an existing dashboard export and a gateway config:

```sh
export LITELLM_CONFIG=/path/to/config.yaml
export LITELLM_UI_PATH=/path/to/litellm/proxy/_experimental/out
export UI_USERNAME=admin
read -rs UI_PASSWORD
export UI_PASSWORD
cargo run -p litellm-gateway
```

Open `/ui/login/` to sign in. HTTPS cookies are enabled by default. For local HTTP development only, set `LITELLM_UI_SECURE_COOKIES=false`. Omitting `LITELLM_UI_PATH` leaves UI routes disabled. `UI_PASSWORD` is required when the UI is enabled, and never falls back to the inference master key

`POST /v2/login` accepts JSON credentials and returns the dashboard's token and redirect URL. `axum-login` handles authenticated sessions through `tower-sessions`. Login rotates the session, sets an absolute 24-hour expiry, and issues an HttpOnly `litellm_session` cookie. The JavaScript-readable `token` cookie contains display claims and a CSRF token in its `key` field. It contains neither the master key nor the session ID

`GET /session/info` and `POST /session/logout` require both the session cookie and `Authorization: Bearer <key>` from the dashboard token. The session's stored user determines authorization; client-supplied JWT claims do not. Logout deletes the server-side session and clears both cookies. JSON-only login prevents cross-origin form submissions, and login attempts use a shared token bucket with a burst of ten and a refill rate of ten per minute per gateway process

The gateway uses a process-local Moka session store with expiry and a capacity of 10,000 sessions. Restarts invalidate sessions, and capacity eviction can sign users out early. Multi-process deployments need a shared `SessionStore` supplied to the router

The current scope is local administrator login, sessions, discovery, and static assets. SSO, database-backed users, cross-origin workers, custom URL prefixes, and management APIs are not implemented. Dashboard screens that call management APIs still require those routes. UI request and response bodies are excluded from the inference gateway's debug body logger

Run the regression suite from `litellm-rust` without a browser, dashboard build, running proxy, or provider credentials:

```sh
cargo test -p litellm-gateway-ui -p litellm-gateway-auth -p litellm-gateway --locked -- --test-threads=1
```

`gateway-ui/tests/routes.rs` sends requests through the real Axum router with temporary asset files and an injected session store. It checks static assets and redirects, discovery, login responses and cookie attributes, credential failures and rate limits, JSON validation and body limits, CSRF enforcement, fixed expiry, password changes, session rotation, and logout revocation

`gateway-auth/tests/ui.rs` checks credential matching, empty configuration, user lookup, and password-dependent session hashes. `gateway/tests/server.rs` checks optional mounting, the production Moka store, credential isolation from inference, local HTTP cookies, and exclusion of login secrets from logs

The command runs serially because the existing gateway logging test can lose tracing events during concurrent tests

These tests cover the gateway's HTTP contract. They do not execute the dashboard's JavaScript or verify visual rendering
