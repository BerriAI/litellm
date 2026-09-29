# Configuration reference

CLI arguments take priority over their environment-variable equivalents. Authentication mode is selected in this order: service account, external OAuth token, then local OAuth. Team mode is explicitly enabled and cannot be combined with the single-identity external authentication modes.

## Commands

| Command | Purpose |
|---|---|
| `auth` | Run the local OAuth flow |
| `start` | Start the MCP server; this is the default command |
| `version` | Print the package version |
| `help` | Print CLI help |

## CLI flags

| Flag | Default | Description |
|---|---:|---|
| `--transport <stdio\|http>` | `stdio` | Select the MCP transport |
| `--port <number>` | `3100` | HTTP listen port |
| `--host <address>` | `127.0.0.1` | HTTP bind address |
| `--team` | off | Enable multi-user team mode; requires HTTP |
| `--issuer-url <url>` | — | Public HTTPS issuer URL required by team mode; HTTP is accepted only for localhost |
| `--no-resources[=<bool>]` | false | Disable `gdrive:///` resources while leaving tools enabled; an explicit false value re-enables resources |
| `--api-timeout=<ms>` | `120000` | Per-attempt timeout; `0` disables it |
| `--token-refresh-timeout=<ms>` | `15000` | Per-attempt timeout for an OAuth access-token refresh; `0` disables it |
| `--retry-max=<n>` | `3` | Maximum retry attempts on retryable errors (429, 503, 504, timeouts, and network failures); `0` disables retries |
| `--retry-base-delay=<ms>` | `1000` | Exponential-backoff base delay, capped at 30 seconds with jitter |

The timeout and retry settings apply to two paths; they are not applied to every Google API request:

- The retry-wrapped content insertion performed by `createGoogleDoc` uses `--api-timeout` per attempt with up to `--retry-max` retries. 500 and 502 are deliberately not retried: a 5xx raised after the request reached Google may mean a non-idempotent batch update was partially applied, and retrying could double-apply edits.
- OAuth access-token refreshes (local OAuth accounts, team members, and an external refresh token) use `--token-refresh-timeout` per attempt and are retried at most once, with `--retry-base-delay` backoff; `--retry-max=0` disables that retry too. A refresh that stalls through both attempts fails the current call with an explicit error instead of hanging every call on the server until Google answers. With the defaults, that is about 31 seconds at worst. A larger timeout suits a slow proxy; `0` restores the previous unbounded wait.

Flags are read from the server's own command line. For a client that launches the server over stdio, append them to the `args` array after the package name:

```json
{
  "mcpServers": {
    "google-drive": {
      "command": "npx",
      "args": ["-y", "@piotr-agier/google-drive-mcp", "--api-timeout=180000", "--retry-max=5"]
    }
  }
}
```

The environment-variable equivalents in [Resources, timeout, and retry](#resources-timeout-and-retry) are an alternative for clients that pass an `env` block instead of extra arguments.

## Credentials and local OAuth

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_DRIVE_OAUTH_CREDENTIALS` | config-directory file | Absolute or relative path to the OAuth credentials JSON |
| `GOOGLE_DRIVE_MCP_TOKEN_PATH` | `$XDG_CONFIG_HOME/google-drive-mcp/tokens.json` | Override the local token store |
| `GOOGLE_DRIVE_MCP_AUTH_PORT` | `3000` | First of five consecutive loopback callback ports |
| `GOOGLE_DRIVE_MCP_SCOPES` | full configured set | Comma-separated scope aliases or full HTTPS scope URLs |
| `XDG_CONFIG_HOME` | `~/.config` | Base directory used for credentials, tokens, and the default team store |

Credentials lookup order is:

1. `GOOGLE_DRIVE_OAUTH_CREDENTIALS`.
2. `$XDG_CONFIG_HOME/google-drive-mcp/gcp-oauth.keys.json`.
3. `gcp-oauth.keys.json` in the package/project root as a legacy fallback.

Token lookup and storage use `GOOGLE_DRIVE_MCP_TOKEN_PATH` first, then the XDG config location.

Supported scope aliases are `drive`, `drive.file`, `drive.readonly`, `documents`, `spreadsheets`, `presentations`, `calendar`, and `calendar.events`. Changing scopes normally requires re-authentication.

## Resources, timeout, and retry

| Variable | Default | Description |
|---|---:|---|
| `GOOGLE_DRIVE_MCP_DISABLE_RESOURCES` | false | Disable MCP resources; accepts `1/0`, `true/false`, `yes/no`, or `on/off` |
| `GOOGLE_DRIVE_MCP_API_TIMEOUT` | `120000` | Fallback for `--api-timeout` |
| `GOOGLE_DRIVE_MCP_TOKEN_REFRESH_TIMEOUT` | `15000` | Fallback for `--token-refresh-timeout` |
| `GOOGLE_DRIVE_MCP_RETRY_MAX` | `3` | Fallback for `--retry-max` |
| `GOOGLE_DRIVE_MCP_RETRY_BASE_DELAY` | `1000` | Fallback for `--retry-base-delay` |

## HTTP transport

| Variable | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | `stdio` or `http` |
| `MCP_HTTP_PORT` | `3100` | HTTP listen port |
| `MCP_HTTP_HOST` | `127.0.0.1` | HTTP bind address |
| `MCP_HTTP_ALLOWED_HOSTS` | issuer hostname in team mode | Additional comma-separated allowed `Host` values |

## Team mode

| Variable | Default | Description |
|---|---|---|
| `MCP_TEAM_MODE` | off | Enable team mode; requires HTTP |
| `MCP_TEAM_ISSUER_URL` | — | Public server URL; equivalent to `--issuer-url` |
| `MCP_TEAM_ALLOWED_DOMAINS` | any Google account | Comma-separated Workspace domains allowed to sign in |
| `MCP_TEAM_ALLOWED_REDIRECT_URIS` | open | Comma-separated allowlist for dynamically registered client redirects |
| `MCP_TEAM_TOKEN_TTL` | `3600` | Access-token lifetime in seconds, from 60 through 86400 |
| `MCP_TEAM_STORE` | `file` | `file` or `memory` |
| `MCP_TEAM_STORE_PATH` | config-directory `team-store.json` | Persistent team-store path |
| `MCP_TRUST_PROXY` | unset | Number of trusted reverse-proxy hops |

## Service account mode

| Variable | Description |
|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to a service-account JSON key; activates service-account mode |
| `GOOGLE_DRIVE_MCP_SUBJECT` | Optional Workspace user to impersonate through domain-wide delegation |

## External OAuth token mode

| Variable | Description |
|---|---|
| `GOOGLE_DRIVE_MCP_ACCESS_TOKEN` | Pre-obtained access token; activates external-token mode |
| `GOOGLE_DRIVE_MCP_REFRESH_TOKEN` | Optional refresh token |
| `GOOGLE_DRIVE_MCP_CLIENT_ID` | Required with a refresh token |
| `GOOGLE_DRIVE_MCP_CLIENT_SECRET` | Required with a refresh token |

## Deprecated variables

| Variable | Replacement |
|---|---|
| `GOOGLE_TOKEN_PATH` | `GOOGLE_DRIVE_MCP_TOKEN_PATH` |
| `GOOGLE_CLIENT_SECRET_PATH` | `GOOGLE_DRIVE_OAUTH_CREDENTIALS` |

See [Authentication](authentication.md) for identity behavior and [Deployment](deployment.md) for HTTP, Docker, and team-mode guidance.
