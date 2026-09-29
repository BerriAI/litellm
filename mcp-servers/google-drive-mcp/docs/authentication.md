# Authentication

The server selects authentication modes in this order: service account, external OAuth token, then local browser-based OAuth. Team mode is a separate multi-user HTTP deployment described in [Deployment](deployment.md#team-mode-multi-user-http-deployments).

## Local OAuth configuration

### OAuth Credentials Configuration

The server supports multiple methods for providing OAuth credentials (in order of priority):

#### 1. **Environment Variable** (Highest Priority)
```bash
export GOOGLE_DRIVE_OAUTH_CREDENTIALS="/path/to/your/gcp-oauth.keys.json"
```

#### 2. **Config Directory** (Recommended)
Place `gcp-oauth.keys.json` in the XDG config directory:
```
~/.config/google-drive-mcp/gcp-oauth.keys.json
```
This is the recommended location — it works reliably with `npx`, global installs, and local setups.

#### 3. **Project Root** (Legacy Fallback)
Place `gcp-oauth.keys.json` in the project root directory. This still works for local development but is unreliable with `npx` or global installs.

### OAuth Scope Configuration

By default, the server requests a broad scope set for Drive/Docs/Sheets/Slides/Calendar.
You can override requested scopes with:

```bash
export GOOGLE_DRIVE_MCP_SCOPES="drive.readonly,documents,spreadsheets"
```

Notes:
- Comma-separated list.
- Values can be full scope URLs or short aliases:
  `drive`, `drive.file`, `drive.readonly`, `documents`, `spreadsheets`, `presentations`, `calendar`, `calendar.events`.
- Changing scopes usually requires re-authentication.

### Auth Server Port Configuration

During OAuth authentication, a local HTTP server is started to receive the callback. By default it tries ports 3000–3004. If those conflict with other services (e.g., a dev server), you can change the starting port:

```bash
export GOOGLE_DRIVE_MCP_AUTH_PORT=3100
```

The server will try 5 consecutive ports starting from the configured value (e.g., 3100–3104).

The callback server binds to the loopback interface and the OAuth redirect URI uses the loopback IP — `http://127.0.0.1:<port>/oauth2callback` (default range `127.0.0.1:3000`–`127.0.0.1:3004`). **Desktop app** OAuth clients (the recommended type — see [Create OAuth 2.0 credentials](setup.md#4-create-oauth-20-credentials)) accept any loopback redirect automatically and need no action. If you instead use a **Web application** OAuth client, you must register `http://127.0.0.1:<port>/oauth2callback` for every port in the range as an authorized redirect URI in Google Cloud Console, or authentication fails with `redirect_uri_mismatch`.

### Token Storage

Authentication tokens are stored securely following the XDG Base Directory specification:

| Priority | Location | Configuration |
|----------|----------|---------------|
| 1 | Custom path | Set `GOOGLE_DRIVE_MCP_TOKEN_PATH` environment variable |
| 2 | XDG Config | `$XDG_CONFIG_HOME/google-drive-mcp/tokens.json` |
| 3 | Default | `~/.config/google-drive-mcp/tokens.json` |

**Token file format (v2):** `tokens.json` uses a versioned schema that holds all connected accounts keyed by alias, plus the global default. A `tokens.json` from versions before 2.3 is auto-migrated on first boot and a `tokens.json.v1-backup-<timestamp>` is written alongside in case you need to roll back. No user action is required.

**Security Notes:**
- Tokens are created with secure permissions (0600)
- Each token-file write is an atomic rename; concurrent refreshes from different accounts serialize through an in-process queue
- Never commit tokens to version control
- Tokens auto-refresh before expiration
- Each refresh request is bounded (15 seconds per attempt, one retry, tunable with `--token-refresh-timeout`), so a stalled Google token endpoint fails the current call with a clear error instead of hanging every call on the server
- Google OAuth apps in "Testing" status have refresh tokens that expire after 7 days (Google's policy)

## External Authentication

For hosted, containerized, or CI/CD deployments where a browser-based OAuth flow is not available, the server supports two alternative authentication modes. They are checked in priority order before falling back to the default local OAuth flow.

### 1. Service Account Mode

Set the standard `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to a service account JSON key file. Best for server-to-server, CI/CD, and container deployments.

```json
{
  "mcpServers": {
    "google-drive": {
      "command": "npx",
      "args": ["@piotr-agier/google-drive-mcp"],
      "env": {
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/service-account-key.json"
      }
    }
  }
}
```

**Note:** The service account must have access to the Google Drive files/folders you want to work with. For Shared Drives, grant the service account's email address the appropriate permissions.

#### Domain-Wide Delegation (impersonating a user)

By default a service account acts as itself. Some Google APIs (for example Drive reads scoped to a user's "My Drive", or Calendar ACL writes against a personal calendar) require acting as a real Workspace user. Set `GOOGLE_DRIVE_MCP_SUBJECT` to the email of the user to impersonate:

```json
{
  "mcpServers": {
    "google-drive": {
      "command": "npx",
      "args": ["@piotr-agier/google-drive-mcp"],
      "env": {
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/service-account-key.json",
        "GOOGLE_DRIVE_MCP_SUBJECT": "user@your-domain.com"
      }
    }
  }
}
```

**Prerequisite:** a Workspace admin must authorize the service account's client ID for the requested scopes under **Admin console > Security > API controls > Manage Domain-wide Delegation**. The scopes granted there must cover the scopes the server requests (see [OAuth Scope Configuration](#oauth-scope-configuration)).

`GOOGLE_DRIVE_MCP_SCOPES` applies in service-account mode too, so you can narrow the JWT to a subset of the delegated scopes.

### 2. External OAuth Token Mode

Provide a pre-obtained OAuth access token via `GOOGLE_DRIVE_MCP_ACCESS_TOKEN`. This is useful when an external service handles the OAuth flow (e.g., a web app that obtains tokens on behalf of the user).

**Access token only** (no auto-refresh — token will eventually expire):
```json
{
  "mcpServers": {
    "google-drive": {
      "command": "npx",
      "args": ["@piotr-agier/google-drive-mcp"],
      "env": {
        "GOOGLE_DRIVE_MCP_ACCESS_TOKEN": "ya29.a0AfH6SM..."
      }
    }
  }
}
```

**With refresh token** (recommended — enables automatic token refresh):
```json
{
  "mcpServers": {
    "google-drive": {
      "command": "npx",
      "args": ["@piotr-agier/google-drive-mcp"],
      "env": {
        "GOOGLE_DRIVE_MCP_ACCESS_TOKEN": "ya29.a0AfH6SM...",
        "GOOGLE_DRIVE_MCP_REFRESH_TOKEN": "1//0dx...",
        "GOOGLE_DRIVE_MCP_CLIENT_ID": "123456789.apps.googleusercontent.com",
        "GOOGLE_DRIVE_MCP_CLIENT_SECRET": "GOCSPX-..."
      }
    }
  }
}
```

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_DRIVE_MCP_ACCESS_TOKEN` | Yes (activates mode) | Google OAuth access token |
| `GOOGLE_DRIVE_MCP_REFRESH_TOKEN` | No | Refresh token for auto-refresh |
| `GOOGLE_DRIVE_MCP_CLIENT_ID` | Required with refresh token | OAuth client ID |
| `GOOGLE_DRIVE_MCP_CLIENT_SECRET` | Required with refresh token | OAuth client secret |

### 3. Local OAuth Flow (Default)

If neither of the above modes is configured, the server uses the existing browser-based OAuth flow. In local OAuth mode the server supports multiple connected accounts — see **[Multi-Account Support](#multi-account-support)** below. Service-account and external-token modes are single-identity by design and do not support the multi-account tools.

## Multi-Account Support

The server can hold credentials for multiple Google accounts simultaneously — for example a personal Gmail plus a Google Workspace account — and route each tool call to the right one. This is available in the default **local OAuth mode**; service-account and external-token modes remain single-identity.

### The `manage_accounts` tool

One admin tool drives the whole lifecycle. It ignores the `account` parameter and is always available regardless of tool filtering.

| Action | `account_id` | What it does |
|---|---|---|
| `list` | — | Returns all connected accounts with alias, email, `sub`, scopes, expiry, and which is the default. Never returns tokens. |
| `add` | required (alias) | Starts an OAuth flow in your browser with `prompt=consent select_account` and `access_type=offline`, so Google shows an account picker and always returns a refresh token. On success, the new record is written to `tokens.json`. If it's the first account, it also becomes the default. Re-running `add` for an existing alias re-consents that account in place (to broaden scopes or recover a revoked grant); the refreshed grant takes effect on the next call, with no server restart. |
| `remove` | required (alias) | Deletes the account's credentials from `tokens.json` and clears it from the default if applicable. The token is **not** revoked server-side — see [Revoking OAuth access](#revoking-oauth-access). |
| `set_default` | required (alias, or `"null"` to clear) | Picks which account is used when a tool call omits `account`. |

**Alias rules:** lowercase alphanumerics with hyphens or underscores, 1–32 characters, starting with a letter or digit. Reserved names (`default`, `all`, `*`, `stdio`, `service-account`, `external-token`, `test`) are rejected.

### Per-tool account selection

Every non-admin tool accepts an optional `account` parameter whose value is one of the connected aliases. When omitted, the server resolves the target in this order:

1. The explicit `account` parameter on the call.
2. The global default set via `manage_accounts set_default`.
3. If exactly one connected account can satisfy the call's scope requirements, it is selected automatically.

**Writes refuse ambiguity.** If two accounts both satisfy a write, the server errors out with the list of eligible aliases and a pointer at `manage_accounts set_default`. Be explicit or pick a default. Reads on an ambiguous call currently require the same explicit choice; cross-account read fanout is planned for a future release.

### Typical flow

```
# In the MCP client:
Use manage_accounts to add my personal and work Google accounts.

# The assistant can now call:
manage_accounts(action="add", account_id="personal")   # browser flow
manage_accounts(action="add", account_id="work")       # browser flow
manage_accounts(action="set_default", account_id="work")
search(query="Q1 budget")                              # uses work (default)
search(query="wedding photos", account="personal")     # explicit override
manage_accounts(action="list")                         # review what's connected
manage_accounts(action="remove", account_id="personal")
```

### Migration from single-account installs

If you are upgrading from a pre-2.3 release that stored one account in `tokens.json`, your credentials are migrated to the v2 schema automatically on first boot. The migrated account is assigned the alias `default` (reserved — you can `manage_accounts set_default` it but not re-create it) and a backup of the old file is written to `tokens.json.v1-backup-<timestamp>`. No re-consent is required.

### Scopes for identity discovery

When you run `manage_accounts add`, the auth URL asks Google for the OpenID `openid` and `https://www.googleapis.com/auth/userinfo.email` scopes in addition to the Drive/Docs/Sheets/Slides/Calendar scopes. This lets the server populate the account's `email` and stable `sub` automatically. These two scopes are *not* added to the process-wide `DEFAULT_SCOPES`, so existing accounts migrated from pre-2.3 installs never see an unexpected consent screen — their record carries `pendingIdentity: true` and the email stays `unknown` until you explicitly re-add the account.

### Scope mismatches and error messages

If the resolver picks an account that doesn't hold a scope the tool needs — e.g. you connected a `personal` account with `drive.readonly` only and call a write tool — the call fails with:

```
Account 'personal' is connected but lacks the required scope for this
operation: https://www.googleapis.com/auth/drive. To re-consent
with broader scopes, run:
  manage_accounts remove personal
  manage_accounts add personal
```

The fastest fix is exactly what the error tells you: remove and re-add the alias; the second call shows Google's consent screen with the current scopes.

### Per-session caveat (HTTP transport)

With the Streamable HTTP transport in its default (single-user) mode, multiple MCP sessions sharing the same server process also share the same active default account. A `set_default` in one session is visible to the others — treat the default HTTP transport as single-user.

For genuinely multi-user deployments, use [Team mode](deployment.md#team-mode-multi-user-http-deployments): each request is authenticated with a per-user bearer token and every tool call runs as the caller, so no session can see or select another user's account.

## Authentication Flow

The server uses OAuth 2.0 for secure authentication:

### Automatic Authentication (First Run)
1. Server detects missing tokens and starts local auth server
2. Your browser opens to Google's consent page
3. Grant the requested permissions
4. Tokens are saved securely to `~/.config/google-drive-mcp/tokens.json`
5. Server continues startup

### Token Management
- **Automatic Refresh**: Tokens refresh automatically before expiration
- **Secure Storage**: Tokens stored with 0600 permissions
- **Migration**: Legacy tokens are automatically migrated to secure location

### Manual Re-authentication

Run the auth command when you need to:
- Bootstrap the very first account on a fresh install (subsequent accounts use `manage_accounts add` — see [Multi-Account Support](#multi-account-support))
- Refresh expired tokens (Google expires refresh tokens after 7 days for apps in "Testing" status)
- Recover from revoked access

```bash
# Using npx
npx -y @piotr-agier/google-drive-mcp auth

# Using local installation
npm run auth
```

## Revoking OAuth access

Removing an account with `manage_accounts remove` deletes its local credentials but does not revoke the Google grant. To revoke it, open [Google Account Permissions](https://myaccount.google.com/permissions), select the OAuth application, and remove its access. Delete the local token entry or token file before authenticating again.

## Security

- Never commit OAuth credentials, service-account keys, access tokens, refresh tokens, or `tokens.json`.
- Store credential and token files outside the repository and restrict their filesystem permissions.
- Request only the scopes required by the enabled workflows.
- Google OAuth apps left in Testing status normally issue refresh tokens that expire after seven days.
- An environment-provided service account or access token takes precedence over local OAuth; use `authGetStatus` to verify the effective identity.

For shared HTTP deployments, follow the additional requirements in [Team-mode security notes](deployment.md#security-notes).
