# CLI Authentication

Use the litellm cli to authenticate to the LiteLLM Gateway. This is great if you're trying to give a large number of developers self-serve access to the LiteLLM Gateway.


## Demo

<iframe width="840" height="500" src="https://www.loom.com/embed/87c5d243cde642ff942783024ff037e3" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Usage 

### Prerequisites - Start LiteLLM Proxy with Beta Flag

:::warning[Beta Feature - Required]

CLI SSO Authentication is currently in beta. You must set this environment variable **when starting up your LiteLLM Proxy**:

```bash
export EXPERIMENTAL_UI_LOGIN="True"
litellm --config config.yaml
```

Or add it to your proxy startup command:

```bash
EXPERIMENTAL_UI_LOGIN="True" litellm --config config.yaml
```

:::

### Configuration

#### JWT Token Expiration

By default, CLI authentication tokens expire after **24 hours**. You can customize this expiration time by setting the `LITELLM_CLI_JWT_EXPIRATION_HOURS` environment variable when starting your LiteLLM Proxy:

```bash
# Set CLI JWT tokens to expire after 48 hours
export LITELLM_CLI_JWT_EXPIRATION_HOURS=48
export EXPERIMENTAL_UI_LOGIN="True"
litellm --config config.yaml
```

Or in a single command:

```bash
LITELLM_CLI_JWT_EXPIRATION_HOURS=48 EXPERIMENTAL_UI_LOGIN="True" litellm --config config.yaml
```

**Examples:**
- `LITELLM_CLI_JWT_EXPIRATION_HOURS=12` - Tokens expire after 12 hours
- `LITELLM_CLI_JWT_EXPIRATION_HOURS=168` - Tokens expire after 7 days (168 hours)
- `LITELLM_CLI_JWT_EXPIRATION_HOURS=720` - Tokens expire after 30 days (720 hours)

:::note[Experimental UI Session]
When `EXPERIMENTAL_UI_LOGIN` is enabled, the **browser UI login** session uses a fixed 10-minute expiry (not configurable). `LITELLM_UI_SESSION_DURATION` applies only to non-experimental flows.
:::

:::tip
You can check your current token's age and expiration status using:
```bash
litellm-proxy whoami
```
:::

#### Attribution metadata (OIDC claims)

Map allowlisted OIDC claims into the LiteLLM user's `metadata` and return them to the CLI in `/sso/cli/poll` as `attribution_metadata`. Use this for stable attribution fields (for example employment type or cost center) without parsing large group lists in the client.

Set on the **proxy** before startup:

```bash
export CLI_SSO_CLAIM_MAP="employment_type->acme_employment_type,org_info.department->department"
export GENERIC_USER_EXTRA_ATTRIBUTES="employment_type,org_info.department"
```

`CLI_SSO_CLAIM_MAP` and `LITELLM_CLI_SSO_CLAIM_MAP` are equivalent. Format: comma-separated `source_claim->metadata_key` pairs. The optional `metadata.` prefix on the destination is stripped; values are stored on the user's `metadata` JSON column.

| Part | Meaning |
|------|---------|
| `source_claim` | OIDC claim path (dot notation), including fields from `GENERIC_USER_EXTRA_ATTRIBUTES` |
| `metadata_key` | Key under LiteLLM user `metadata` (supports nested keys via dots) |

Only non-secret scalar values (`string`, `int`, `float`, `bool`) are persisted and returned. Lists, objects, and destination keys containing fragments like `token` or `secret` are dropped.

Example poll response (after SSO completes):

```json
{
  "status": "ready",
  "key": "eyJ...",
  "user_id": "user@company.com",
  "attribution_metadata": {
    "acme_employment_type": "full_time",
    "department": "Engineering"
  }
}
```

**Local testing without a real IdP:** run `python scripts/mock_oidc_server_for_cli_sso.py` from the LiteLLM repo, point Generic SSO env vars at `http://127.0.0.1:8765`, then run `python scripts/test_cli_sso_claims_e2e.py`.

### Steps

1. **Install the CLI**

   If you have [uv](https://github.com/astral-sh/uv) installed, you can try this:

   ```shell
   uv tool install 'litellm[proxy]'
   ```

   If that works, you'll see something like this:

   ```shell
   ...
   Installed 2 executables: litellm, litellm-proxy
   ```

   and now you can use the tool by just typing `litellm-proxy` in your terminal:

   ```shell
   litellm-proxy
   ```

2. **Set up environment variables**

   On your local machine, set the proxy URL:

   ```bash
   export LITELLM_PROXY_URL=http://localhost:4000
   ```

   *(Replace with your actual proxy URL)*

3. **Login**

   ```shell
   litellm-proxy login
   ```

   This will open a browser window to authenticate. If you have connected LiteLLM Proxy to your SSO provider, you should be able to login with your SSO credentials. Once logged in, you can use the CLI to make requests to the LiteLLM Gateway.

4. **Make a test request to view models**

   ```shell
   litellm-proxy models list
   ```

   This will list all the models available to you.