# LiteLLM Proxy CLI

The LiteLLM Proxy CLI is a command-line tool for managing your LiteLLM proxy server. It provides commands for managing models, viewing server status, and interacting with the proxy server.

## Installation

```bash
uv tool install 'litellm[proxy]'
```

## Configuration

The CLI can be configured using environment variables or command-line options:

- `LITELLM_PROXY_URL`: Base URL of the LiteLLM proxy server (default: http://localhost:4000)
- `LITELLM_PROXY_API_KEY`: API key for authentication

## Global Options

- `--version`, `-v`: Print the LiteLLM Proxy client and server version and exit.

Example:

```bash
lite version
# or
lite --version
# or
lite -v
```

## Commands

### Models Management

The CLI provides several commands for managing models on your LiteLLM proxy server:

#### List Models

View all available models:

```bash
lite models list [--format table|json]
```

Options:

- `--format`: Output format (table or json, default: table)

#### Model Information

Get detailed information about all models:

```bash
lite models info [options]
```

Options:

- `--format`: Output format (table or json, default: table)
- `--columns`: Comma-separated list of columns to display. Valid columns:
  - `public_model`
  - `upstream_model`
  - `credential_name`
  - `created_at`
  - `updated_at`
  - `id`
  - `input_cost`
  - `output_cost`

Default columns: `public_model`, `upstream_model`, `updated_at`

#### Add Model

Add a new model to the proxy:

```bash
lite models add <model-name> [options]
```

Options:

- `--param`, `-p`: Model parameters in key=value format (can be specified multiple times)
- `--info`, `-i`: Model info in key=value format (can be specified multiple times)

Example:

```bash
lite models add gpt-4 -p api_key=sk-123 -p api_base=https://api.openai.com -i description="GPT-4 model"
```

#### Get Model Info

Get information about a specific model:

```bash
lite models get [--id MODEL_ID] [--name MODEL_NAME]
```

Options:

- `--id`: ID of the model to retrieve
- `--name`: Name of the model to retrieve

#### Delete Model

Delete a model from the proxy:

```bash
lite models delete <model-id>
```

#### Update Model

Update an existing model's configuration:

```bash
lite models update <model-id> [options]
```

Options:

- `--param`, `-p`: Model parameters in key=value format (can be specified multiple times)
- `--info`, `-i`: Model info in key=value format (can be specified multiple times)

#### Import Models

Import models from a YAML file:

```bash
lite models import models.yaml
```

Options:

- `--dry-run`: Show what would be imported without making any changes.
- `--only-models-matching-regex <regex>`: Only import models where `litellm_params.model` matches the given regex.
- `--only-access-groups-matching-regex <regex>`: Only import models where at least one item in `model_info.access_groups` matches the given regex.

Examples:

1. Import all models from a YAML file:

```bash
lite models import models.yaml
```

2. Dry run (show what would be imported):

```bash
lite models import models.yaml --dry-run
```

3. Only import models where the model name contains 'gpt':

```bash
lite models import models.yaml --only-models-matching-regex gpt
```

4. Only import models with access group containing 'beta':

```bash
lite models import models.yaml --only-access-groups-matching-regex beta
```

5. Combine both filters:

```bash
lite models import models.yaml --only-models-matching-regex gpt --only-access-groups-matching-regex beta
```

### Credentials Management

The CLI provides commands for managing credentials on your LiteLLM proxy server:

#### List Credentials

View all available credentials:

```bash
lite credentials list [--format table|json]
```

Options:

- `--format`: Output format (table or json, default: table)

The table format displays:
- Credential Name
- Custom LLM Provider

#### Create Credential

Create a new credential:

```bash
lite credentials create <credential-name> --info <json-string> --values <json-string>
```

Options:

- `--info`: JSON string containing credential info (e.g., custom_llm_provider)
- `--values`: JSON string containing credential values (e.g., api_key)

Example:

```bash
lite credentials create azure-cred \
  --info '{"custom_llm_provider": "azure"}' \
  --values '{"api_key": "sk-123", "api_base": "https://example.azure.openai.com"}'
```

#### Get Credential

Get information about a specific credential:

```bash
lite credentials get <credential-name>
```

#### Delete Credential

Delete a credential:

```bash
lite credentials delete <credential-name>
```

### Keys Management

The CLI provides commands for managing API keys on your LiteLLM proxy server:

#### List Keys

View all API keys:

```bash
lite keys list [--format table|json] [options]
```

Options:

- `--format`: Output format (table or json, default: table)
- `--page`: Page number for pagination
- `--size`: Number of items per page
- `--user-id`: Filter keys by user ID
- `--team-id`: Filter keys by team ID
- `--organization-id`: Filter keys by organization ID
- `--key-hash`: Filter by specific key hash
- `--key-alias`: Filter by key alias
- `--return-full-object`: Return the full key object
- `--include-team-keys`: Include team keys in the response

#### Generate Key

Generate a new API key:

```bash
lite keys generate [options]
```

Options:

- `--models`: Comma-separated list of allowed models
- `--aliases`: JSON string of model alias mappings
- `--spend`: Maximum spend limit for this key
- `--duration`: Duration for which the key is valid (e.g. '24h', '7d')
- `--key-alias`: Alias/name for the key
- `--team-id`: Team ID to associate the key with
- `--user-id`: User ID to associate the key with
- `--budget-id`: Budget ID to associate the key with
- `--config`: JSON string of additional configuration parameters

Example:

```bash
lite keys generate --models gpt-4,gpt-3.5-turbo --spend 100 --duration 24h --key-alias my-key --team-id team123
```

#### Delete Keys

Delete API keys by key or alias:

```bash
lite keys delete [--keys <comma-separated-keys>] [--key-aliases <comma-separated-aliases>]
```

Options:

- `--keys`: Comma-separated list of API keys to delete
- `--key-aliases`: Comma-separated list of key aliases to delete

Example:

```bash
lite keys delete --keys sk-key1,sk-key2 --key-aliases alias1,alias2
```

#### Get Key Info

Get information about a specific API key:

```bash
lite keys info --key <key-hash>
```

Options:

- `--key`: The key hash to get information about

Example:

```bash
lite keys info --key sk-key1
```

### User Management

The CLI provides commands for managing users on your LiteLLM proxy server:

#### List Users

View all users:

```bash
lite users list
```

#### Get User Info

Get information about a specific user:

```bash
lite users get --id <user-id>
```

#### Create User

Create a new user:

```bash
lite users create --email user@example.com --role internal_user --alias "Alice" --team team1 --max-budget 100.0
```

#### Delete User

Delete one or more users by user_id:

```bash
lite users delete <user-id-1> <user-id-2>
```

### Chat Commands

The CLI provides commands for interacting with chat models through your LiteLLM proxy server:

#### Chat Completions

Create a chat completion:

```bash
lite chat completions <model> [options]
```

Arguments:
- `model`: The model to use (e.g., gpt-4, claude-2)

Options:
- `--message`, `-m`: Messages in 'role:content' format. Can be specified multiple times to create a conversation.
- `--temperature`, `-t`: Sampling temperature between 0 and 2
- `--top-p`: Nucleus sampling parameter between 0 and 1
- `--n`: Number of completions to generate
- `--max-tokens`: Maximum number of tokens to generate
- `--presence-penalty`: Presence penalty between -2.0 and 2.0
- `--frequency-penalty`: Frequency penalty between -2.0 and 2.0
- `--user`: Unique identifier for the end user

Examples:

1. Simple completion:
```bash
lite chat completions gpt-4 -m "user:Hello, how are you?"
```

2. Multi-message conversation:
```bash
lite chat completions gpt-4 \
  -m "system:You are a helpful assistant" \
  -m "user:What's the capital of France?" \
  -m "assistant:The capital of France is Paris." \
  -m "user:What's its population?"
```

3. With generation parameters:
```bash
lite chat completions gpt-4 \
  -m "user:Write a story" \
  --temperature 0.7 \
  --max-tokens 500 \
  --top-p 0.9
```

### HTTP Commands

The CLI provides commands for making direct HTTP requests to your LiteLLM proxy server:

#### Make HTTP Request

Make an HTTP request to any endpoint:

```bash
lite http request <method> <uri> [options]
```

Arguments:
- `method`: HTTP method (GET, POST, PUT, DELETE, etc.)
- `uri`: URI path (will be appended to base_url)

Options:
- `--data`, `-d`: Data to send in the request body (as JSON string)
- `--json`, `-j`: JSON data to send in the request body (as JSON string)
- `--header`, `-H`: HTTP headers in 'key:value' format. Can be specified multiple times.

Examples:

1. List models:
```bash
lite http request GET /models
```

2. Create a chat completion:
```bash
lite http request POST /chat/completions -j '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

3. Test connection with custom headers:
```bash
lite http request GET /health/test_connection -H "X-Custom-Header:value"
```

### Run a Coding Agent

Launch a coding agent with all of its LLM traffic routed through your LiteLLM proxy. Each supported agent is its own command, so there is nothing to remember beyond the agent's name:

```bash
lite claude
lite codex
lite opencode
```

Anything you type after the agent name is forwarded to it untouched, so the usual flags keep working:

```bash
lite claude --resume
lite codex exec "summarize the repo"
```

Each command resolves your LiteLLM key (logging in via SSO when none is stored and you are at a terminal; otherwise it expects `LITELLM_PROXY_API_KEY` or `--api-key`), checks the key against the proxy so bad credentials fail immediately instead of deep inside the agent, exports the environment variables the agent reads, then replaces itself with the agent process.

The right variables are picked per agent. Claude Code gets `ANTHROPIC_BASE_URL` (the proxy root, so it appends `/v1/messages`) and `ANTHROPIC_AUTH_TOKEN`, with any stray `ANTHROPIC_API_KEY` cleared so the proxy token wins. Codex and OpenCode get `OPENAI_BASE_URL` (the proxy plus `/v1`) and `OPENAI_API_KEY`. Codex ignores `OPENAI_BASE_URL`, so it is additionally pointed at the proxy through a custom provider passed as `-c` config overrides (HTTP/SSE Responses transport, since the proxy does not speak the Responses WebSocket protocol).

Options (these belong to the wrapper, so put them before the agent's own flags):

- `--skip-verify`: Skip the pre-launch key check (useful offline or with non-standard auth).

To pin the model, pass the agent's own model flag (for example `lite claude --model my-proxy-model` or `lite codex -m my-proxy-model`), or export the variable the agent reads (`ANTHROPIC_MODEL` / `ANTHROPIC_SMALL_FAST_MODEL` for Claude Code); the wrapper preserves anything you already have set. Whatever model the agent ends up requesting must exist on the proxy, since requests land on the proxy's `/v1/messages` (Anthropic) or `/v1/chat/completions` and `/v1/responses` (OpenAI) endpoints.

#### About the `lite login` credential

The token minted by `lite login` is a short-lived, per-session agent credential, not a managed virtual key. It is scoped to the user and team you authenticated as, inherits that user's and team's models and budgets, and is enforced on the proxy exactly like a virtual key on the same team (guardrails, routing, logging, spend). Spend is tracked against the shared team and user budgets, so running several agents (or logging in more than once) does not hand each session its own separate budget; they all draw down the same team/user allowance. There is no separate per-session cap, so sustained agent use is not capped at a small chat-session limit.

The credential is short-lived by design (default 24h, configurable via `LITELLM_CLI_JWT_EXPIRATION_HOURS`); run `lite login` again to refresh it, which also re-reads your latest team and user settings. It does not appear in the Keys UI and cannot be rotated or revoked mid-session. `lite auth print-token` (usable as Claude Code's `apiKeyHelper`) prints it while it's still fresh and fails once it expires -- there is no silent renewal, so a long-running session needs a fresh `lite login` once a day. `lite claude`, `lite codex`, and `lite opencode` work with it on a default deployment; `EXPERIMENTAL_UI_LOGIN` is not required. If you need a long-lived, rotatable key that shows up in the Keys UI, create a dedicated virtual key in the dashboard and pass it via `--api-key` or `LITELLM_PROXY_API_KEY` instead.

### Route Every Claude Code Session Through the Proxy

`lite claude` wraps a single invocation, but `lite up` goes further: it patches `~/.claude/settings.json`, Claude Code's own config file, so that every Claude Code session started afterward -- from any terminal, launched normally with just `claude`, no wrapper needed -- routes through your LiteLLM proxy. It sets `env.ANTHROPIC_BASE_URL` to the proxy URL and `apiKeyHelper` to a `lite auth print-token` invocation, drops any stray static `ANTHROPIC_API_KEY` so the helper-issued token wins, and leaves every other setting in the file untouched. It backs up the original file before patching it.

Two things need to already be true: you've run `lite login`, since the apiKeyHelper depends on that stored token, and the proxy is already reachable, since `lite up` does not start one for you.

```bash
lite login
litellm --config litellm/proxy/dev_config.yaml &
lite up
```

`lite up` runs in the foreground and blocks. Press Ctrl-C to stop it, which restores the original settings file and exits. If the process is ever killed uncleanly instead -- `kill -9`, a crash -- the settings file is left patched, and `lite down` is the manual recovery path: run it at any later point to restore from the same backup.

This is a one-time file patch and restore, not a live traffic interceptor. A Claude Code session already running before `lite up` started keeps whatever `ANTHROPIC_BASE_URL` and token it loaded at its own startup, and a session still running when `lite up` stops keeps routing through the proxy until it exits; only sessions *started* while the patch is in effect are affected, and only *new* sessions after a restore go back to Anthropic directly.

Cursor is not supported: it has no equivalent file-based config to hot-patch this way, since its model routing lives in its own app storage and is configured through its GUI.

### QA Complexity-Based Auto-Routing Against Your Real Proxy

`lite autoroute` lets you try LiteLLM's complexity-based auto-routing -- picking a cheaper or more expensive model depending on how complex a prompt looks -- against models your key already has access to on your real, running proxy, without editing that proxy's `config.yaml` and without any real request ever bypassing it. It builds a second, throwaway proxy locally that forwards every request back to your real proxy, and points Claude Code at that local proxy for the duration of the session.

#### Install the CLI

`lite autoroute up` builds and runs a throwaway litellm proxy locally, so unlike the rest of this CLI it needs the proxy server runtime, not just the thin `litellm[cli]` client. Install `litellm[proxy]` (which ships the `lite` command too) with a single curl command -- no existing Python tooling required, `uv` is bootstrapped automatically if missing:

```bash
curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/install.sh | sh
```

To QA an unreleased branch or commit instead of the latest PyPI release, set `LITELLM_CLI_REF`:

```bash
curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/<branch-or-commit>/scripts/install.sh | \
  LITELLM_CLI_REF=<branch-or-commit> sh
```

The thin `scripts/install-cli.sh` installs only `litellm[cli]`, which is enough for `lite login`, `lite claude`, and `lite up`, but not for `lite autoroute up`; running it against a `litellm[cli]` install fails fast with a message telling you to install the proxy runtime.

Point the CLI at your real proxy and key before running any `lite model-groups` or `lite autoroute` command -- like every other command in this CLI, they read `LITELLM_PROXY_URL`/`LITELLM_PROXY_API_KEY` (or `--base-url`/`--api-key`), no `lite login` required:

```bash
export LITELLM_PROXY_URL=http://localhost:4000
export LITELLM_PROXY_API_KEY=sk-...
```

#### List Your Accessible Model Groups

```bash
lite model-groups list [--format table|json]
```

Lists the model groups your key can reach on the proxy, via `/model_group/info`, along with each group's mode (`chat`, `embedding`, etc.) and per-token pricing. This is also what `lite autoroute configure` uses internally to discover what it can offer you.

#### Configure the Auto-Router

```bash
lite autoroute configure
```

An interactive wizard. It runs the same model-group discovery as above, splits the results into chat-capable and embedding-capable pools, and asks you to assign one or more models from the chat pool to each of the four complexity tiers -- SIMPLE, MEDIUM, COMPLEX, REASONING. Each tier's picker is a type-to-filter fuzzy search (fzf-style) rather than a scrollable numbered list, so it stays usable even with hundreds of model groups: type a substring to narrow the list, tab to toggle a model into the selection, enter to confirm (assigning more than one model to a tier is exactly when this matters -- complexity_router picks randomly among a tier's pool per request, and adaptive mode specifically depends on having more than one candidate to choose from). From there it optionally offers: classifying prompt complexity with an LLM (again picked from your discovered pool) instead of the free built-in heuristic scorer, semantic keyword matching for tier assignment (needs an embedding model from the pool), and adaptive (bandit-based) selection layered on top of tiering.

The wizard writes the result to `~/.litellm/autorouter/config.yaml` with `0600` permissions, since the file embeds your real proxy API key. Every model referenced anywhere in that config -- tier targets, the classifier model, the embedding model -- becomes its own `litellm_proxy/<model-name>` deployment whose `api_base` and `api_key` point back at your real proxy. That is the trick that keeps your real proxy's config untouched: every actual network call this generates, whether it is the routed completion, an LLM-classifier call, or an embedding call, forwards transparently through your real, already-running proxy with your real key.

You do not need to tell Claude Code to request `autorouter` by name yourself: `lite autoroute up` also sets `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL`, and `ANTHROPIC_DEFAULT_OPUS_MODEL` to `autorouter` in `~/.claude/settings.json`, so every one of Claude Code's own model tiers requests it directly regardless of `/model` or whatever it defaults to otherwise. (A bare `model_name: "*"` deployment looks like the obvious way to catch any request instead, but litellm's Router looks up auto-router deployments by the literal requested model string with no wildcard resolution, so a `"*"` entry would never actually match real traffic -- these env var overrides are what makes it work.)

You must run `configure` at least once before `up`; running `up` first fails with a clear error telling you to configure first.

#### Launch the Ephemeral Auto-Router Proxy

```bash
lite autoroute up
```

Starts a local, throwaway litellm proxy on a random free port, running the config `configure` generated, with the fixed local development key `sk-1234` baked into the generated config. It waits for the ephemeral proxy to report healthy, then patches `~/.claude/settings.json` the same way `lite up` does, except with a static `ANTHROPIC_AUTH_TOKEN` env var instead of an `apiKeyHelper`. Any `claude` session started afterward, from any terminal, routes through the ephemeral proxy. Pass `--debug` or `--detailed-debug` to forward those logging flags to the local proxy; startup output includes the config and log paths, port, and key.

`lite autoroute up` runs in the foreground and streams the ephemeral proxy's own log file into your terminal, so you can watch its routing decisions -- which tier and model got picked for each request -- as you use Claude Code normally. Press Ctrl-C (or send SIGTERM) to stop it; this kills the child proxy process and restores your original Claude Code settings, in that order.

#### Recover From an Unclean Shutdown

```bash
lite autoroute down
```

If the `lite autoroute up` process dies uncleanly -- `kill -9`, a crash -- rather than being stopped with Ctrl-C, `down` is the manual recovery path: it kills any leftover ephemeral proxy process found via a recorded pid file and restores Claude Code's settings from whatever backup is on disk.

#### Example

```bash
lite autoroute configure
lite autoroute up
# use Claude Code as normal in another terminal; routing decisions stream live
lite autoroute down   # only needed if `up` was killed uncleanly instead of Ctrl-C'd
```

#### Caveats

Adaptive mode's learned state does not persist across `lite autoroute up` sessions -- there is no local database, so every session starts adaptive selection cold. A Claude Code session already running before `up` started, or still running when it stops, keeps whatever settings it loaded at its own startup; like `lite up`, this is a one-time file patch and restore, not a live traffic interceptor. Only Claude Code is supported, for the same reason as `lite up`: no other supported agent (for example Cursor) has an equivalent hot-patchable config file.

A session that outlives `up` (or is still running the moment you stop it) keeps sending requests, master key included, to that now-freed loopback port until you restart it. Once the ephemeral proxy process exits, nothing stops another local account on the same machine from binding that same port and receiving those requests instead -- unlike `lite up`'s `apiKeyHelper`, which is re-resolved per request, `autoroute`'s master key is a static value, so whoever receives them gets a live-looking token along with the prompt content. Restart any Claude Code session before you consider the machine clean, run `lite autoroute down` promptly rather than leaving a stopped session's settings patched, and do not run `lite autoroute up` on a shared or multi-tenant host.

Do not run `lite up` and `lite autoroute up` at the same time. Each patches `~/.claude/settings.json` and keeps its own separate backup, with no coordination between them: whichever one you stop or crash out of last is the one whose backup gets restored, which can silently leave the *other* mode's settings (a static master key and a now-dead loopback URL, or a stale `apiKeyHelper`) active. Run `lite down` or `lite autoroute down` (whichever applies) before switching to the other mode.

## Environment Variables

The CLI respects the following environment variables:

- `LITELLM_PROXY_URL`: Base URL of the proxy server
- `LITELLM_PROXY_API_KEY`: API key for authentication

## Examples

1. List all models in table format:

```bash
lite models list
```

2. Add a new model with parameters:

```bash
lite models add gpt-4 -p api_key=sk-123 -p max_tokens=2048
```

3. Get model information in JSON format:

```bash
lite models info --format json
```

4. Update model parameters:

```bash
lite models update model-123 -p temperature=0.7 -i description="Updated model"
```

5. List all credentials in table format:

```bash
lite credentials list
```

6. Create a new credential for Azure:

```bash
lite credentials create azure-prod \
  --info '{"custom_llm_provider": "azure"}' \
  --values '{"api_key": "sk-123", "api_base": "https://prod.azure.openai.com"}'
```

7. Make a custom HTTP request:

```bash
lite http request POST /chat/completions \
  -j '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}' \
  -H "X-Custom-Header:value"
```

8. User management:

```bash
# List users
lite users list

# Get user info
lite users get --id u1

# Create a user
lite users create --email a@b.com --role internal_user --alias "Alice" --team team1 --max-budget 100.0

# Delete users
lite users delete u1 u2
```

9. Import models from a YAML file (with filters):

```bash
# Only import models where the model name contains 'gpt'
lite models import models.yaml --only-models-matching-regex gpt

# Only import models with access group containing 'beta'
lite models import models.yaml --only-access-groups-matching-regex beta

# Combine both filters
lite models import models.yaml --only-models-matching-regex gpt --only-access-groups-matching-regex beta
```

## Error Handling

The CLI will display appropriate error messages when:

- The proxy server is not accessible
- Authentication fails
- Invalid parameters are provided
- The requested model or credential doesn't exist
- Invalid JSON is provided for credential creation
- Any other operation fails

For detailed debugging, use the `--debug` flag with any command.
