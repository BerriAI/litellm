# LiteLLM Proxy CLI

The `lite` CLI is a command-line tool for managing your LiteLLM proxy
server. It provides commands for managing models, credentials, API keys, users,
and more, as well as making chat and HTTP requests to the proxy server.

| Feature                | What you can do                                 |
|------------------------|-------------------------------------------------|
| Models Management      | List, add, update, and delete models            |
| Credentials Management | Manage provider credentials                     |
| Keys Management        | Generate, list, and delete API keys             |
| User Management        | Create, list, and delete users                  |
| Chat Completions       | Run chat completions                            |
| HTTP Requests          | Make custom HTTP requests to the proxy server   |

## Quick Start

1. **Install the CLI**

   The `lite` client is a thin laptop install: it points at a LiteLLM proxy and runs your coding agents through it, with none of the proxy server runtime pulled in. The one-line installer needs only `curl`; it bootstraps [uv](https://github.com/astral-sh/uv) when it's missing and lets uv provision a compatible Python for you:

   ```shell
   curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/scripts/install-cli.sh | sh
   ```

   On macOS you can install it with Homebrew instead:

   ```shell
   brew install BerriAI/litellm/lite
   ```

   Already have uv and prefer to drive it yourself? Install the package directly:

   ```shell
   uv tool install 'litellm[cli]'
   ```

   Any of these gives you the `lite` command; if you already run a proxy server from `litellm[proxy]`, it ships there too. Start by typing it in your terminal:

   ```shell
   lite
   ```

2. **Set up environment variables**

   ```bash
   export LITELLM_PROXY_URL=http://localhost:4000
   export LITELLM_PROXY_API_KEY=sk-your-key
   ```

   *(Replace with your actual proxy URL and API key)*

3. **Make your first request (list models)**

   ```bash
   lite models list
   ```

   If the CLI is set up correctly, you should see a list of available models or a table output.

4. **Troubleshooting**

   - If you see an error, check your environment variables and proxy server status.

## Authentication using CLI

You can use the CLI to authenticate to the LiteLLM Gateway. This is great if you're trying to give a large number of developers self-serve access to the LiteLLM Gateway.

:::info

For an indepth guide, see [CLI Authentication](./cli_sso).

:::

### Prerequisites

:::warning[Beta Feature - Required Environment Variable]

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

### Steps

1. **Set up the proxy URL**

   ```bash
   export LITELLM_PROXY_URL=http://localhost:4000
   ```

   *(Replace with your actual proxy URL)*

2. **Login**

   ```bash
   lite login
   ```

   This will open a browser window to authenticate. If you have connected LiteLLM Proxy to your SSO provider, you can login with your SSO credentials. Once logged in, you can use the CLI to make requests to the LiteLLM Gateway.

3. **Test your authentication**

   ```bash
   lite models list
   ```

   This will list all the models available to you.

## Main Commands

### Models Management

- List, add, update, get, and delete models on the proxy.
- Example:

  ```bash
  lite models list
  lite models add gpt-4 \
    --param api_key=sk-123 \
    --param max_tokens=2048
  lite models update <model-id> -p temperature=0.7
  lite models delete <model-id>
  ```

  [API used (OpenAPI)](https://litellm-api.up.railway.app/#/model%20management)

### Credentials Management

- List, create, get, and delete credentials for LLM providers.
- Example:

  ```bash
  lite credentials list
  lite credentials create azure-prod \
    --info='{"custom_llm_provider": "azure"}' \
    --values='{"api_key": "sk-123", "api_base": "https://prod.azure.openai.com"}'
  lite credentials get azure-cred
  lite credentials delete azure-cred
  ```

  [API used (OpenAPI)](https://litellm-api.up.railway.app/#/credential%20management)

### Keys Management

- List, generate, get info, and delete API keys.
- Example:

  ```bash
  lite keys list
  lite keys generate \
    --models=gpt-4 \
    --spend=100 \
    --duration=24h \
    --key-alias=my-key
  lite keys info --key sk-key1
  lite keys delete --keys sk-key1,sk-key2 --key-aliases alias1,alias2
  ```

  [API used (OpenAPI)](https://litellm-api.up.railway.app/#/key%20management)

### User Management

- List, create, get info, and delete users.
- Example:

  ```bash
  lite users list
  lite users create \
    --email=user@example.com \
    --role=internal_user \
    --alias="Alice" \
    --team=team1 \
    --max-budget=100.0
  lite users get --id <user-id>
  lite users delete <user-id>
  ```

  [API used (OpenAPI)](https://litellm-api.up.railway.app/#/Internal%20User%20management)

### Chat Completions

- Ask for chat completions from the proxy server.
- Example:

  ```bash
  lite chat completions gpt-4 -m "user:Hello, how are you?"
  ```

  [API used (OpenAPI)](https://litellm-api.up.railway.app/#/chat%2Fcompletions)

### General HTTP Requests

- Make direct HTTP requests to the proxy server.
- Example:

  ```bash
  lite http request \
    POST /chat/completions \
    --json '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
  ```

  [All APIs (OpenAPI)](https://litellm-api.up.railway.app/#/)

## Environment Variables

- `LITELLM_PROXY_URL`: Base URL of the proxy server
- `LITELLM_PROXY_API_KEY`: API key for authentication

## Examples

1. **List all models:**

   ```bash
   lite models list
   ```

2. **Add a new model:**

   ```bash
   lite models add gpt-4 \
     --param api_key=sk-123 \
     --param max_tokens=2048
   ```

3. **Create a credential:**

   ```bash
   lite credentials create azure-prod \
     --info='{"custom_llm_provider": "azure"}' \
     --values='{"api_key": "sk-123", "api_base": "https://prod.azure.openai.com"}'
   ```

4. **Generate an API key:**

   ```bash
   lite keys generate \
     --models=gpt-4 \
     --spend=100 \
     --duration=24h \
     --key-alias=my-key
   ```

5. **Chat completion:**

   ```bash
   lite chat completions gpt-4 \
     -m "user:Write a story"
   ```

6. **Custom HTTP request:**

   ```bash
   lite http request \
     POST /chat/completions \
     --json '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
   ```

## Error Handling

The CLI will display error messages for:

- Server not accessible
- Authentication failures
- Invalid parameters or JSON
- Nonexistent models/credentials
- Any other operation failures

Use the `--debug` flag for detailed debugging output.

For full command reference and advanced usage, see the [CLI README](https://github.com/BerriAI/litellm/blob/main/litellm/proxy/client/cli/README.md).
