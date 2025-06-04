# LiteLLM Proxy CLI

The `litellm-proxy` CLI is a command-line tool for managing your LiteLLM proxy
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

   If you have [uv](https://github.com/astral-sh/uv) installed, you can try this:

   ```shell
   uvx --from=litellm[proxy] litellm-proxy
   ```

   and if things are working, you should see something like this:

   ```shell
   Usage: litellm-proxy [OPTIONS] COMMAND [ARGS]...

   LiteLLM Proxy CLI - Manage your LiteLLM proxy server

   Options:
   --base-url TEXT  Base URL of the LiteLLM proxy server  [env var:
                     LITELLM_PROXY_URL]
   --api-key TEXT   API key for authentication  [env var:
                     LITELLM_PROXY_API_KEY]
   --help           Show this message and exit.

   Commands:
   chat         Chat with models through the LiteLLM proxy server
   credentials  Manage credentials for the LiteLLM proxy server
   http         Make HTTP requests to the LiteLLM proxy server
   keys         Manage API keys for the LiteLLM proxy server
   models       Manage models on your LiteLLM proxy server
   ```

   If this works, you can make use of the tool more convenient by doing:

   ```shell
   uv tool install litellm[proxy]
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

   In the future if you want to upgrade, you can do so with:

   ```shell
   uv tool upgrade litellm[proxy]
   ```

   or if you want to uninstall, you can do so with:

   ```shell
   uv tool uninstall litellm
   ```

   If you don't have uv or otherwise want to use pip, you can activate a virtual
   environment and install the package manually:

   ```bash
   pip install 'litellm[proxy]'
   ```

2. **Set up environment variables**

   ```bash
   export LITELLM_PROXY_URL=http://localhost:4000
   export LITELLM_PROXY_API_KEY=sk-your-key
   ```

   *(Replace with your actual proxy URL and API key)*

3. **Make your first request (list models)**

   ```bash
   litellm-proxy models list
   ```

   If the CLI is set up correctly, you should see a list of available models or a table output.

4. **Troubleshooting**

   - If you see an error, check your environment variables and proxy server status.

## Configuration

You can configure the CLI using environment variables or command-line options:

- `LITELLM_PROXY_URL`: Base URL of the LiteLLM proxy server (default: http://localhost:4000)
- `LITELLM_PROXY_API_KEY`: API key for authentication

## Main Commands

### Models Management

- List, add, update, get, and delete models on the proxy.
- Example:

  ```bash
  litellm-proxy models list
  litellm-proxy models add gpt-4 \
    --param api_key=sk-123 \
    --param max_tokens=2048
  litellm-proxy models update <model-id> -p temperature=0.7
  litellm-proxy models delete <model-id>
  ```

  [API used (OpenAPI)](https://litellm-api.up.railway.app/#/model%20management)

### Credentials Management

- List, create, get, and delete credentials for LLM providers.
- Example:

  ```bash
  litellm-proxy credentials list
  litellm-proxy credentials create azure-prod \
    --info='{"custom_llm_provider": "azure"}' \
    --values='{"api_key": "sk-123", "api_base": "https://prod.azure.openai.com"}'
  litellm-proxy credentials get azure-cred
  litellm-proxy credentials delete azure-cred
  ```

  [API used (OpenAPI)](https://litellm-api.up.railway.app/#/credential%20management)

### Keys Management

- List, generate, get info, and delete API keys.
- Example:

  ```bash
  litellm-proxy keys list
  litellm-proxy keys generate \
    --models=gpt-4 \
    --spend=100 \
    --duration=24h \
    --key-alias=my-key
  litellm-proxy keys info --key sk-key1
  litellm-proxy keys delete --keys sk-key1,sk-key2 --key-aliases alias1,alias2
  ```

  [API used (OpenAPI)](https://litellm-api.up.railway.app/#/key%20management)

### User Management

- List, create, get info, and delete users.
- Example:

  ```bash
  litellm-proxy users list
  litellm-proxy users create \
    --email=user@example.com \
    --role=internal_user \
    --alias="Alice" \
    --team=team1 \
    --max-budget=100.0
  litellm-proxy users get --id <user-id>
  litellm-proxy users delete <user-id>
  ```

  [API used (OpenAPI)](https://litellm-api.up.railway.app/#/Internal%20User%20management)

### Chat Completions

- Ask for chat completions from the proxy server.
- Example:

  ```bash
  litellm-proxy chat completions gpt-4 -m "user:Hello, how are you?"
  ```

  [API used (OpenAPI)](https://litellm-api.up.railway.app/#/chat%2Fcompletions)

### General HTTP Requests

- Make direct HTTP requests to the proxy server.
- Example:

  ```bash
  litellm-proxy http request \
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
   litellm-proxy models list
   ```

2. **Add a new model:**

   ```bash
   litellm-proxy models add gpt-4 \
     --param api_key=sk-123 \
     --param max_tokens=2048
   ```

3. **Create a credential:**

   ```bash
   litellm-proxy credentials create azure-prod \
     --info='{"custom_llm_provider": "azure"}' \
     --values='{"api_key": "sk-123", "api_base": "https://prod.azure.openai.com"}'
   ```

4. **Generate an API key:**

   ```bash
   litellm-proxy keys generate \
     --models=gpt-4 \
     --spend=100 \
     --duration=24h \
     --key-alias=my-key
   ```

5. **Chat completion:**

   ```bash
   litellm-proxy chat completions gpt-4 \
     -m "user:Write a story"
   ```

6. **Custom HTTP request:**

   ```bash
   litellm-proxy http request \
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
