# LiteLLM Proxy CLI

The `litellm-proxy` CLI is a command-line tool for managing your LiteLLM proxy server. It provides commands for managing models, credentials, API keys, users, and more, as well as making chat and HTTP requests to the proxy server.

| Feature                | What you can do                                 |
|------------------------|-------------------------------------------------|
| Models Management      | List, add, update, and delete models            |
| Credentials Management | Manage provider credentials                     |
| Keys Management        | Generate, list, and delete API keys             |
| User Management        | Create, list, and delete users                  |
| Chat Completions       | Run chat completions                            |
| HTTP Requests          | Make custom HTTP requests to the proxy server   |

## Installation

```bash
pip install 'litellm[proxy]'
```

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
  litellm-proxy models add gpt-4 -p api_key=sk-123 -p api_base=https://api.openai.com
  litellm-proxy models update <model-id> -p temperature=0.7
  litellm-proxy models delete <model-id>
  ```

### Credentials Management
- List, create, get, and delete credentials for LLM providers.
- Example:
  ```bash
  litellm-proxy credentials list
  litellm-proxy credentials create azure-cred --info '{"custom_llm_provider": "azure"}' --values '{"api_key": "sk-123", "api_base": "https://example.azure.openai.com"}'
  litellm-proxy credentials get azure-cred
  litellm-proxy credentials delete azure-cred
  ```

### Keys Management
- List, generate, get info, and delete API keys.
- Example:
  ```bash
  litellm-proxy keys list
  litellm-proxy keys generate --models gpt-4,gpt-3.5-turbo --spend 100 --duration 24h --key-alias my-key
  litellm-proxy keys info --key sk-key1
  litellm-proxy keys delete --keys sk-key1,sk-key2 --key-aliases alias1,alias2
  ```

### User Management
- List, create, get info, and delete users.
- Example:
  ```bash
  litellm-proxy users list
  litellm-proxy users create --email user@example.com --role internal_user --alias "Alice" --team team1 --max-budget 100.0
  litellm-proxy users get --id <user-id>
  litellm-proxy users delete <user-id>
  ```

### Chat & HTTP Requests
- Make chat completions or direct HTTP requests to the proxy server.
- Example:
  ```bash
  litellm-proxy chat completions gpt-4 -m "user:Hello, how are you?"
  litellm-proxy http request POST /chat/completions -j '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
  ```

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
