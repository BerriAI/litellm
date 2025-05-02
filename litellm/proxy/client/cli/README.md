# LiteLLM Proxy CLI

The LiteLLM Proxy CLI is a command-line tool for managing your LiteLLM proxy server. It provides commands for managing models, viewing server status, and interacting with the proxy server.

## Installation

```bash
pip install 'litellm[proxy]'
```

## Configuration

The CLI can be configured using environment variables or command-line options:

- `LITELLM_PROXY_URL`: Base URL of the LiteLLM proxy server (default: http://localhost:4000)
- `LITELLM_PROXY_API_KEY`: API key for authentication

## Commands

### Models Management

The CLI provides several commands for managing models on your LiteLLM proxy server:

#### List Models

View all available models:

```bash
litellm-proxy models list [--format table|json]
```

Options:

- `--format`: Output format (table or json, default: table)

#### Add Model

Add a new model to the proxy:

```bash
litellm-proxy models add <model-name> [options]
```

Options:

- `--param`, `-p`: Model parameters in key=value format (can be specified multiple times)
- `--info`, `-i`: Model info in key=value format (can be specified multiple times)

Example:

```bash
litellm-proxy models add gpt-4 -p api_key=sk-123 -p api_base=https://api.openai.com -i description="GPT-4 model"
```

#### Get Model Info

Get information about a specific model:

```bash
litellm-proxy models get [--id MODEL_ID] [--name MODEL_NAME]
```

Options:

- `--id`: ID of the model to retrieve
- `--name`: Name of the model to retrieve

#### Delete Model

Delete a model from the proxy:

```bash
litellm-proxy models delete <model-id>
```

#### Update Model

Update an existing model's configuration:

```bash
litellm-proxy models update <model-id> [options]
```

Options:

- `--param`, `-p`: Model parameters in key=value format (can be specified multiple times)
- `--info`, `-i`: Model info in key=value format (can be specified multiple times)

### Credentials Management

The CLI provides commands for managing credentials on your LiteLLM proxy server:

#### List Credentials

View all available credentials:

```bash
litellm-proxy credentials list [--format table|json]
```

Options:

- `--format`: Output format (table or json, default: table)

The table format displays:
- Credential Name
- Custom LLM Provider

#### Create Credential

Create a new credential:

```bash
litellm-proxy credentials create <credential-name> --info <json-string> --values <json-string>
```

Options:

- `--info`: JSON string containing credential info (e.g., custom_llm_provider)
- `--values`: JSON string containing credential values (e.g., api_key)

Example:

```bash
litellm-proxy credentials create azure-cred \
  --info '{"custom_llm_provider": "azure"}' \
  --values '{"api_key": "sk-123", "api_base": "https://example.azure.openai.com"}'
```

#### Get Credential

Get information about a specific credential:

```bash
litellm-proxy credentials get <credential-name>
```

#### Delete Credential

Delete a credential:

```bash
litellm-proxy credentials delete <credential-name>
```

### Chat Commands

The CLI provides commands for interacting with chat models through your LiteLLM proxy server:

#### Chat Completions

Create a chat completion:

```bash
litellm-proxy chat completions <model> [options]
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
litellm-proxy chat completions gpt-4 -m "user:Hello, how are you?"
```

2. Multi-message conversation:
```bash
litellm-proxy chat completions gpt-4 \
  -m "system:You are a helpful assistant" \
  -m "user:What's the capital of France?" \
  -m "assistant:The capital of France is Paris." \
  -m "user:What's its population?"
```

3. With generation parameters:
```bash
litellm-proxy chat completions gpt-4 \
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
litellm-proxy http request <method> <uri> [options]
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
litellm-proxy http request GET /models
```

2. Create a chat completion:
```bash
litellm-proxy http request POST /chat/completions -j '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

3. Test connection with custom headers:
```bash
litellm-proxy http request GET /health/test_connection -H "X-Custom-Header:value"
```

### Model Information

Get detailed information about all models:

```bash
litellm-proxy models info [options]
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

## Environment Variables

The CLI respects the following environment variables:

- `LITELLM_PROXY_URL`: Base URL of the proxy server
- `LITELLM_PROXY_API_KEY`: API key for authentication

## Examples

1. List all models in table format:

```bash
litellm-proxy models list
```

2. Add a new model with parameters:

```bash
litellm-proxy models add gpt-4 -p api_key=sk-123 -p max_tokens=2048
```

3. Get model information in JSON format:

```bash
litellm-proxy models info --format json
```

4. Update model parameters:

```bash
litellm-proxy models update model-123 -p temperature=0.7 -i description="Updated model"
```

5. List all credentials in table format:

```bash
litellm-proxy credentials list
```

6. Create a new credential for Azure:

```bash
litellm-proxy credentials create azure-prod \
  --info '{"custom_llm_provider": "azure"}' \
  --values '{"api_key": "sk-123", "api_base": "https://prod.azure.openai.com"}'
```

7. Make a custom HTTP request:

```bash
litellm-proxy http request POST /chat/completions \
  -j '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}' \
  -H "X-Custom-Header:value"
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
