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

## Global Options

- `--version`, `-v`: Print the LiteLLM Proxy client and server version and exit.

Example:

```bash
litellm-proxy version
# or
litellm-proxy --version
# or
litellm-proxy -v
```

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

#### Model Information

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

#### Import Models

Import models from a YAML file:

```bash
litellm-proxy models import models.yaml
```

Options:

- `--dry-run`: Show what would be imported without making any changes.
- `--only-models-matching-regex <regex>`: Only import models where `litellm_params.model` matches the given regex.
- `--only-access-groups-matching-regex <regex>`: Only import models where at least one item in `model_info.access_groups` matches the given regex.

Examples:

1. Import all models from a YAML file:

```bash
litellm-proxy models import models.yaml
```

2. Dry run (show what would be imported):

```bash
litellm-proxy models import models.yaml --dry-run
```

3. Only import models where the model name contains 'gpt':

```bash
litellm-proxy models import models.yaml --only-models-matching-regex gpt
```

4. Only import models with access group containing 'beta':

```bash
litellm-proxy models import models.yaml --only-access-groups-matching-regex beta
```

5. Combine both filters:

```bash
litellm-proxy models import models.yaml --only-models-matching-regex gpt --only-access-groups-matching-regex beta
```

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

### Keys Management

The CLI provides commands for managing API keys on your LiteLLM proxy server:

#### List Keys

View all API keys:

```bash
litellm-proxy keys list [--format table|json] [options]
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
litellm-proxy keys generate [options]
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
litellm-proxy keys generate --models gpt-4,gpt-3.5-turbo --spend 100 --duration 24h --key-alias my-key --team-id team123
```

#### Delete Keys

Delete API keys by key or alias:

```bash
litellm-proxy keys delete [--keys <comma-separated-keys>] [--key-aliases <comma-separated-aliases>]
```

Options:

- `--keys`: Comma-separated list of API keys to delete
- `--key-aliases`: Comma-separated list of key aliases to delete

Example:

```bash
litellm-proxy keys delete --keys sk-key1,sk-key2 --key-aliases alias1,alias2
```

#### Get Key Info

Get information about a specific API key:

```bash
litellm-proxy keys info --key <key-hash>
```

Options:

- `--key`: The key hash to get information about

Example:

```bash
litellm-proxy keys info --key sk-key1
```

### User Management

The CLI provides commands for managing users on your LiteLLM proxy server:

#### List Users

View all users:

```bash
litellm-proxy users list
```

#### Get User Info

Get information about a specific user:

```bash
litellm-proxy users get --id <user-id>
```

#### Create User

Create a new user:

```bash
litellm-proxy users create --email user@example.com --role internal_user --alias "Alice" --team team1 --max-budget 100.0
```

#### Delete User

Delete one or more users by user_id:

```bash
litellm-proxy users delete <user-id-1> <user-id-2>
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

8. User management:

```bash
# List users
litellm-proxy users list

# Get user info
litellm-proxy users get --id u1

# Create a user
litellm-proxy users create --email a@b.com --role internal_user --alias "Alice" --team team1 --max-budget 100.0

# Delete users
litellm-proxy users delete u1 u2
```

9. Import models from a YAML file (with filters):

```bash
# Only import models where the model name contains 'gpt'
litellm-proxy models import models.yaml --only-models-matching-regex gpt

# Only import models with access group containing 'beta'
litellm-proxy models import models.yaml --only-access-groups-matching-regex beta

# Combine both filters
litellm-proxy models import models.yaml --only-models-matching-regex gpt --only-access-groups-matching-regex beta
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
