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

## Error Handling

The CLI will display appropriate error messages when:

- The proxy server is not accessible
- Authentication fails
- Invalid parameters are provided
- The requested model doesn't exist
- Any other operation fails

For detailed debugging, use the `--debug` flag with any command.
