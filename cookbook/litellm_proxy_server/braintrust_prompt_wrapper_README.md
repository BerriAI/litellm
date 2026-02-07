# Braintrust Prompt Wrapper for LiteLLM

This directory contains a wrapper server that enables LiteLLM to use prompts from [Braintrust](https://www.braintrust.dev/) through the generic prompt management API.

## Architecture

```
┌─────────────┐         ┌──────────────────────┐         ┌─────────────┐
│   LiteLLM   │ ──────> │  Wrapper Server      │ ──────> │  Braintrust │
│   Client    │         │  (This Server)       │         │     API     │
└─────────────┘         └──────────────────────┘         └─────────────┘
    Uses generic             Transforms                    Stores actual
    prompt manager        Braintrust format              prompt templates
                         to LiteLLM format
```

## Components

### 1. Generic Prompt Manager (`litellm/integrations/generic_prompt_management/`)

A generic client that can work with any API implementing the `/beta/litellm_prompt_management` endpoint.

**Expected API Response Format:**
```json
{
  "prompt_id": "string",
  "prompt_template": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello {name}"}
  ],
  "prompt_template_model": "gpt-4",
  "prompt_template_optional_params": {
    "temperature": 0.7,
    "max_tokens": 100
  }
}
```

### 2. Braintrust Wrapper Server (`braintrust_prompt_wrapper_server.py`)

A FastAPI server that:
- Implements the `/beta/litellm_prompt_management` endpoint
- Fetches prompts from Braintrust API
- Transforms Braintrust response format to LiteLLM format

## Setup

### Install Dependencies

```bash
pip install fastapi uvicorn httpx litellm
```

### Set Environment Variables

```bash
export BRAINTRUST_API_KEY="your-braintrust-api-key"
```

## Usage

### Step 1: Start the Wrapper Server

```bash
python braintrust_prompt_wrapper_server.py
```

The server will start on `http://localhost:8080` by default.

You can customize the port and host:
```bash
export PORT=8000
export HOST=0.0.0.0
python braintrust_prompt_wrapper_server.py
```

### Step 2: Use with LiteLLM

```python
import litellm
from litellm.integrations.generic_prompt_management import GenericPromptManager

# Configure the generic prompt manager to use your wrapper server
generic_config = {
    "api_base": "http://localhost:8080",
    "api_key": "your-braintrust-api-key",  # Will be passed to Braintrust
    "timeout": 30,
}

# Create the prompt manager
prompt_manager = GenericPromptManager(**generic_config)

# Use with completion
response = litellm.completion(
    model="generic_prompt/gpt-4",
    prompt_id="your-braintrust-prompt-id",
    prompt_variables={"name": "World"},  # Variables to substitute
    messages=[{"role": "user", "content": "Additional message"}]
)

print(response)
```

### Step 3: Direct API Testing

You can also test the wrapper API directly:

```bash
# Test with curl
curl -H "Authorization: Bearer YOUR_BRAINTRUST_TOKEN" \
     "http://localhost:8080/beta/litellm_prompt_management?prompt_id=YOUR_PROMPT_ID"

# Health check
curl http://localhost:8080/health

# Service info
curl http://localhost:8080/
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## Braintrust Format Transformation

The wrapper automatically transforms Braintrust's response format:

**Braintrust API Response:**
```json
{
  "id": "prompt-123",
  "prompt_data": {
    "prompt": {
      "type": "chat",
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant"
        }
      ]
    },
    "options": {
      "model": "gpt-4",
      "params": {
        "temperature": 0.7,
        "max_tokens": 100
      }
    }
  }
}
```

**Transformed to LiteLLM Format:**
```json
{
  "prompt_id": "prompt-123",
  "prompt_template": [
    {
      "role": "system",
      "content": "You are a helpful assistant"
    }
  ],
  "prompt_template_model": "gpt-4",
  "prompt_template_optional_params": {
    "temperature": 0.7,
    "max_tokens": 100
  }
}
```

## Supported Parameters

The wrapper automatically maps these Braintrust parameters to LiteLLM:

- `temperature`
- `max_tokens` / `max_completion_tokens`
- `top_p`
- `frequency_penalty`
- `presence_penalty`
- `n`
- `stop`
- `response_format`
- `tool_choice`
- `function_call`
- `tools`

## Variable Substitution

The generic prompt manager supports simple variable substitution:

```python
# In your Braintrust prompt:
# "Hello {name}, welcome to {place}!"

# In your code:
prompt_variables = {
    "name": "Alice",
    "place": "Wonderland"
}

# Result:
# "Hello Alice, welcome to Wonderland!"
```

Supports both `{variable}` and `{{variable}}` syntax.

## Error Handling

The wrapper provides detailed error messages:

- **401**: Missing or invalid Braintrust API token
- **404**: Prompt not found in Braintrust
- **502**: Failed to connect to Braintrust API
- **500**: Error transforming response

## Production Deployment

For production use:

1. **Use HTTPS**: Deploy behind a reverse proxy with SSL
2. **Authentication**: Add authentication to the wrapper endpoint if needed
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Caching**: Consider caching prompt responses
5. **Monitoring**: Add logging and monitoring

Example with Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi uvicorn httpx

COPY braintrust_prompt_wrapper_server.py .

ENV PORT=8080
ENV HOST=0.0.0.0

EXPOSE 8080

CMD ["python", "braintrust_prompt_wrapper_server.py"]
```

## Extending to Other Providers

This pattern can be used with any prompt management provider:

1. Create a wrapper server that implements `/beta/litellm_prompt_management`
2. Transform the provider's response to LiteLLM format
3. Use the generic prompt manager to connect

Example providers:
- Langsmith
- PromptLayer
- Humanloop
- Custom internal systems

## Troubleshooting

### "No Braintrust API token provided"
- Set `BRAINTRUST_API_KEY` environment variable
- Or pass token in `Authorization: Bearer TOKEN` header

### "Failed to connect to Braintrust API"
- Check your internet connection
- Verify Braintrust API is accessible
- Check firewall settings

### "Prompt not found"
- Verify the prompt ID exists in Braintrust
- Check that your API token has access to the prompt

## License

This wrapper is part of the LiteLLM project and follows the same license.

