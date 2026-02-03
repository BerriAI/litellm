# Mock Prompt Management Server

A reference implementation of the [LiteLLM Generic Prompt Management API](https://docs.litellm.ai/docs/adding_provider/generic_prompt_management_api).

This FastAPI server demonstrates how to build a prompt management API that integrates with LiteLLM without requiring a PR to the LiteLLM repository.

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic
```

### 2. Start the Server

```bash
python mock_prompt_management_server.py
```

The server will start on `http://localhost:8080`

### 3. Test the Endpoint

```bash
# Get a prompt
curl "http://localhost:8080/beta/litellm_prompt_management?prompt_id=hello-world-prompt"

# Get a prompt with authentication
curl "http://localhost:8080/beta/litellm_prompt_management?prompt_id=hello-world-prompt" \
  -H "Authorization: Bearer test-token-12345"

# List all prompts
curl "http://localhost:8080/prompts"

# Get prompt variables
curl "http://localhost:8080/prompts/hello-world-prompt/variables"
```

## Using with LiteLLM

### Configuration

Create a `config.yaml` file:

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

prompts:
  - prompt_id: "hello-world-prompt"
    litellm_params:
      prompt_integration: "generic_prompt_management"
      api_base: http://localhost:8080
      api_key: test-token-12345
```

### Start LiteLLM Proxy

```bash
litellm --config config.yaml
```

### Make a Request

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "prompt_id": "hello-world-prompt",
    "prompt_variables": {
      "domain": "data science",
      "task": "analyzing customer behavior"
    },
    "messages": [
      {"role": "user", "content": "Please help me get started"}
    ]
  }'
```

## Available Prompts

The server includes several example prompts:

| Prompt ID | Description | Variables |
|-----------|-------------|-----------|
| `hello-world-prompt` | Basic helpful assistant | `domain`, `task` |
| `code-review-prompt` | Code review assistant | `years_experience`, `language`, `code` |
| `customer-support-prompt` | Customer support agent | `company_name`, `customer_message` |
| `data-analysis-prompt` | Data analysis expert | `analysis_type`, `dataset_name`, `data` |
| `creative-writing-prompt` | Creative writing assistant | `genre`, `length`, `topic` |

## Authentication

The server supports optional Bearer token authentication. Valid tokens for testing:

- `test-token-12345`
- `dev-token-67890`
- `prod-token-abcdef`

If no `Authorization` header is provided, requests are allowed (for testing purposes).

## API Endpoints

### LiteLLM Spec Endpoints

#### `GET /beta/litellm_prompt_management`

Get a prompt by ID (required by LiteLLM).

**Query Parameters:**
- `prompt_id` (required): The prompt ID
- `project_name` (optional): Project filter
- `slug` (optional): Slug filter
- `version` (optional): Version filter

**Response:**
```json
{
  "prompt_id": "hello-world-prompt",
  "prompt_template": [
    {
      "role": "system",
      "content": "You are a helpful assistant specialized in {domain}."
    },
    {
      "role": "user",
      "content": "Help me with: {task}"
    }
  ],
  "prompt_template_model": "gpt-4",
  "prompt_template_optional_params": {
    "temperature": 0.7,
    "max_tokens": 500
  }
}
```

### Convenience Endpoints (Not in LiteLLM Spec)

#### `GET /health`

Health check endpoint.

#### `GET /prompts`

List all available prompts.

#### `GET /prompts/{prompt_id}/variables`

Get all variables used in a prompt template.

#### `POST /prompts`

Create a new prompt (in-memory only, for testing).

## Example: Full Integration Test

### 1. Start the Mock Server

```bash
python mock_prompt_management_server.py
```

### 2. Test with Python

```python
from litellm import completion

# The completion will:
# 1. Fetch the prompt from your API
# 2. Replace {domain} with "machine learning"
# 3. Replace {task} with "building a recommendation system"
# 4. Merge with your messages
# 5. Use the model and params from the prompt

response = completion(
    model="gpt-4",
    prompt_id="hello-world-prompt",
    prompt_variables={
        "domain": "machine learning",
        "task": "building a recommendation system"
    },
    messages=[
        {"role": "user", "content": "I have user behavior data from the past year."}
    ],
    # Configure the generic prompt manager
    generic_prompt_config={
        "api_base": "http://localhost:8080",
        "api_key": "test-token-12345",
    }
)

print(response.choices[0].message.content)
```

## Customization

### Adding New Prompts

Edit the `PROMPTS_DB` dictionary in `mock_prompt_management_server.py`:

```python
PROMPTS_DB = {
    "my-custom-prompt": {
        "prompt_id": "my-custom-prompt",
        "prompt_template": [
            {
                "role": "system",
                "content": "You are a {role}."
            },
            {
                "role": "user",
                "content": "{user_input}"
            }
        ],
        "prompt_template_model": "gpt-4",
        "prompt_template_optional_params": {
            "temperature": 0.8,
            "max_tokens": 1000
        }
    }
}
```

### Using a Database

Replace the `PROMPTS_DB` dictionary with database queries:

```python
@app.get("/beta/litellm_prompt_management")
async def get_prompt(prompt_id: str):
    # Fetch from database
    prompt = await db.prompts.find_one({"prompt_id": prompt_id})
    
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return PromptResponse(**prompt)
```

### Adding Access Control

Use the custom query parameters for access control:

```python
@app.get("/beta/litellm_prompt_management")
async def get_prompt(
    prompt_id: str,
    project_name: Optional[str] = None,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None)
):
    token = verify_api_key(authorization)
    
    # Check if user has access to this project
    if not has_project_access(token, project_name):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Fetch and return prompt
    ...
```

## Production Considerations

Before deploying to production:

1. **Use a real database** instead of in-memory storage
2. **Implement proper authentication** with JWT tokens or API keys
3. **Add rate limiting** to prevent abuse
4. **Use HTTPS** for encrypted communication
5. **Add logging and monitoring** for observability
6. **Implement caching** for frequently accessed prompts
7. **Add versioning** for prompt management
8. **Implement access control** based on teams/users
9. **Add input validation** for all parameters
10. **Use environment variables** for configuration

## Related Documentation

- [Generic Prompt Management API Documentation](https://docs.litellm.ai/docs/adding_provider/generic_prompt_management_api)
- [LiteLLM Prompt Management](https://docs.litellm.ai/docs/proxy/prompt_management)
- [Generic Guardrail API](https://docs.litellm.ai/docs/adding_provider/generic_guardrail_api)

## Questions?

This is a reference implementation for the LiteLLM Generic Prompt Management API. For questions or issues, please open an issue on the [LiteLLM GitHub repository](https://github.com/BerriAI/litellm).

