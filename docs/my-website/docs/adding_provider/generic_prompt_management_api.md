# [BETA] Generic Prompt Management API - Integrate Without a PR

## The Problem

As a prompt management provider, integrating with LiteLLM traditionally requires:
- Making a PR to the LiteLLM repository
- Waiting for review and merge
- Maintaining provider-specific code in LiteLLM's codebase
- Updating the integration for changes to your API

## The Solution

The **Generic Prompt Management API** lets you integrate with LiteLLM **instantly** by implementing a simple API endpoint. No PR required.

### Key Benefits

1. **No PR Needed** - Deploy and integrate immediately
3. **Simple Contract** - One GET endpoint, standard JSON response
4. **Variable Substitution** - Support for prompt variables with `{variable}` syntax
5. **Custom Parameters** - Pass provider-specific query params via config
6. **Full Control** - You own and maintain your prompt management API
7. **Model & Parameters Override** - Optionally override model and parameters from your prompts

## Get Started in 3 Steps

### Step 1: Configure LiteLLM

Add to your `config.yaml`:

```yaml
prompts:
  - prompt_id: "simple_prompt"
    litellm_params:
      prompt_integration: "generic_prompt_management"
      api_base: http://localhost:8080
      api_key: os.environ/YOUR_API_KEY
```

### Step 2: Implement Your API Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/beta/litellm_prompt_management")
async def get_prompt(prompt_id: str):
    return {
        "prompt_id": prompt_id,
        "prompt_template": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Help me with {task}"}
        ],
        "prompt_template_model": "gpt-4",
        "prompt_template_optional_params": {"temperature": 0.7}
    }
```

### Step 3: Use in Your App

```python
from litellm import completion

response = completion(
    model="gpt-4",
    prompt_id="simple_prompt",
    prompt_variables={"task": "data analysis"},
    messages=[{"role": "user", "content": "I have sales data"}]
)
```

That's it! LiteLLM fetches your prompt, applies variables, and makes the request

## API Contract

### Endpoint

Implement `GET /beta/litellm_prompt_management`

### Request Format

Your endpoint will receive a GET request with query parameters:

```
GET /beta/litellm_prompt_management?prompt_id={prompt_id}&{custom_params}
```

**Query Parameters:**
- `prompt_id` (required): The ID of the prompt to fetch
- Custom parameters: Any additional parameters you configured in `provider_specific_query_params`

**Example:**
```
GET /beta/litellm_prompt_management?prompt_id=hello-world-prompt-2bac&project_name=litellm&slug=hello-world-prompt-2bac
```

### Response Format

```json
{
  "prompt_id": "hello-world-prompt-2bac",
  "prompt_template": [
    {
      "role": "system",
      "content": "You are a helpful assistant specialized in {domain}."
    },
    {
      "role": "user",
      "content": "Help me with {task}"
    }
  ],
  "prompt_template_model": "gpt-4",
  "prompt_template_optional_params": {
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 0.9
  }
}
```

**Response Fields:**
- `prompt_id` (string, required): The ID of the prompt
- `prompt_template` (array, required): Array of OpenAI-format messages with optional `{variable}` placeholders
- `prompt_template_model` (string, optional): Model to use for this prompt (overrides client model unless `ignore_prompt_manager_model: true`)
- `prompt_template_optional_params` (object, optional): Additional parameters like temperature, max_tokens, etc. (merged with client params unless `ignore_prompt_manager_optional_params: true`)

## LiteLLM Configuration

Add to `config.yaml`:

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

prompts:
  - prompt_id: "simple_prompt"
    litellm_params:
      prompt_integration: "generic_prompt_management"
      provider_specific_query_params:
        project_name: litellm
        slug: hello-world-prompt-2bac
      api_base: http://localhost:8080
      api_key: os.environ/YOUR_PROMPT_API_KEY  # optional
      ignore_prompt_manager_model: true  # optional, keep client's model
      ignore_prompt_manager_optional_params: true  # optional, don't merge prompt manager's params (e.g. temperature, max_tokens, etc.)
```

### Configuration Parameters

- `prompt_integration`: Must be `"generic_prompt_management"`
- `provider_specific_query_params`: Custom query parameters sent to your API (optional)
- `api_base`: Base URL of your prompt management API
- `api_key`: Optional API key for authentication (sent as `Bearer` token)
- `ignore_prompt_manager_model`: If `true`, use the model specified by client instead of prompt's model (default: `false`)
- `ignore_prompt_manager_optional_params`: If `true`, don't merge prompt's optional params with client params (default: `false`)

## Usage

### Using with LiteLLM SDK

**Basic usage with prompt ID:**

```python
from litellm import completion

response = completion(
    model="gpt-4",
    prompt_id="simple_prompt",
    messages=[{"role": "user", "content": "Additional message"}]
)
```

**With prompt variables:**

```python
response = completion(
    model="gpt-4",
    prompt_id="simple_prompt",
    prompt_variables={
        "domain": "data science",
        "task": "analyzing customer churn"
    },
    messages=[{"role": "user", "content": "Please provide a detailed analysis"}]
)
```

The prompt template will have `{domain}` replaced with "data science" and `{task}` replaced with "analyzing customer churn".

### Using with LiteLLM Proxy

**1. Start the proxy with your config:**

```bash
litellm --config /path/to/config.yaml
```

**2. Make requests with prompt_id:**

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4",
    "prompt_id": "simple_prompt",
    "prompt_variables": {
      "domain": "healthcare",
      "task": "patient risk assessment"
    },
    "messages": [
      {"role": "user", "content": "Analyze the following data..."}
    ]
  }'
```

**3. Using with OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Analyze the data"}
    ],
    extra_body={
        "prompt_id": "simple_prompt",
        "prompt_variables": {
            "domain": "finance",
            "task": "fraud detection"
        }
    }
)
```

## Implementation Example

See [mock_prompt_management_server.py](https://github.com/BerriAI/litellm/blob/main/cookbook/mock_prompt_management_server/mock_prompt_management_server.py) for a complete reference implementation with multiple example prompts, authentication, and convenience endpoints.

**Minimal FastAPI example:**

```python
from fastapi import FastAPI, HTTPException, Header
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

app = FastAPI()

# In-memory prompt storage (replace with your database)
PROMPTS = {
    "hello-world-prompt": {
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
    },
    "code-review-prompt": {
        "prompt_id": "code-review-prompt",
        "prompt_template": [
            {
                "role": "system",
                "content": "You are an expert code reviewer. Review code for {language}."
            },
            {
                "role": "user",
                "content": "Review the following code:\n\n{code}"
            }
        ],
        "prompt_template_model": "gpt-4-turbo",
        "prompt_template_optional_params": {
            "temperature": 0.3,
            "max_tokens": 1000
        }
    }
}

class PromptResponse(BaseModel):
    prompt_id: str
    prompt_template: List[Dict[str, str]]
    prompt_template_model: Optional[str] = None
    prompt_template_optional_params: Optional[Dict[str, Any]] = None

@app.get("/beta/litellm_prompt_management", response_model=PromptResponse)
async def get_prompt(
    prompt_id: str,
    authorization: Optional[str] = Header(None),
    project_name: Optional[str] = None,
    slug: Optional[str] = None,
):
    """
    Get a prompt by ID with optional filtering by project_name and slug.
    
    Args:
        prompt_id: The ID of the prompt to fetch
        authorization: Optional Bearer token for authentication
        project_name: Optional project name filter
        slug: Optional slug filter
    """
    
    # Optional: Validate authorization
    if authorization:
        token = authorization.replace("Bearer ", "")
        # Validate your token here
        if not is_valid_token(token):
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Optional: Apply additional filtering based on custom params
    if project_name or slug:
        # You can use these parameters to filter or validate access
        # For example, check if the user has access to this project
        pass
    
    # Fetch the prompt from your storage
    if prompt_id not in PROMPTS:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt '{prompt_id}' not found"
        )
    
    prompt_data = PROMPTS[prompt_id]
    
    return PromptResponse(**prompt_data)

def is_valid_token(token: str) -> bool:
    """Validate API token - implement your logic here"""
    # Example: Check against your database or secret store
    valid_tokens = ["your-secret-token", "another-valid-token"]
    return token in valid_tokens

# Optional: Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Optional: List all prompts endpoint
@app.get("/prompts")
async def list_prompts(authorization: Optional[str] = Header(None)):
    """List all available prompts"""
    if authorization:
        token = authorization.replace("Bearer ", "")
        if not is_valid_token(token):
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {
        "prompts": [
            {"prompt_id": pid, "model": p.get("prompt_template_model")}
            for pid, p in PROMPTS.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Running the Example Server

1. Install dependencies:
```bash
pip install fastapi uvicorn
```

2. Save the code above to `prompt_server.py`

3. Run the server:
```bash
python prompt_server.py
```

4. Test the endpoint:
```bash
curl "http://localhost:8080/beta/litellm_prompt_management?prompt_id=hello-world-prompt&project_name=litellm&slug=hello-world-prompt-2bac"
```

Expected response:
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

## Advanced Features

### Variable Substitution

LiteLLM automatically substitutes variables in your prompt templates using the `{variable}` syntax. Both `{variable}` and `{{variable}}` formats are supported.

**Example prompt template:**
```json
{
  "prompt_template": [
    {
      "role": "system",
      "content": "You are an expert in {domain} with {years} years of experience."
    }
  ]
}
```

**Client request:**
```python
completion(
    model="gpt-4",
    prompt_id="expert_prompt",
    prompt_variables={
        "domain": "machine learning",
        "years": "10"
    }
)
```

**Result:**
```
"You are an expert in machine learning with 10 years of experience."
```

### Caching

LiteLLM automatically caches fetched prompts in memory. The cache key includes:
- `prompt_id`
- `prompt_label` (if provided)
- `prompt_version` (if provided)

This means your API endpoint is only called once per unique prompt configuration.

### Model Override Behavior

**Default behavior (without `ignore_prompt_manager_model`):**
```yaml
prompts:
  - prompt_id: "my_prompt"
    litellm_params:
      prompt_integration: "generic_prompt_management"
      api_base: http://localhost:8080
```

If your API returns `"prompt_template_model": "gpt-4"`, LiteLLM will use `gpt-4` regardless of what the client specified.

**With `ignore_prompt_manager_model: true`:**
```yaml
prompts:
  - prompt_id: "my_prompt"
    litellm_params:
      prompt_integration: "generic_prompt_management"
      api_base: http://localhost:8080
      ignore_prompt_manager_model: true
```

LiteLLM will use the model specified by the client, ignoring the prompt's model.

### Parameter Merging Behavior

**Default behavior (without `ignore_prompt_manager_optional_params`):**

Client params are merged with prompt params, with prompt params taking precedence:
```python
# Prompt returns: {"temperature": 0.7, "max_tokens": 500}
# Client sends: {"temperature": 0.9, "top_p": 0.95}
# Final params: {"temperature": 0.7, "max_tokens": 500, "top_p": 0.95}
```

**With `ignore_prompt_manager_optional_params: true`:**

Only client params are used:
```python
# Prompt returns: {"temperature": 0.7, "max_tokens": 500}
# Client sends: {"temperature": 0.9, "top_p": 0.95}
# Final params: {"temperature": 0.9, "top_p": 0.95}
```

## Security Considerations

1. **Authentication**: Use the `api_key` parameter to secure your prompt management API
2. **Authorization**: Implement team/user-based access control using the custom query parameters
3. **Rate Limiting**: Add rate limiting to prevent abuse of your API
4. **Input Validation**: Validate all query parameters before processing
5. **HTTPS**: Always use HTTPS in production for encrypted communication
6. **Secrets**: Store API keys in environment variables, not in config files

## Use Cases

✅ **Use Generic Prompt Management API when:**
- You want instant integration without waiting for PRs
- You maintain your own prompt management service
- You need full control over prompt versioning and updates
- You want to build custom prompt management features
- You need to integrate with your internal systems

✅ **Common scenarios:**
- Internal prompt management system for your organization
- Multi-tenant prompt management with team-based access control
- A/B testing different prompt versions
- Prompt experimentation and analytics
- Integration with existing prompt engineering workflows

## When to Use This

✅ **Use Generic Prompt Management API when:**
- You want instant integration without waiting for PRs
- You maintain your own prompt management service
- You need full control over updates and features
- You want custom prompt storage and versioning logic

❌ **Make a PR when:**
- You want deeper integration with LiteLLM internals
- Your integration requires complex LiteLLM-specific logic
- You want to be featured as a built-in provider
- You're building a reusable integration for the community

## Troubleshooting

### Prompt not found
- Verify the `prompt_id` matches exactly (case-sensitive)
- Check that your API endpoint is accessible from LiteLLM
- Verify authentication if using `api_key`

### Variables not substituted
- Ensure variables use `{variable}` or `{{variable}}` syntax
- Check that variable names in `prompt_variables` match template exactly
- Variables are case-sensitive

### Model not being overridden
- Check if `ignore_prompt_manager_model: true` is set in config
- Verify your API is returning `prompt_template_model` in the response

### Parameters not being applied
- Check if `ignore_prompt_manager_optional_params: true` is set
- Verify your API is returning `prompt_template_optional_params`
- Ensure parameter names match OpenAI's parameter names

## Questions?

This is a **beta API**. We're actively improving it based on feedback. Open an issue or PR if you need additional capabilities.

## Related Documentation

- [Prompt Management Overview](../proxy/prompt_management.md)
- [Generic Guardrail API](./generic_guardrail_api.md)
- [LiteLLM Proxy Setup](../proxy/quick_start.md)

