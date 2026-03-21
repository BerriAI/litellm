# Arize Phoenix Prompt Management Integration

This integration enables using prompt versions from Arize Phoenix with LiteLLM's completion function.

## Features

- Fetch prompt versions from Arize Phoenix API
- Workspace-based access control through Arize Phoenix permissions
- Mustache/Handlebars-style variable templating (`{{variable}}`)
- Support for multi-message chat templates
- Automatic model and parameter configuration from prompt metadata
- OpenAI and Anthropic provider parameter support

## Configuration

Configure Arize Phoenix access in your application:

```python
import litellm

# Configure Arize Phoenix access
# api_base should include your workspace, e.g., "https://app.phoenix.arize.com/s/your-workspace/v1"
api_key = "your-arize-phoenix-token"
api_base = "https://app.phoenix.arize.com/s/krrishdholakia/v1"
```

## Usage

### Basic Usage

```python
import litellm

# Use with completion
response = litellm.completion(
    model="arize/gpt-4o",
    prompt_id="UHJvbXB0VmVyc2lvbjox",  # Your prompt version ID
    prompt_variables={"question": "What is artificial intelligence?"},
    api_key="your-arize-phoenix-token",
    api_base="https://app.phoenix.arize.com/s/krrishdholakia/v1",
)

print(response.choices[0].message.content)
```

### With Additional Messages

You can also combine prompt templates with additional messages:

```python
response = litellm.completion(
    model="arize/gpt-4o",
    prompt_id="UHJvbXB0VmVyc2lvbjox",
    prompt_variables={"question": "Explain quantum computing"},
    api_key="your-arize-phoenix-token",
    api_base="https://app.phoenix.arize.com/s/krrishdholakia/v1",
    messages=[
        {"role": "user", "content": "Please keep your response under 100 words."}
    ],
)
```

### Direct Manager Usage

You can also use the prompt manager directly:

```python
from litellm.integrations.arize.arize_phoenix_prompt_manager import ArizePhoenixPromptManager

# Initialize the manager
manager = ArizePhoenixPromptManager(
    api_key="your-arize-phoenix-token",
    api_base="https://app.phoenix.arize.com/s/krrishdholakia/v1",
    prompt_id="UHJvbXB0VmVyc2lvbjox",
)

# Get rendered messages
messages, metadata = manager.get_prompt_template(
    prompt_id="UHJvbXB0VmVyc2lvbjox",
    prompt_variables={"question": "What is machine learning?"}
)

print("Rendered messages:", messages)
print("Metadata:", metadata)
```

## Prompt Format

Arize Phoenix prompts support the following structure:

```json
{
    "data": {
        "description": "A chatbot prompt",
        "model_provider": "OPENAI",
        "model_name": "gpt-4o",
        "template": {
            "type": "chat",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a chatbot"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "{{question}}"
                        }
                    ]
                }
            ]
        },
        "template_type": "CHAT",
        "template_format": "MUSTACHE",
        "invocation_parameters": {
            "type": "openai",
            "openai": {
                "temperature": 1.0
            }
        },
        "id": "UHJvbXB0VmVyc2lvbjox"
    }
}
```

### Variable Substitution

Variables in your prompt templates use Mustache/Handlebars syntax:
- `{{variable_name}}` - Simple variable substitution

Example:
```
Template: "Hello {{name}}, your order {{order_id}} is ready!"
Variables: {"name": "Alice", "order_id": "12345"}
Result: "Hello Alice, your order 12345 is ready!"
```

## API Reference

### ArizePhoenixPromptManager

Main class for managing Arize Phoenix prompts.

**Methods:**
- `get_prompt_template(prompt_id, prompt_variables)` - Get and render a prompt template
- `get_available_prompts()` - List available prompt IDs
- `reload_prompts()` - Reload prompts from Arize Phoenix

### ArizePhoenixClient

Low-level client for Arize Phoenix API.

**Methods:**
- `get_prompt_version(prompt_version_id)` - Fetch a prompt version
- `test_connection()` - Test API connection

## Error Handling

The integration provides detailed error messages:

- **404**: Prompt version not found
- **401**: Authentication failed (check your access token)
- **403**: Access denied (check workspace permissions)

Example:
```python
try:
    response = litellm.completion(
        model="arize/gpt-4o",
        prompt_id="invalid-id",
        arize_config=arize_config,
    )
except Exception as e:
    print(f"Error: {e}")
```

## Getting Your Prompt Version ID and API Base

1. Log in to Arize Phoenix
2. Navigate to your workspace
3. Go to Prompts section
4. Select a prompt version
5. The ID will be in the URL: `/s/{workspace}/v1/prompt_versions/{PROMPT_VERSION_ID}`

Your `api_base` should be: `https://app.phoenix.arize.com/s/{workspace}/v1`

For example:
- Workspace: `krrishdholakia`
- API Base: `https://app.phoenix.arize.com/s/krrishdholakia/v1`
- Prompt Version ID: `UHJvbXB0VmVyc2lvbjox`

You can also fetch it via API:
```bash
curl -L -X GET 'https://app.phoenix.arize.com/s/krrishdholakia/v1/prompt_versions/UHJvbXB0VmVyc2lvbjox' \
  -H 'Authorization: Bearer YOUR_TOKEN'
```

## Support

For issues or questions:
- LiteLLM Issues: https://github.com/BerriAI/litellm/issues
- Arize Phoenix Docs: https://docs.arize.com/phoenix

