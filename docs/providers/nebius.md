# Nebius AI

[Nebius AI](https://nebius.ai/) offers a cloud platform with AI services, including an API compatible with the OpenAI API. This document covers how to use Nebius AI with LiteLLM.

## API Key Setup

You can set your API key in the environment variables:

```bash
export NEBIUS_API_KEY=your_api_key
export NEBIUS_API_BASE=https://api.studio.nebius.ai/v1  # Optional, this is the default
```

Alternatively, you can pass them directly in your code:

```python
from litellm import completion

response = completion(
    model="nebius/yandex-yati",  # Example model, replace with actual model name
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    api_key="your_nebius_api_key",
    api_base="https://api.studio.nebius.ai/v1"  # Optional, this is the default
)
```

## Usage

### Chat Completions

```python
from litellm import completion

# Using environment variables for API key
response = completion(
    model="nebius/Qwen/Qwen3-30B-A3B-fast",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
)

print(response.choices[0].message.content)
```

### Streaming Response

```python
from litellm import completion

response = completion(
    model="nebius/Qwen/Qwen3-30B-A3B-fast",
    messages=[{"role": "user", "content": "Write a short story about AI"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
print()
```

### Embeddings

```python
from litellm import embedding

response = embedding(
    model="nebius/BAAI/bge-en-icl",
    input=["The food was delicious", "The service was excellent"]
)

print(response.data[0].embedding)  # First embedding vector
```

## Supported Models

Please replace the placeholder model names with actual Nebius AI models in your implementation. 

### Chat Models
- Add your supported chat models here (e.g., yandex-yati)

### Embedding Models
- Add your supported embedding models here (e.g., sentence-transformers)

## Supported Parameters

The Nebius provider supports the following parameters:

### Chat Completion Parameters

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| frequency_penalty | number | Penalizes new tokens based on their frequency in the text |
| function_call | string/object | Controls how the model calls functions |
| functions | array | List of functions for which the model may generate JSON inputs |
| logit_bias | map | Modifies the likelihood of specified tokens |
| max_tokens | integer | Maximum number of tokens to generate |
| n | integer | Number of completions to generate |
| presence_penalty | number | Penalizes tokens based on if they appear in the text so far |
| response_format | object | Format of the response, e.g., {"type": "json"} |
| seed | integer | Sampling seed for deterministic results |
| stop | string/array | Sequences where the API will stop generating tokens |
| stream | boolean | Whether to stream the response |
| temperature | number | Controls randomness (0-2) |
| top_p | number | Controls nucleus sampling |
| tool_choice | string/object | Controls which (if any) function to call |
| tools | array | List of tools the model can use |
| user | string | User identifier |

### Embedding Parameters

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| input | string/array | Text to embed |
| user | string | User identifier |

## Error Handling

The integration uses the standard LiteLLM error handling. Common errors include:

- **Authentication Error**: Check your API key
- **Model Not Found**: Ensure you're using a valid model name
- **Rate Limit Error**: You've exceeded your rate limits
- **Timeout Error**: Request took too long to complete

## Example Configuration

To set Nebius as your default provider:

```python
import litellm

litellm.set_keys(nebius_key="your_api_key")

# Optional custom base URL
litellm.api_base = "https://api.studio.nebius.ai/v1"

# Now you can call without specifying the provider
response = litellm.completion(
    model="nebius/Qwen/Qwen3-30B-A3B-fast",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Router Configuration

You can include Nebius in your LiteLLM router:

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "nebius-chat",
            "litellm_params": {
                "model": "nebius/Qwen/Qwen3-30B-A3B-fast",
                "api_key": "your_nebius_api_key",
                "api_base": "https://api.studio.nebius.ai/v1"
            }
        }
    ]
)

response = router.completion(
    model="nebius-chat",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Fallback Configuration

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "my-nebius",
            "litellm_params": {
                "model": "nebius/Qwen/Qwen3-30B-A3B-fast",
                "api_key": "your_nebius_api_key"
            }
        },
        {
            "model_name": "backup-model",
            "litellm_params": {
                "model": "meta-llama/Llama-3.3-70B-Instruct-fast",
                "api_key": "your_openai_api_key"
            }
        }
    ],
    fallbacks=[
        {
            "my-nebius": ["backup-model"]
        }
    ]
)
```

## Troubleshooting

If you encounter issues with the Nebius integration, try the following:

1. Ensure your API key is correct
2. Verify your API base URL
3. Check model availability in your Nebius account
4. Enable debug mode for more information:

```python
import litellm
litellm.set_verbose = True
```

## Additional Resources

- [Nebius AI Documentation](https://nebius.ai/docs)
- [LiteLLM GitHub Repository](https://github.com/BerriAI/litellm) 