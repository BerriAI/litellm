import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI Moderation

## Overview

| Property | Details |
|-------|-------|
| Description | Use OpenAI's built-in Moderation API to detect and block harmful content including hate speech, harassment, self-harm, sexual content, and violence. |
| Provider | [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation) |
| Supported Actions | `BLOCK` (raises HTTP 400 exception when violations detected) |
| Supported Modes | `pre_call`, `during_call`, `post_call` |
| Streaming Support | ✅ Full support for streaming responses |
| API Requirements | OpenAI API key |

## Quick Start

### 1. Define Guardrails on your LiteLLM config.yaml 

Define your guardrails under the `guardrails` section:

<Tabs>
<TabItem value="config" label="Config.yaml">

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "openai-moderation-pre"
    litellm_params:
      guardrail: openai_moderation
      mode: "pre_call"
      api_key: os.environ/OPENAI_API_KEY  # Optional if already set globally
      model: "omni-moderation-latest"     # Optional, defaults to omni-moderation-latest
      api_base: "https://api.openai.com/v1"  # Optional, defaults to OpenAI API
```

#### Supported values for `mode`

- `pre_call` Run **before** LLM call, on **user input**
- `during_call` Run **during** LLM call, on **user input**. Same as `pre_call` but runs in parallel as LLM call. Response not returned until guardrail check completes.
- `post_call` Run **after** LLM call, on **LLM response**

#### Supported OpenAI Moderation Models

- `omni-moderation-latest` (default) - Latest multimodal moderation model
- `text-moderation-latest` - Latest text-only moderation model

</TabItem>

<TabItem value="env" label="Environment Variables">

Set your OpenAI API key:

```bash title="Setup Environment Variables"
export OPENAI_API_KEY="your-openai-api-key"
```

</TabItem>
</Tabs>

### 2. Start LiteLLM Gateway 

```shell
litellm --config config.yaml --detailed_debug
```

### 3. Test request 

<Tabs>
<TabItem label="Blocked Request" value="blocked">

Expect this to fail since the request contains harmful content:

```shell
curl -i http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "I hate all people and want to hurt them"}
    ],
    "guardrails": ["openai-moderation-pre"]
  }'
```

Expected response on failure:

```json
{
  "error": {
    "message": {
      "error": "Violated OpenAI moderation policy",
      "moderation_result": {
        "violated_categories": ["hate", "violence"],
        "category_scores": {
          "hate": 0.95,
          "violence": 0.87,
          "harassment": 0.12,
          "self-harm": 0.01,
          "sexual": 0.02
        }
      }
    },
    "type": "None",
    "param": "None", 
    "code": "400"
  }
}
```

</TabItem>

<TabItem label="Successful Call" value="allowed">

```shell
curl -i http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "guardrails": ["openai-moderation-pre"]
  }'
```

Expected response:

```json
{
  "id": "chatcmpl-4a1c1a4a-3e1d-4fa4-ae25-7ebe84c9a9a2",
  "created": 1741082354,
  "model": "gpt-4",
  "object": "chat.completion",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "The capital of France is Paris.",
        "role": "assistant"
      }
    }
  ],
  "usage": {
    "completion_tokens": 8,
    "prompt_tokens": 13,
    "total_tokens": 21
  }
}
```

</TabItem>
</Tabs>

## Advanced Configuration

### Multiple Guardrails for Input and Output

You can configure separate guardrails for user input and LLM responses:

```yaml showLineNumbers title="Multiple Guardrails Config"
guardrails:
  - guardrail_name: "openai-moderation-input" 
    litellm_params:
      guardrail: openai_moderation
      mode: "pre_call"
      api_key: os.environ/OPENAI_API_KEY
      
  - guardrail_name: "openai-moderation-output"
    litellm_params:
      guardrail: openai_moderation
      mode: "post_call" 
      api_key: os.environ/OPENAI_API_KEY
```

### Custom API Configuration

Configure custom OpenAI API endpoints or different models:

```yaml showLineNumbers title="Custom API Config"
guardrails:
  - guardrail_name: "openai-moderation-custom"
    litellm_params:
      guardrail: openai_moderation
      mode: "pre_call"
      api_key: os.environ/OPENAI_API_KEY
      api_base: "https://your-custom-openai-endpoint.com/v1"
      model: "text-moderation-latest"
```

## Streaming Support

The OpenAI Moderation guardrail fully supports streaming responses. When used in `post_call` mode, it will:

1. Collect all streaming chunks
2. Assemble the complete response
3. Apply moderation to the full content
4. Block the entire stream if violations are detected
5. Return the original stream if content is safe

```yaml showLineNumbers title="Streaming Config"
guardrails:
  - guardrail_name: "openai-moderation-streaming"
    litellm_params:
      guardrail: openai_moderation
      mode: "post_call"  # Works with streaming responses
      api_key: os.environ/OPENAI_API_KEY
```

## Content Categories

The OpenAI Moderation API detects the following categories of harmful content:

| Category | Description |
|----------|-------------|
| `hate` | Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste |
| `harassment` | Content that harasses, bullies, or intimidates an individual |
| `self-harm` | Content that promotes, encourages, or depicts acts of self-harm |
| `sexual` | Content meant to arouse sexual excitement or promote sexual services |
| `violence` | Content that depicts death, violence, or physical injury |

Each category is evaluated with both a boolean flag and a confidence score (0.0 to 1.0).

## Error Handling

When content violates OpenAI's moderation policy:

- **HTTP Status**: 400 Bad Request
- **Error Type**: `HTTPException`
- **Error Details**: Includes violated categories and confidence scores
- **Behavior**: Request is immediately blocked

## Best Practices

### 1. Use Pre-call for User Input

```yaml
guardrails:
  - guardrail_name: "input-moderation"
    litellm_params:
      guardrail: openai_moderation
      mode: "pre_call"  # Block harmful user inputs early
```

### 2. Use Post-call for LLM Responses

```yaml
guardrails:
  - guardrail_name: "output-moderation"
    litellm_params:
      guardrail: openai_moderation  
      mode: "post_call"  # Ensure LLM responses are safe
```

### 3. Combine with Other Guardrails

```yaml
guardrails:
  - guardrail_name: "openai-moderation"
    litellm_params:
      guardrail: openai_moderation
      mode: "pre_call"
      
  - guardrail_name: "custom-pii-detection"
    litellm_params:
      guardrail: presidio
      mode: "pre_call"
```

## Troubleshooting

### Common Issues

1. **Invalid API Key**: Ensure your OpenAI API key is correctly set
   ```bash
   export OPENAI_API_KEY="sk-your-actual-key"
   ```

2. **Rate Limiting**: OpenAI Moderation API has rate limits. Monitor usage in high-volume scenarios.

3. **Network Issues**: Verify connectivity to OpenAI's API endpoints.

### Debug Mode

Enable detailed logging to troubleshoot issues:

```shell
litellm --config config.yaml --detailed_debug
```

Look for logs starting with `OpenAI Moderation:` to trace guardrail execution.

## API Costs

The OpenAI Moderation API is **free to use** for content policy compliance. This makes it a cost-effective guardrail option compared to other commercial moderation services.

## Need Help?

For additional support:
- Check the [OpenAI Moderation API documentation](https://platform.openai.com/docs/guides/moderation)
- Review [LiteLLM Guardrails documentation](./quick_start)
- Join our [Discord community](https://discord.gg/wuPM9dRgDw) 