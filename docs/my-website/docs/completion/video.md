import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Using Video Models

## Quick Start
Example passing images to a model 


## Proxy Config setup

Admins can configure file storage settings and retention policies in their LiteLLM config:

```yaml
model_list: 
  - model_name: vertex_ai/*
    litellm_params:
      model: vertex_ai/*
  - model_name: bedrock/*
    litellm_params:
      model: bedrock/*


files_settings:
  # Configure storage providers and retention  
    - custom_llm_provider: azure
      api_base: https://exampleopenaiendpoint-production.up.railway.app
      api_key: fake-key
      api_version: "2023-03-15-preview"
    - custom_llm_provider: openai
      api_key: os.environ/OPENAI_API_KEY
    - custom_llm_provider: bedrock
      api_key: os.environ/BEDROCK_API_KEY
      api_base: https://bedrock.us-east-1.amazonaws.com
      retention_period: 7
    - custom_llm_provider: vertex_ai
      bucket_name: my-vertex_ai-bucket
      retention_period: 7
```

This configuration:
- Sets up storage providers (GCS and S3) with their retention policies
- Configures provider-specific file endpoints for Azure and OpenAI
- Retention is enforced through bucket lifecycle rules
- All files uploaded will follow these retention settings

## 1. Local File and process on Vertex + Bedrock 

When uploading a local file, the process follows these steps:

1. Client makes a POST request to `/files` endpoint with the local file
2. Client MUST specify required storage location(s) based on intended model usage
3. LiteLLM uploads the file to specified storage locations
4. A file ID is returned to the client
5. Files have a 7-day retention policy (configured by admin)

```bash
# custom_llm_provider is required
curl https://api.litellm.ai/v1/files \
  -H "Authorization: Bearer sk-1234" \
  -F purpose="fine-tune" \
  -F file="@mydata.jsonl" \
  -F custom_llm_provider='["vertex_ai"]'  # Required: specify ["vertex_ai"], ["bedrock"], or ["vertex_ai", "bedrock"]
```

You can optionally specify storage locations:

```bash
curl https://api.litellm.ai/v1/files \
  -H "Authorization: Bearer sk-1234" \
  -F purpose="fine-tune" \
  -F file="@mydata.jsonl" \
  -F custom_llm_provider='["vertex_ai"]'  # or ["vertex_ai", "bedrock"]
```

After uploading, you can use the file_id in a /chat/completions request:

```bash
curl "https://api.litellm.ai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "bedrock/amazon.nova-pro-v1:0",
    "input": [
      {
        "role": "user",
        "content": [
          {
            "type": "input_file",
            "file_id": "file-6F2ksmvXxt4VdoqmHRw6kL"
          },
          {
            "type": "input_text",
            "text": "What is happening in this video?"
          }
        ]
      }
    ]
  }'
```

## 2. Existing file in S3 and process on Vertex + Bedrock 

For files already in S3, you can avoid downloading and re-uploading by using the `/upload` endpoint:

```bash
# custom_llm_provider is required
curl -X POST "https://api.litellm.ai/files" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "bedrock://your-bedrock-bucket/path/to/mydata.jsonl",
    "custom_llm_provider": ["vertex_ai"]  # Required: specify target storage location
  }'
```

This performs a direct copy from S3 to the specified storage location(s), which is more efficient than downloading and re-uploading.

After the S3 file is copied, you can use the returned file_id in your /chat/completions request:

```bash
curl "https://api.litellm.ai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "vertex_ai/gemini-pro-vision",
    "input": [
      {
        "role": "user",
        "content": [
          {
            "type": "input_file",
            "file_id": "file-8H3jsmwYyt6WepqnKRw9mN"
          },
          {
            "type": "input_text",
            "text": "Describe the main events in this video"
          }
        ]
      }
    ]
  }'
```


## Best Practices

**Storage Location Selection**: 
   - You MUST specify storage locations based on your model requirements
   - For Vertex AI models, specify `["vertex_ai"]`
   - For Bedrock models, specify `["bedrock"]`
   - For fallback scenarios, specify both `["vertex_ai", "bedrock"]`
   - Avoid unnecessary copies by selecting only required destinations
## Model Compatibility

This file handling approach is compatible with various model backends:
- Vertex AI
- AWS Bedrock
- vLLM (for video processing, using their multimodal input format)

