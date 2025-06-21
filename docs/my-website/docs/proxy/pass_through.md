import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create Pass Through Endpoints 

Route requests from your LiteLLM proxy to any external API. Perfect for custom models, image generation APIs, or any service you want to proxy through LiteLLM.

Onboard third-party endpoints like Bria API and Mistral OCR, set a cost per request, and give your developers access.

## Quick Start with UI (Recommended)

The easiest way to create pass through endpoints is through the LiteLLM UI. In this example, we'll onboard the [Bria API](https://docs.bria.ai/image-generation/endpoints/text-to-image-base) and set a cost per request.

### Step 1: Create Route Mappings

`path`: This is the route clients should use when calling LiteLLM Proxy.
`target`: This is the URL the request will be forwarded to.

<Image 
  img={require('../../img/pt_1.png')}
  style={{width: '60%', display: 'block', margin: '2rem auto'}}
/>

This creates the following route mappings:

- `https://<litellm-proxy-base-url>/bria` → `https://engine.prod.bria-api.com`
- `https://<litellm-proxy-base-url>/v1/text-to-image/base/model` → `https://engine.prod.bria-api.com/v1/text-to-image/base/model`
- `https://<litellm-proxy-base-url>/v1/enhance_image` → `https://engine.prod.bria-api.com/v1/enhance_image`

### Step 2: Configure Headers and Pricing

<Image 
  img={require('../../img/pt_2.png')}
  style={{width: '60%', display: 'block', margin: '2rem auto'}}
/>

For the Bria API, add the required header:
- `api_token: string`

### Step 3: Test Your Endpoint

Make a request to the Bria API through your LiteLLM Proxy:

```shell
curl -i -X POST \
  'http://localhost:4000/bria/v1/text-to-image/base/2.3' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <your-litellm-api-key>' \
  -d '{
    "prompt": "a book",
    "num_results": 2,
    "sync": true
  }'
```

---

## Config.yaml setup

You can also create pass through endpoints using the `config.yaml` file. Here's how to add a `/v1/rerank` route that forwards to Cohere's API:

### Example Configuration

```yaml
general_settings:
  master_key: sk-1234
  pass_through_endpoints:
    - path: "/v1/rerank"                                  # Route on LiteLLM Proxy
      target: "https://api.cohere.com/v1/rerank"          # Target endpoint
      headers:                                            # Headers to forward
        Authorization: "bearer os.environ/COHERE_API_KEY"
        content-type: application/json
        accept: application/json
      forward_headers: true                               # Forward all incoming headers
```

### Start and Test

1. **Start the proxy:**
   ```shell
   litellm --config config.yaml --detailed_debug
   ```

2. **Make a test request:**
   ```shell
   curl --request POST \
     --url http://localhost:4000/v1/rerank \
     --header 'accept: application/json' \
     --header 'content-type: application/json' \
     --data '{
       "model": "rerank-english-v3.0",
       "query": "What is the capital of the United States?",
       "top_n": 3,
       "documents": ["Carson City is the capital city of the American state of Nevada."]
     }'
   ```

### Expected Response
```json
{
  "id": "37103a5b-8cfb-48d3-87c7-da288bedd429",
  "results": [
    {
      "index": 2,
      "relevance_score": 0.999071
    }
  ],
  "meta": {
    "api_version": {"version": "1"},
    "billed_units": {"search_units": 1}
  }
}
```

---

## ✨ Enterprise Features

### Authentication & Rate Limiting

Enable LiteLLM authentication and rate limiting on pass through endpoints:

```yaml
general_settings:
  master_key: sk-1234
  pass_through_endpoints:
    - path: "/v1/rerank"
      target: "https://api.cohere.com/v1/rerank"
      auth: true                                          # Enable LiteLLM auth
      headers:
        Authorization: "bearer os.environ/COHERE_API_KEY"
        content-type: application/json
```

Test with LiteLLM key:
```shell
curl --request POST \
  --url http://localhost:4000/v1/rerank \
  --header 'Authorization: Bearer sk-1234' \
  --header 'content-type: application/json' \
  --data '{"model": "rerank-english-v3.0", "query": "test"}'
```

---

## Configuration Reference

### Complete Specification

```yaml
general_settings:
  pass_through_endpoints:
    - path: string                    # Route on LiteLLM Proxy Server
      target: string                  # Target URL for forwarding
      auth: boolean                   # Enable LiteLLM authentication (Enterprise)
      forward_headers: boolean        # Forward all incoming headers
      headers:                        # Custom headers to add
        Authorization: string         # Auth header for target API
        content-type: string         # Request content type
        accept: string               # Expected response format
        LANGFUSE_PUBLIC_KEY: string  # For Langfuse endpoints
        LANGFUSE_SECRET_KEY: string  # For Langfuse endpoints
        <custom-header>: string      # Any custom header
```

### Header Options
- **Authorization**: Authentication for the target API
- **content-type**: Request body format specification  
- **accept**: Expected response format
- **LANGFUSE_PUBLIC_KEY/SECRET_KEY**: For Langfuse integration
- **Custom headers**: Any additional key-value pairs

---

## Advanced: Custom Adapters

For complex integrations (like Anthropic/Bedrock clients), you can create custom adapters that translate between different API schemas.

### 1. Create an Adapter

```python
from litellm import adapter_completion
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.anthropic import AnthropicMessagesRequest, AnthropicResponse

class AnthropicAdapter(CustomLogger):
    def translate_completion_input_params(self, kwargs):
        """Translate Anthropic format to OpenAI format"""
        request_body = AnthropicMessagesRequest(**kwargs)
        return litellm.AnthropicConfig().translate_anthropic_to_openai(
            anthropic_message_request=request_body
        )

    def translate_completion_output_params(self, response):
        """Translate OpenAI response back to Anthropic format"""
        return litellm.AnthropicConfig().translate_openai_response_to_anthropic(
            response=response
        )

anthropic_adapter = AnthropicAdapter()
```

### 2. Configure the Endpoint

```yaml
model_list:
  - model_name: my-claude-endpoint
    litellm_params:
      model: gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  master_key: sk-1234
  pass_through_endpoints:
    - path: "/v1/messages"
      target: custom_callbacks.anthropic_adapter
      headers:
        litellm_user_api_key: "x-api-key"
```

### 3. Test Custom Endpoint

```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
  -H 'x-api-key: sk-1234' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'content-type: application/json' \
  -d '{
    "model": "my-claude-endpoint",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello, world"}]
  }'
```

---

Need help? Check out our [provider-specific parameters guide](../completion/provider_specific_params.md) for more advanced configurations.
