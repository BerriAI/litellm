import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create Pass Through Endpoints 

Route requests from your LiteLLM proxy to any external API. Perfect for custom models, image generation APIs, or any service you want to proxy through LiteLLM.

**Key Benefits:**
- Onboard third-party endpoints like Bria API and Mistral OCR
- Set custom pricing per request
- Proxy Admins don't need to give developers api keys to upstream llm providers like Bria, Mistral OCR, etc.
- Maintain centralized authentication, spend tracking, budgeting

## Quick Start with UI (Recommended)

The easiest way to create pass through endpoints is through the LiteLLM UI. In this example, we'll onboard the [Bria API](https://docs.bria.ai/image-generation/endpoints/text-to-image-base) and set a cost per request.

### Step 1: Create Route Mappings

To create a pass through endpoint:

1. Navigate to the LiteLLM Proxy UI
2. Go to the `Models + Endpoints` tab
3. Click on `Pass Through Endpoints`
4. Click "Add Pass Through Endpoint"
5. Enter the following details:

**Required Fields:**
- `Path Prefix`: The route clients will use when calling LiteLLM Proxy (e.g., `/bria`, `/mistral-ocr`)
- `Target URL`: The URL where requests will be forwarded

<Image 
  img={require('../../img/pt_1.png')}
  style={{width: '60%', display: 'block', margin: '2rem auto'}}
/>

**Route Mapping Example:**

The above configuration creates these route mappings:

| LiteLLM Proxy Route | Target URL |
|-------------------|------------|
| `/bria` | `https://engine.prod.bria-api.com` |
| `/bria/v1/text-to-image/base/model` | `https://engine.prod.bria-api.com/v1/text-to-image/base/model` |
| `/bria/v1/enhance_image` | `https://engine.prod.bria-api.com/v1/enhance_image` |
| `/bria/<any-sub-path>` | `https://engine.prod.bria-api.com/<any-sub-path>` |

:::info
All routes are prefixed with your LiteLLM proxy base URL: `https://<litellm-proxy-base-url>`
:::

### Step 2: Configure Headers and Pricing

Configure the required authentication and pricing:

**Authentication Setup:**
- The Bria API requires an `api_token` header
- Enter your Bria API key as the value for the `api_token` header

**Pricing Configuration:**
- Set a cost per request (e.g., $12.00 in this example)
- This enables cost tracking and billing for your users

<Image 
  img={require('../../img/pt_2.png')}
  style={{width: '60%', display: 'block', margin: '2rem auto'}}
/>

### Step 3: Save Your Endpoint 

Once you've completed the configuration:
1. Review your settings
2. Click "Add Pass Through Endpoint"
3. Your endpoint will be created and immediately available

### Step 4: Test Your Endpoint

Verify your setup by making a test request to the Bria API through your LiteLLM Proxy:

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

**Expected Response:**
If everything is configured correctly, you should receive a response from the Bria API containing the generated image data.

---

## Config.yaml Setup

You can also create pass through endpoints using the `config.yaml` file. Here's how to add a `/v1/rerank` route that forwards to Cohere's API:

### Example Configuration

```yaml
general_settings:
  master_key: sk-1234
  pass_through_endpoints:
    - path: "/v1/rerank"                                  # Route on LiteLLM Proxy
      target: "https://api.cohere.com/v1/rerank"          # Target endpoint
      auth: true                                          # Require virtual key (Enterprise)
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

## Authentication & Access Control✨ 

Control who can access your pass-through endpoints using LiteLLM's virtual key authentication.

:::info

✨ This is an Enterprise only feature [Get Started with Enterprise here](https://www.litellm.ai/#pricing)

:::

### Enable Authentication on Pass-Through Endpoints

When `auth: true` is set, the endpoint requires a valid LiteLLM virtual key. This enables centralized access control, spend tracking, and budgeting.

#### Config.yaml Example

```yaml
general_settings:
  master_key: sk-1234
  pass_through_endpoints:
    - path: "/v1/rerank"
      target: "https://api.cohere.com/v1/rerank"
      auth: true  # Require LiteLLM virtual key authentication
      headers:
        Authorization: "bearer os.environ/COHERE_API_KEY"
      forward_headers: true
```

#### Making Authenticated Requests

Once `auth: true` is enabled, clients must include a valid LiteLLM virtual key in their requests:

```shell
curl --request POST \
  --url http://localhost:4000/v1/rerank \
  --header 'Authorization: Bearer <your-litellm-virtual-key>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "rerank-english-v3.0",
    "query": "What is the capital of the United States?",
    "top_n": 3,
    "documents": ["Carson City is the capital city of the American state of Nevada."]
  }'
```

### Virtual Key Access to Pass-Through Endpoints

LiteLLM virtual keys can access pass-through endpoints in two ways:

#### 1. Automatic Access via `llm_api_routes`

Virtual keys with `llm_api_routes` in their `allowed_routes` automatically get access to **all** pass-through endpoints.

**Example: Generate a key with full pass-through access**

```shell
curl 'http://localhost:4000/key/generate' \
  --header 'Authorization: Bearer <master-key>' \
  --header 'Content-Type: application/json' \
  --data '{
    "allowed_routes": ["llm_api_routes"],
    "models": ["gpt-3.5-turbo"]
  }'
```

This key can access:
- All standard LLM endpoints (`/chat/completions`, `/embeddings`, etc.)
- **All pass-through endpoints** (including those from both config.yaml and UI)

#### 2. Specific Access via `allowed_routes`

To restrict access to specific pass-through endpoints, include the exact path in `allowed_routes`:

**Example: Generate a key with restricted pass-through access**

```shell
curl 'http://localhost:4000/key/generate' \
  --header 'Authorization: Bearer <master-key>' \
  --header 'Content-Type: application/json' \
  --data '{
    "allowed_routes": ["/v1/rerank", "/chat/completions"],
    "models": ["gpt-3.5-turbo"]
  }'
```

This key can only access:
- `/chat/completions` endpoint
- `/v1/rerank` pass-through endpoint (and no other pass-through endpoints)

#### 3. Specific Access via `allowed_passthrough_routes`

Use `allowed_passthrough_routes` to grant access **only** to pass-through endpoints without specifying LLM routes. This is useful when you want to restrict a key to pass-through endpoints only.

**Example: Generate a key with access to specific pass-through endpoints**

```shell
curl 'http://localhost:4000/key/generate' \
  --header 'Authorization: Bearer <master-key>' \
  --header 'Content-Type: application/json' \
  --data '{
    "allowed_passthrough_routes": ["/v1/rerank", "/bria/*"],
    "max_budget": 100
  }'
```

This key can only access:
- `/v1/rerank` pass-through endpoint
- Any routes under `/bria/` (e.g., `/bria/v1/text-to-image`, `/bria/v1/enhance_image`)
- **No access** to standard LLM routes like `/chat/completions` or `/embeddings`

:::note
If `allowed_routes` is specified, `allowed_passthrough_routes` is ignored. Use `allowed_passthrough_routes` when you want to restrict access to pass-through endpoints only, without needing to specify LLM routes.
:::

---

## Tutorial - Add Azure OpenAI Assistants API as a Pass Through Endpoint

In this video, we'll add the Azure OpenAI Assistants API as a pass through endpoint to LiteLLM Proxy.

<iframe width="840" height="500" src="https://www.loom.com/embed/12965cb299d24fc0bd7b6b413ab6d0ad" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

<br/>
<br/>


---

## Troubleshooting

### Common Issues

**Authentication Errors:**
- Verify API keys are correctly set in headers
- Ensure the target API accepts the provided authentication method

**Routing Issues:**
- Confirm the path prefix matches your request URL
- Verify the target URL is accessible
- Check for trailing slashes in configuration

**Response Errors:**
- Enable detailed debugging with `--detailed_debug`
- Check LiteLLM proxy logs for error details
- Verify the target API's expected request format

### Getting Help

[Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)

[Community Discord 💭](https://discord.gg/wuPM9dRgDw)

Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬

Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
