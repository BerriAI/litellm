# Akto - LLM Traffic Monitoring & API Security

## What is Akto?

[Akto](https://www.akto.io/) is an API security platform that provides monitoring, testing, and guardrails for AI/ML workloads. For LLM applications, Akto ingests request/response traffic for security analysis, vulnerability detection, and compliance monitoring.

## Usage with LiteLLM Proxy (LLM Gateway)

**Step 1**: Create a `config.yaml` file and set `litellm_settings`: `success_callback`

```yaml
model_list:
  - model_name: gpt-5.4
    litellm_params:
      model: gpt-5.4

litellm_settings:
  success_callback: ["akto"]
  failure_callback: ["akto"]
```

**Step 2**: Set required environment variables

```shell
export AKTO_DATA_INGESTION_API_BASE="http://your-akto-instance:8080"
export AKTO_API_KEY="your-akto-api-key"

# Optional
export AKTO_ACCOUNT_ID="1000000"  # default: 1000000
export AKTO_VXLAN_ID="0"          # default: 0
```

**Step 3**: Start the proxy, make a test request

Start proxy

```shell
litellm --config config.yaml --debug
```

Test Request

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-5.4",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
}'
```

## What's Logged to Akto?

When LiteLLM logs to Akto, it sends the full HTTP transaction in Akto's MIRRORING format:

### For Every LLM Call
- **Request**: Messages, model, tools, tool calls
- **Response**: Full model response (choices, usage)
- **Headers**: All proxy request headers (sensitive headers like `Authorization`, `Cookie` are stripped)
- **Metadata**: User ID, team ID, API route, client IP
- **Status**: HTTP status code (200 for success, 500 for failures)
- **Timing**: Request timestamp

### For Errors
- **Status Code**: Extracted from the exception (e.g., 429 for rate limits, 500 for server errors)
- **Request Context**: The original request that caused the error

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AKTO_DATA_INGESTION_API_BASE` | Yes | Akto data ingestion API base URL |
| `AKTO_API_KEY` | Yes | Akto API key for authentication |
| `AKTO_ACCOUNT_ID` | No | Akto account ID (default: `1000000`) |
| `AKTO_VXLAN_ID` | No | Akto VXLAN ID (default: `0`) |

## Troubleshooting

### 1. Missing API Key
```
Error: Missing keys=['AKTO_DATA_INGESTION_API_BASE'] in environment.
```

Set your Akto environment variables:
```shell
export AKTO_DATA_INGESTION_API_BASE="http://your-akto-instance:8080"
export AKTO_API_KEY="your-api-key"
```

### 2. Events Not Appearing
- Check that your API key is correct
- Verify network connectivity to the Akto data ingestion service
- Check LiteLLM logs for `Akto logging error` or `Akto ingestion returned` warnings

### 3. Health Check
Verify the Akto integration is healthy:
```shell
curl 'http://localhost:4000/health/services?service=akto' \
  -H 'Authorization: Bearer your-litellm-key'
```
