# LiteLLM Rust Gateway - API Documentation

This document provides comprehensive documentation for all API endpoints in the LiteLLM Rust Gateway.

## Authentication

All endpoints (except health checks) require authentication via Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.example.com/v1/...
```

The API key is validated against the `LiteLLM_VerificationToken` table and checked for:
- Expiration
- Budget limits
- Model access permissions
- Rate limits (RPM, TPM, parallel requests)

## Core LLM Routes

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint with full middleware support.

**Request:**
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 100
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

**Features:**
- Streaming support (SSE)
- Fallback routing across multiple deployments
- Circuit breaker for provider failures
- Automatic retries with exponential backoff
- Spend tracking and cost calculation
- Guardrails integration
- Callback execution (Langfuse, Datadog, Webhooks, Slack)

### POST /v1/messages

Anthropic Messages API endpoint with full middleware support.

**Request:**
```json
{
  "model": "claude-3-opus-20240229",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100
}
```

**Response:**
```json
{
  "id": "msg_123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello! How can I help you today?"
    }
  ],
  "model": "claude-3-opus-20240229",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 10,
    "output_tokens": 20
  }
}
```

**Features:**
- Same middleware as chat completions
- Per-key authentication
- Rate limiting and budget enforcement
- Spend tracking

### POST /v1/embeddings

OpenAI-compatible embeddings endpoint.

**Request:**
```json
{
  "model": "text-embedding-3-small",
  "input": "The text to embed",
  "encoding_format": "float"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023064255, -0.009327292, ...],
      "index": 0
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

**Features:**
- Full middleware support
- Spend tracking based on token count
- Rate limiting

### POST /v1/images/generations

OpenAI-compatible image generation endpoint.

**Request:**
```json
{
  "model": "dall-e-3",
  "prompt": "A cute baby sea otter",
  "n": 1,
  "size": "1024x1024"
}
```

**Response:**
```json
{
  "created": 1589478378,
  "data": [
    {
      "url": "https://...",
      "revised_prompt": "A cute baby sea otter floating on its back"
    }
  ]
}
```

**Features:**
- Full middleware support
- Guardrails for prompt validation
- Spend tracking

### POST /v1/images/edits

OpenAI-compatible image editing endpoint.

**Request:** (multipart/form-data)
```
file: (binary)
mask: (binary, optional)
prompt: "Add a sun hat"
model: "dall-e-2"
n: 1
size: "1024x1024"
```

**Response:**
```json
{
  "created": 1589478378,
  "data": [
    {
      "url": "https://..."
    }
  ]
}
```

**Features:**
- Multipart form data support
- Full middleware support
- Guardrails integration

### POST /v1/audio/speech

OpenAI-compatible text-to-speech endpoint.

**Request:**
```json
{
  "model": "tts-1",
  "input": "Hello, world!",
  "voice": "alloy",
  "response_format": "mp3"
}
```

**Response:** Binary audio data with appropriate Content-Type header.

**Features:**
- Full middleware support
- Multiple voice options
- Multiple output formats (mp3, opus, aac, flac)

### POST /v1/audio/transcriptions

OpenAI-compatible speech-to-text endpoint.

**Request:** (multipart/form-data)
```
file: (binary)
model: "whisper-1"
language: "en"
prompt: "Optional context"
response_format: "json"
temperature: 0.0
```

**Response:**
```json
{
  "text": "Hello, world!"
}
```

**Features:**
- Multipart form data support
- Full middleware support
- Multiple output formats (json, text, srt, verbose_json, vtt)

## Admin Routes

### GET /v1/models

List all available models.

**Request:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.example.com/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4",
      "object": "model",
      "created": 0,
      "owned_by": "openai",
      "litellm_params": {
        "model": "openai/gpt-4",
        "api_base": null
      }
    }
  ]
}
```

### GET /key/info

Retrieve information about an API key.

**Request:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" "https://api.example.com/key/info?key=HASHED_TOKEN"
```

**Query Parameters:**
- `key` (optional): Hashed token to query. If not provided, returns info for the authenticated key.

**Response:**
```json
{
  "token": "hashed_token",
  "key_name": "my-key",
  "key_alias": "production-key",
  "user_id": "user-123",
  "team_id": "team-456",
  "org_id": "org-789",
  "max_budget": 100.0,
  "spend": 25.50,
  "models": ["gpt-4", "claude-3"],
  "tpm_limit": 100000,
  "rpm_limit": 1000
}
```

### GET /spend/logs

Retrieve spend logs with filtering and pagination.

**Request:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" "https://api.example.com/spend/logs?start_time=2024-01-01&end_time=2024-01-31&user_id=user-123&limit=100&offset=0"
```

**Query Parameters:**
- `start_time` (optional): Start date filter (ISO 8601)
- `end_time` (optional): End date filter (ISO 8601)
- `user_id` (optional): Filter by user ID
- `team_id` (optional): Filter by team ID
- `model` (optional): Filter by model name
- `limit` (optional): Number of records to return (default: 100, max: 1000)
- `offset` (optional): Offset for pagination (default: 0)

**Response:**
```json
{
  "data": [
    {
      "request_id": "req-123",
      "call_type": "chat_completion",
      "api_key": "hashed_key",
      "spend": 0.01,
      "total_tokens": 100,
      "prompt_tokens": 50,
      "completion_tokens": 50,
      "model": "gpt-4",
      "user": "user-123",
      "team_id": "team-456",
      "organization_id": "org-789",
      "metadata": {...},
      "startTime": "2024-01-15T10:30:00Z",
      "endTime": "2024-01-15T10:30:05Z",
      "status": "success"
    }
  ]
}
```

### GET /user/info

Retrieve user information.

**Request:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" "https://api.example.com/user/info?user_id=user-123"
```

**Query Parameters:**
- `user_id` (required): User ID to query

**Response:**
```json
{
  "user_id": "user-123",
  "user_email": "user@example.com",
  "user_role": "admin",
  "max_budget": 100.0,
  "budget_duration": "monthly",
  "spend": 25.50,
  "models": ["gpt-4", "claude-3"],
  "tpm_limit": 100000,
  "rpm_limit": 1000,
  "metadata": {...},
  "createdAt": "2024-01-01T00:00:00Z",
  "updatedAt": "2024-01-15T10:30:00Z"
}
```

### GET /team/info

Retrieve team information.

**Request:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" "https://api.example.com/team/info?team_id=team-456"
```

**Query Parameters:**
- `team_id` (required): Team ID to query

**Response:**
```json
{
  "team_id": "team-456",
  "team_alias": "Production Team",
  "organization_id": "org-789",
  "max_budget": 1000.0,
  "budget_duration": "monthly",
  "spend": 250.50,
  "models": ["gpt-4", "claude-3", "text-embedding-3-small"],
  "tpm_limit": 1000000,
  "rpm_limit": 10000,
  "metadata": {...},
  "createdAt": "2024-01-01T00:00:00Z",
  "updatedAt": "2024-01-15T10:30:00Z"
}
```

## Health Check Routes

### GET /health/liveness

Basic liveness probe for Kubernetes.

**Response:**
```json
{
  "status": "ok"
}
```

### GET /health/readiness

Readiness probe checking database and cache connectivity.

**Response:**
```json
{
  "status": "ok",
  "database": "connected",
  "cache": "connected"
}
```

### GET /health/deep

Deep health check with detailed diagnostics.

**Response:**
```json
{
  "status": "ok",
  "database": {
    "status": "connected",
    "latency_ms": 5
  },
  "cache": {
    "status": "connected",
    "latency_ms": 2
  },
  "router": {
    "status": "ok",
    "deployments": 6
  }
}
```

## Metrics

### GET /metrics

Prometheus-compatible metrics endpoint.

**Response:** (text/plain)
```
# HELP litellm_requests_total Total number of requests
# TYPE litellm_requests_total counter
litellm_requests_total{model="gpt-4",status="success"} 1234

# HELP litellm_request_duration_seconds Request duration in seconds
# TYPE litellm_request_duration_seconds histogram
litellm_request_duration_seconds_bucket{model="gpt-4",le="0.1"} 100
litellm_request_duration_seconds_bucket{model="gpt-4",le="0.5"} 500
litellm_request_duration_seconds_bucket{model="gpt-4",le="1.0"} 800
litellm_request_duration_seconds_bucket{model="gpt-4",le="+Inf"} 1000

# HELP litellm_tokens_total Total tokens used
# TYPE litellm_tokens_total counter
litellm_tokens_total{model="gpt-4",type="prompt"} 50000
litellm_tokens_total{model="gpt-4",type="completion"} 30000

# HELP litellm_spend_usd_total Total spend in USD
# TYPE litellm_spend_usd_total counter
litellm_spend_usd_total{model="gpt-4"} 12.34
```

## Error Responses

All endpoints return errors in the following format:

```json
{
  "error": {
    "message": "Error description",
    "type": "error_type"
  }
}
```

**Common error types:**
- `authentication_error`: Invalid or missing API key
- `authorization_error`: API key lacks required permissions
- `rate_limit_error`: Rate limit exceeded
- `budget_error`: Budget exceeded
- `not_found`: Resource not found
- `validation_error`: Invalid request parameters
- `provider_error`: Upstream provider error
- `database_error`: Database error
- `not_implemented`: Feature not implemented

## Rate Limiting

Rate limits are enforced per API key and include:
- **RPM** (Requests Per Minute): Maximum number of requests per minute
- **TPM** (Tokens Per Minute): Maximum number of tokens per minute
- **Parallel Requests**: Maximum number of concurrent requests

Rate limit headers are included in responses:
```
X-RateLimit-Limit-RPM: 1000
X-RateLimit-Remaining-RPM: 999
X-RateLimit-Limit-TPM: 100000
X-RateLimit-Remaining-TPM: 99900
X-RateLimit-Limit-Parallel: 50
X-RateLimit-Remaining-Parallel: 49
```

## Streaming

Chat completions and messages endpoints support streaming via Server-Sent Events (SSE).

**Request:**
```json
{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true
}
```

**Response:** (SSE stream)
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Callbacks

The gateway supports multiple callback integrations for observability and monitoring:

### Langfuse

Traces and spans are automatically sent to Langfuse for LLM observability.

**Configuration:**
```yaml
litellm_settings:
  callbacks:
    - type: langfuse
      public_key: pk-lf-...
      secret_key: sk-lf-...
      host: https://cloud.langfuse.com
```

### Datadog

Metrics and logs are sent to Datadog for monitoring and alerting.

**Configuration:**
```yaml
litellm_settings:
  callbacks:
    - type: datadog
      api_key: dd-api-key
      app_key: dd-app-key  # optional
      host: https://api.datadoghq.com
```

### Webhooks

Custom webhooks can be configured to receive request/response data.

**Configuration:**
```yaml
litellm_settings:
  callbacks:
    - type: webhooks
      url: https://your-webhook-url.com/callback
      headers:  # optional
        X-Custom-Header: value
      auth_token: bearer-token  # optional
```

### Slack

Notifications can be sent to Slack channels for errors and alerts.

**Configuration:**
```yaml
litellm_settings:
  callbacks:
    - type: slack
      webhook_url: https://hooks.slack.com/services/...
      channel: "#alerts"  # optional
      username: "LiteLLM Bot"  # optional
      icon_emoji: ":robot_face:"  # optional
```

## Guardrails

Guardrails can be configured to validate requests and responses.

**Configuration:**
```yaml
litellm_settings:
  guardrails:
    - guardrail_name: prompt_injection
      guardrail_type: lakera
      enabled: true
    - guardrail_name: pii_detection
      guardrail_type: presidio
      enabled: true
```

Guardrails are executed before the provider call and can block or modify requests.

## Fallback Routing

The gateway supports automatic fallback routing across multiple deployments.

**Configuration:**
```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
    mode: fallback
    healthy: true
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4
      api_key: os.environ/AZURE_API_KEY
      api_base: https://my-resource.openai.azure.com
    mode: fallback
    healthy: true
```

When the primary deployment fails, the gateway automatically retries with the next healthy deployment.

## Circuit Breaker

The gateway implements a circuit breaker pattern to prevent cascading failures.

**Configuration:**
```yaml
router_settings:
  allowed_fails: 3
  cooldown_seconds: 60
```

After `allowed_fails` consecutive failures, the circuit opens and requests are routed to other deployments for `cooldown_seconds`.

## Retry Logic

Failed requests are automatically retried with exponential backoff.

**Configuration:**
```yaml
router_settings:
  num_retries: 3
  timeout: 300
```

Retries use exponential backoff with jitter to prevent thundering herd problems.
