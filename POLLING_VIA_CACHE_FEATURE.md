# Polling Via Cache Feature

## Overview

The Polling Via Cache feature allows users to make background Response API calls that return immediately with a polling ID, while the actual LLM response is streamed in the background and cached in Redis. Clients can poll the cached response to retrieve partial or complete results.

## Configuration

Add the following to your `litellm_config.yaml`:

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis
    ttl: 3600
    host: "127.0.0.1"
    port: "6379"
  
  # Response API polling configuration
  responses:
    background_mode:
      # Enable polling via cache for background responses
      # Options: 
      #   - "all" or ["all"]: Enable for all models
      #   - ["gpt-4o", "gpt-4"]: Enable for specific models
      #   - ["openai", "anthropic"]: Enable for specific providers
      polling_via_cache: ["all"]
```

## How It Works

### 1. Request Flow

When `background=true` is set in a Response API request:

1. **Detection**: Proxy checks if polling_via_cache is enabled and Redis is available
2. **UUID Generation**: Creates a polling ID with prefix `litellm_poll_`
3. **Initial State**: Stores initial state in Redis (TTL: 1 hour)
4. **Background Task**: Starts async task to stream response and update cache
5. **Immediate Return**: Returns polling ID to client

### 2. Background Streaming

The background task:
- Forces `stream=true` on the request
- Streams the response from the provider
- Updates Redis cache with cumulative content
- Stores final response when complete
- Handles errors and stores them in cache

### 3. Polling

Clients use the existing GET endpoint with the polling ID:
- Proxy detects `litellm_poll_` prefix
- Returns cached state instead of calling provider
- Includes cumulative content, status, and metadata

## API Usage

### 1. Start Background Response

```bash
curl -X POST http://localhost:4000/v1/responses \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "input": "Write a long essay about artificial intelligence",
    "background": true
  }'
```

**Response:**
```json
{
  "id": "litellm_poll_abc123def456",
  "object": "response.polling",
  "status": "pending",
  "created_at": 1700000000,
  "message": "Response is being generated in background. Use GET /v1/responses/{id} to retrieve partial or complete response."
}
```

### 2. Poll for Response

```bash
curl -X GET http://localhost:4000/v1/responses/litellm_poll_abc123def456 \
  -H "Authorization: Bearer sk-1234"
```

**Response (while streaming):**
```json
{
  "id": "litellm_poll_abc123def456",
  "object": "response.polling",
  "status": "streaming",
  "created_at": "2024-11-19T10:00:00Z",
  "updated_at": "2024-11-19T10:00:05Z",
  "content": "Artificial intelligence (AI) is a rapidly evolving field...",
  "content_length": 500,
  "chunk_count": 15,
  "metadata": {
    "model": "gpt-4o",
    "input": "Write a long essay about artificial intelligence"
  },
  "error": null,
  "final_response": null
}
```

**Response (completed):**
```json
{
  "id": "litellm_poll_abc123def456",
  "object": "response.polling",
  "status": "completed",
  "created_at": "2024-11-19T10:00:00Z",
  "updated_at": "2024-11-19T10:00:30Z",
  "content": "Artificial intelligence (AI) is a rapidly evolving field... [full essay]",
  "content_length": 5000,
  "chunk_count": 150,
  "metadata": {
    "model": "gpt-4o",
    "input": "Write a long essay about artificial intelligence"
  },
  "error": null,
  "final_response": { /* OpenAI response object */ }
}
```

### 3. Delete/Cancel Response

```bash
curl -X DELETE http://localhost:4000/v1/responses/litellm_poll_abc123def456 \
  -H "Authorization: Bearer sk-1234"
```

**Response:**
```json
{
  "id": "litellm_poll_abc123def456",
  "object": "response.deleted",
  "deleted": true
}
```

## Status Values

| Status | Description |
|--------|-------------|
| `pending` | Request received, background task not yet started |
| `streaming` | Background task is actively streaming response |
| `completed` | Response fully generated and cached |
| `error` | An error occurred during generation |
| `cancelled` | Response was cancelled by user |

## Implementation Details

### Polling ID Format

- **Prefix**: `litellm_poll_`
- **Format**: `litellm_poll_{uuid}`
- **Example**: `litellm_poll_abc123-def456-789ghi`

This prefix allows the GET endpoint to distinguish between:
- Polling IDs (handled by Redis cache)
- Provider response IDs (passed through to provider API)

### Redis Cache Structure

**Key**: `litellm:polling:response:litellm_poll_{uuid}`

**Value** (JSON):
```json
{
  "polling_id": "litellm_poll_abc123",
  "object": "response.polling",
  "status": "streaming",
  "created_at": "2024-11-19T10:00:00Z",
  "updated_at": "2024-11-19T10:00:05Z",
  "request_data": { /* original request */ },
  "user_id": "user_123",
  "team_id": "team_456",
  "content": "cumulative content so far...",
  "chunks": [ /* all streaming chunks */ ],
  "metadata": {
    "model": "gpt-4o",
    "input": "..."
  },
  "error": null,
  "final_response": null
}
```

**TTL**: 3600 seconds (1 hour)

### Security

- User/Team ID verification on GET and DELETE
- Only the user who created the request (or team members) can access it
- Automatic expiry after 1 hour prevents stale data

## Configuration Options

### Enable for All Models

```yaml
responses:
  background_mode:
    polling_via_cache: ["all"]
```

### Enable for Specific Models

```yaml
responses:
  background_mode:
    polling_via_cache: ["gpt-4o", "gpt-4", "claude-3"]
```

### Enable for Specific Providers

```yaml
responses:
  background_mode:
    polling_via_cache: ["openai", "anthropic"]
```

This will match any model starting with `openai/` or `anthropic/`.

## Benefits

1. **Immediate Response**: Client gets polling ID instantly, no waiting
2. **Partial Results**: Can retrieve partial content while generation continues
3. **Progress Monitoring**: Poll at intervals to show progress to users
4. **Error Handling**: Errors are cached and can be retrieved
5. **Scalability**: Background tasks don't block API requests

## Limitations

1. **Requires Redis**: Feature only works with Redis cache configured
2. **1 Hour TTL**: Responses expire after 1 hour
3. **No Streaming to Client**: Client must poll, no real-time streaming
4. **Memory Usage**: Full response stored in Redis

## Example Client Implementation

### Python

```python
import time
import requests

# Start background response
response = requests.post(
    "http://localhost:4000/v1/responses",
    headers={"Authorization": "Bearer sk-1234"},
    json={
        "model": "gpt-4o",
        "input": "Write a long essay",
        "background": True
    }
)

polling_id = response.json()["id"]
print(f"Started background response: {polling_id}")

# Poll for results
while True:
    poll_response = requests.get(
        f"http://localhost:4000/v1/responses/{polling_id}",
        headers={"Authorization": "Bearer sk-1234"}
    )
    
    data = poll_response.json()
    status = data["status"]
    content = data["content"]
    
    print(f"Status: {status}, Content length: {len(content)}")
    
    if status == "completed":
        print("Final response:", content)
        break
    elif status == "error":
        print("Error:", data["error"])
        break
    
    time.sleep(2)  # Poll every 2 seconds
```

### JavaScript

```javascript
async function pollResponse(pollingId) {
  while (true) {
    const response = await fetch(
      `http://localhost:4000/v1/responses/${pollingId}`,
      { headers: { 'Authorization': 'Bearer sk-1234' } }
    );
    
    const data = await response.json();
    console.log(`Status: ${data.status}, Content: ${data.content.substring(0, 50)}...`);
    
    if (data.status === 'completed') {
      console.log('Final response:', data.content);
      break;
    } else if (data.status === 'error') {
      console.error('Error:', data.error);
      break;
    }
    
    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2s
  }
}

// Start background response
const startResponse = await fetch('http://localhost:4000/v1/responses', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer sk-1234',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model: 'gpt-4o',
    input: 'Write a long essay',
    background: true
  })
});

const { id } = await startResponse.json();
await pollResponse(id);
```

## Testing

To test the feature:

1. **Start Redis** (if not already running):
   ```bash
   redis-server --port 6379
   ```

2. **Start LiteLLM Proxy**:
   ```bash
   python -m litellm.proxy.proxy_cli --config litellm_config.yaml --detailed_debug
   ```

3. **Make a background request**:
   ```bash
   curl -X POST http://localhost:4000/v1/responses \
     -H "Authorization: Bearer sk-test-key" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-4o",
       "input": "Count from 1 to 100",
       "background": true
     }'
   ```

4. **Poll for results**:
   ```bash
   # Replace with your polling_id
   curl http://localhost:4000/v1/responses/litellm_poll_XXX \
     -H "Authorization: Bearer sk-test-key"
   ```

5. **Check Redis**:
   ```bash
   redis-cli
   > KEYS litellm:polling:response:*
   > GET litellm:polling:response:litellm_poll_XXX
   ```

## Troubleshooting

### Issue: Polling not enabled

**Symptom**: Requests with `background=true` return immediately without streaming

**Solution**: 
- Verify Redis is running and accessible
- Check `redis_usage_cache` is initialized
- Ensure `polling_via_cache` is configured

### Issue: Polling ID not found

**Symptom**: GET returns 404

**Possible causes**:
- Response expired (>1 hour old)
- Redis connection lost
- Wrong polling ID

### Issue: Empty content

**Symptom**: Content length is 0

**Possible causes**:
- Background task still starting
- Error in streaming
- Check logs for background task errors

## Future Enhancements

Potential improvements:
1. WebSocket support for real-time updates
2. Configurable TTL per request
3. Compression for large responses
4. Pagination for very long responses
5. Metrics and monitoring endpoints


