# OpenAI Response Object Format - Polling Via Cache Implementation

## Overview

The polling via cache feature now follows the official OpenAI Response object format as documented at:
- **Response Object**: https://platform.openai.com/docs/api-reference/responses/object
- **Streaming Events**: https://platform.openai.com/docs/api-reference/responses-streaming

## Response Object Structure

The Response object stored in Redis cache follows this structure:

```json
{
  "id": "litellm_poll_abc123-def456",
  "object": "response",
  "status": "in_progress" | "completed" | "cancelled" | "failed" | "incomplete",
  "status_details": {
    "type": "completed" | "incomplete" | "cancelled" | "failed",
    "reason": "stop" | "length" | "content_filter" | "user_requested",
    "error": {
      "type": "internal_error",
      "message": "Error message",
      "code": "error_code"
    }
  },
  "output": [
    {
      "id": "item_001",
      "type": "message",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "Response content here..."
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 100,
    "output_tokens": 500,
    "total_tokens": 600
  },
  "metadata": {
    "custom_field": "custom_value"
  },
  "created_at": 1700000000
}
```

### Internal Polling Fields

For internal tracking, additional fields are stored under `_polling_state`:

```json
{
  "_polling_state": {
    "updated_at": "2024-11-19T10:00:05Z",
    "request_data": { /* original request */ },
    "user_id": "user_123",
    "team_id": "team_456",
    "model": "gpt-4o",
    "input": "User prompt..."
  }
}
```

## Status Values

Following OpenAI's format:

| Status | Description |
|--------|-------------|
| `in_progress` | Response is currently being generated |
| `completed` | Response has been fully generated |
| `cancelled` | Response was cancelled by user |
| `failed` | Response generation failed with an error |
| `incomplete` | Response was cut off (length limit, content filter) |

## Streaming Events Processing

The background streaming task processes these OpenAI streaming events:

### 1. `response.created`
Initial response created event (handled by initial state creation).

### 2. `response.output_item.added`
```json
{
  "type": "response.output_item.added",
  "item": {
    "id": "item_001",
    "type": "message",
    "role": "assistant",
    "status": "in_progress"
  }
}
```

### 3. `response.content_part.added`
```json
{
  "type": "response.content_part.added",
  "item_id": "item_001",
  "output_index": 0,
  "part": {
    "type": "text",
    "text": "Initial text..."
  }
}
```

### 4. `response.content_part.done`
```json
{
  "type": "response.content_part.done",
  "item_id": "item_001",
  "part": {
    "type": "text",
    "text": "Complete text content"
  }
}
```

### 5. `response.output_item.done`
```json
{
  "type": "response.output_item.done",
  "item": {
    "id": "item_001",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [
      {
        "type": "text",
        "text": "Complete content"
      }
    ]
  }
}
```

### 6. `response.done`
```json
{
  "type": "response.done",
  "response": {
    "id": "litellm_poll_abc123",
    "status": "completed",
    "status_details": {
      "type": "completed",
      "reason": "stop"
    },
    "usage": {
      "input_tokens": 100,
      "output_tokens": 500,
      "total_tokens": 600
    }
  }
}
```

## API Examples

### Creating a Background Response

```bash
curl -X POST http://localhost:4000/v1/responses \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "input": "Write an essay about AI",
    "background": true,
    "metadata": {
      "user": "john_doe",
      "session_id": "sess_123"
    }
  }'
```

**Response:**
```json
{
  "id": "litellm_poll_abc123def456",
  "object": "response",
  "status": "in_progress",
  "status_details": null,
  "output": [],
  "usage": null,
  "metadata": {
    "user": "john_doe",
    "session_id": "sess_123"
  },
  "created_at": 1700000000
}
```

### Polling for Response (In Progress)

```bash
curl -X GET http://localhost:4000/v1/responses/litellm_poll_abc123def456 \
  -H "Authorization: Bearer sk-1234"
```

**Response:**
```json
{
  "id": "litellm_poll_abc123def456",
  "object": "response",
  "status": "in_progress",
  "status_details": null,
  "output": [
    {
      "id": "item_001",
      "type": "message",
      "role": "assistant",
      "status": "in_progress",
      "content": [
        {
          "type": "text",
          "text": "Artificial intelligence (AI) is a rapidly..."
        }
      ]
    }
  ],
  "usage": null,
  "metadata": {
    "user": "john_doe",
    "session_id": "sess_123"
  },
  "created_at": 1700000000
}
```

### Polling for Response (Completed)

```bash
curl -X GET http://localhost:4000/v1/responses/litellm_poll_abc123def456 \
  -H "Authorization: Bearer sk-1234"
```

**Response:**
```json
{
  "id": "litellm_poll_abc123def456",
  "object": "response",
  "status": "completed",
  "status_details": {
    "type": "completed",
    "reason": "stop"
  },
  "output": [
    {
      "id": "item_001",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {
          "type": "text",
          "text": "Artificial intelligence (AI) is a rapidly evolving field... [full essay]"
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 25,
    "output_tokens": 1200,
    "total_tokens": 1225
  },
  "metadata": {
    "user": "john_doe",
    "session_id": "sess_123"
  },
  "created_at": 1700000000
}
```

### Error Response

```json
{
  "id": "litellm_poll_abc123def456",
  "object": "response",
  "status": "failed",
  "status_details": {
    "type": "failed",
    "error": {
      "type": "internal_error",
      "message": "Provider timeout",
      "code": "background_streaming_error"
    }
  },
  "output": [],
  "usage": null,
  "metadata": {},
  "created_at": 1700000000
}
```

## Output Item Types

### Message Output
```json
{
  "id": "item_001",
  "type": "message",
  "role": "assistant",
  "status": "completed",
  "content": [
    {
      "type": "text",
      "text": "Message content"
    }
  ]
}
```

### Function Call Output
```json
{
  "id": "item_002",
  "type": "function_call",
  "status": "completed",
  "name": "get_weather",
  "call_id": "call_abc123",
  "arguments": "{\"location\": \"San Francisco\"}"
}
```

### Function Call Output Result
```json
{
  "id": "item_003",
  "type": "function_call_output",
  "call_id": "call_abc123",
  "output": "{\"temperature\": 72, \"condition\": \"sunny\"}"
}
```

## Redis Cache Storage

### Key Format
```
litellm:polling:response:litellm_poll_{uuid}
```

### TTL
- Default: 3600 seconds (1 hour)
- Configurable via `ttl` parameter

### Storage Example
```redis
> KEYS litellm:polling:response:*
1) "litellm:polling:response:litellm_poll_abc123def456"

> GET "litellm:polling:response:litellm_poll_abc123def456"
"{\"id\":\"litellm_poll_abc123def456\",\"object\":\"response\",\"status\":\"completed\",...}"

> TTL "litellm:polling:response:litellm_poll_abc123def456"
(integer) 2847
```

## Client Implementation Example

### Python Client

```python
import time
import requests

def poll_response(polling_id, api_key):
    """Poll for response following OpenAI format"""
    url = f"http://localhost:4000/v1/responses/{polling_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    while True:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        status = data["status"]
        print(f"Status: {status}")
        
        # Extract content from output items
        for item in data.get("output", []):
            if item["type"] == "message":
                content = ""
                for part in item.get("content", []):
                    if part["type"] == "text":
                        content += part["text"]
                print(f"Content: {content[:100]}...")
        
        # Check status
        if status == "completed":
            print("\n✅ Response completed!")
            print(f"Usage: {data.get('usage')}")
            return data
        elif status == "failed":
            error = data.get("status_details", {}).get("error", {})
            print(f"\n❌ Error: {error.get('message')}")
            return None
        elif status == "cancelled":
            print("\n⚠️ Response cancelled")
            return None
        
        time.sleep(2)  # Poll every 2 seconds

# Start background response
response = requests.post(
    "http://localhost:4000/v1/responses",
    headers={
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json"
    },
    json={
        "model": "gpt-4o",
        "input": "Write an essay",
        "background": True
    }
)

polling_id = response.json()["id"]
result = poll_response(polling_id, "sk-1234")
```

### JavaScript/TypeScript Client

```typescript
interface ResponseObject {
  id: string;
  object: "response";
  status: "in_progress" | "completed" | "cancelled" | "failed" | "incomplete";
  status_details: {
    type: string;
    reason?: string;
    error?: {
      type: string;
      message: string;
      code: string;
    };
  } | null;
  output: Array<{
    id: string;
    type: "message" | "function_call" | "function_call_output";
    content?: Array<{ type: "text"; text: string }>;
    [key: string]: any;
  }>;
  usage: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  } | null;
  metadata: Record<string, any>;
  created_at: number;
}

async function pollResponse(pollingId: string, apiKey: string): Promise<ResponseObject> {
  const url = `http://localhost:4000/v1/responses/${pollingId}`;
  const headers = { Authorization: `Bearer ${apiKey}` };
  
  while (true) {
    const response = await fetch(url, { headers });
    const data: ResponseObject = await response.json();
    
    console.log(`Status: ${data.status}`);
    
    // Extract text content
    for (const item of data.output) {
      if (item.type === "message" && item.content) {
        const text = item.content
          .filter(p => p.type === "text")
          .map(p => p.text)
          .join("");
        console.log(`Content: ${text.substring(0, 100)}...`);
      }
    }
    
    if (data.status === "completed") {
      console.log("✅ Response completed!");
      console.log("Usage:", data.usage);
      return data;
    } else if (data.status === "failed") {
      throw new Error(data.status_details?.error?.message || "Unknown error");
    } else if (data.status === "cancelled") {
      throw new Error("Response was cancelled");
    }
    
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}
```

## Compatibility Notes

1. **OpenAI API Compatibility**: The response format is fully compatible with OpenAI's Response API
2. **Polling ID Prefix**: The `litellm_poll_` prefix allows the proxy to distinguish between polling IDs and provider response IDs
3. **Internal Fields**: The `_polling_state` object is for internal use only and not exposed in the API response
4. **Provider Agnostic**: Works with any LLM provider through LiteLLM's unified interface

## Migration from Previous Format

If you were using the previous format, here are the key changes:

| Old Field | New Field | Notes |
|-----------|-----------|-------|
| `polling_id` | `id` | Standard field name |
| `object: "response.polling"` | `object: "response"` | OpenAI format |
| `status: "pending"` | `status: "in_progress"` | Aligned with OpenAI |
| `status: "streaming"` | `status: "in_progress"` | Same as above |
| `content` | `output[].content[]` | Structured output items |
| `error` | `status_details.error` | Nested error object |
| N/A | `usage` | Added token usage tracking |

## References

- OpenAI Response Object: https://platform.openai.com/docs/api-reference/responses/object
- OpenAI Response Streaming: https://platform.openai.com/docs/api-reference/responses-streaming
- LiteLLM Documentation: https://docs.litellm.ai/

