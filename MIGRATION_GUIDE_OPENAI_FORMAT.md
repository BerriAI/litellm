# Migration Guide: OpenAI Response Format

This guide helps you migrate from the previous polling format to the new OpenAI Response object format.

## Quick Reference

### Field Name Changes

| Old Field | New Field | Location | Notes |
|-----------|-----------|----------|-------|
| `polling_id` | `id` | Top level | Renamed for OpenAI compatibility |
| `object: "response.polling"` | `object: "response"` | Top level | Changed to match OpenAI |
| `content` (string) | `output[].content[]` | Nested | Now structured array |
| `chunks` | N/A | Removed | Data now in `output` items |
| `error` (string) | `status_details.error` (object) | Nested | Structured error format |
| `final_response` | N/A | Removed | Full data always in response |
| `content_length` | N/A | Removed | Calculate from `output` |
| `chunk_count` | N/A | Removed | Use `output.length` |

### Status Value Changes

| Old Status | New Status |
|-----------|-----------|
| `pending` | `in_progress` |
| `streaming` | `in_progress` |
| `completed` | `completed` |
| `error` | `failed` |
| `cancelled` | `cancelled` |

## Code Migration Examples

### 1. Extracting Text Content

**Before:**
```python
response = requests.get(f"{url}/v1/responses/{polling_id}")
data = response.json()

content = data.get("content", "")
content_length = data.get("content_length", 0)
```

**After:**
```python
response = requests.get(f"{url}/v1/responses/{polling_id}")
data = response.json()

# Extract text from output items
content = ""
for item in data.get("output", []):
    if item.get("type") == "message":
        for part in item.get("content", []):
            if part.get("type") == "text":
                content += part.get("text", "")

content_length = len(content)
```

**Helper Function:**
```python
def extract_text_content(response_obj):
    """Extract text content from OpenAI Response object"""
    text = ""
    for item in response_obj.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "text":
                    text += part.get("text", "")
    return text

# Usage
content = extract_text_content(data)
```

### 2. Checking Status

**Before:**
```python
status = data.get("status")

if status == "pending" or status == "streaming":
    print("Still processing...")
elif status == "completed":
    print("Done!")
elif status == "error":
    error_msg = data.get("error", "Unknown error")
    print(f"Error: {error_msg}")
```

**After:**
```python
status = data.get("status")

if status == "in_progress":
    print("Still processing...")
elif status == "completed":
    print("Done!")
    # Check completion details
    status_details = data.get("status_details", {})
    reason = status_details.get("reason", "unknown")
    print(f"Completed: {reason}")
elif status == "failed":
    # Structured error object
    error = data.get("status_details", {}).get("error", {})
    error_type = error.get("type", "unknown")
    error_msg = error.get("message", "Unknown error")
    error_code = error.get("code", "")
    print(f"Error [{error_type}]: {error_msg} (code: {error_code})")
```

### 3. Polling Loop

**Before:**
```python
while True:
    response = requests.get(f"{url}/v1/responses/{polling_id}")
    data = response.json()
    
    status = data["status"]
    content = data.get("content", "")
    
    print(f"Status: {status}, Content: {len(content)} chars")
    
    if status == "completed":
        return data
    elif status == "error":
        raise Exception(data.get("error"))
    
    time.sleep(2)
```

**After:**
```python
def extract_text_content(response_obj):
    text = ""
    for item in response_obj.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "text":
                    text += part.get("text", "")
    return text

while True:
    response = requests.get(f"{url}/v1/responses/{polling_id}")
    data = response.json()
    
    status = data["status"]
    content = extract_text_content(data)
    
    print(f"Status: {status}, Content: {len(content)} chars")
    
    if status == "completed":
        # Show usage if available
        usage = data.get("usage")
        if usage:
            print(f"Tokens used: {usage.get('total_tokens')}")
        return data
    elif status == "failed":
        error = data.get("status_details", {}).get("error", {})
        raise Exception(error.get("message", "Unknown error"))
    elif status == "cancelled":
        raise Exception("Response was cancelled")
    
    time.sleep(2)
```

### 4. Creating Background Response

**Before & After (Same):**
```python
response = requests.post(
    f"{url}/v1/responses",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "gpt-4o",
        "input": "Your prompt",
        "background": True
    }
)

data = response.json()
polling_id = data["id"]  # Still works! (was polling_id, now just id)
```

**Note:** The request format is unchanged, but the response structure is different.

### 5. Error Handling

**Before:**
```python
if data.get("status") == "error":
    error_message = data.get("error", "Unknown error")
    print(f"Error: {error_message}")
```

**After:**
```python
if data.get("status") == "failed":
    status_details = data.get("status_details", {})
    error = status_details.get("error", {})
    
    error_type = error.get("type", "unknown")
    error_message = error.get("message", "Unknown error")
    error_code = error.get("code", "")
    
    print(f"Error [{error_type}]: {error_message}")
    if error_code:
        print(f"Error code: {error_code}")
```

### 6. Accessing Metadata

**Before & After (Similar):**
```python
metadata = data.get("metadata", {})
```

**Note:** Metadata structure is unchanged.

### 7. Getting Usage Information

**Before:**
```python
# Not available in old format
```

**After:**
```python
usage = data.get("usage")
if usage:
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    
    print(f"Token usage:")
    print(f"  Input: {input_tokens}")
    print(f"  Output: {output_tokens}")
    print(f"  Total: {total_tokens}")
```

## Complete Migration Example

### Before (Old Format)

```python
import time
import requests

def poll_response_old(url, api_key, polling_id):
    """Old format polling"""
    headers = {"Authorization": f"Bearer {api_key}"}
    
    while True:
        response = requests.get(
            f"{url}/v1/responses/{polling_id}",
            headers=headers
        )
        data = response.json()
        
        status = data.get("status")
        content = data.get("content", "")
        content_length = data.get("content_length", 0)
        
        print(f"[{status}] {content_length} chars")
        
        if status == "completed":
            print(f"✅ Done! Content: {content[:100]}...")
            return content
        elif status == "error":
            raise Exception(f"Error: {data.get('error')}")
        elif status in ["pending", "streaming"]:
            time.sleep(2)
        else:
            raise Exception(f"Unknown status: {status}")
```

### After (OpenAI Format)

```python
import time
import requests

def extract_text_content(response_obj):
    """Extract text content from OpenAI Response object"""
    text = ""
    for item in response_obj.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "text":
                    text += part.get("text", "")
    return text

def poll_response_new(url, api_key, polling_id):
    """New OpenAI format polling"""
    headers = {"Authorization": f"Bearer {api_key}"}
    
    while True:
        response = requests.get(
            f"{url}/v1/responses/{polling_id}",
            headers=headers
        )
        data = response.json()
        
        status = data.get("status")
        content = extract_text_content(data)
        content_length = len(content)
        
        print(f"[{status}] {content_length} chars")
        
        if status == "completed":
            usage = data.get("usage", {})
            tokens = usage.get("total_tokens", 0)
            print(f"✅ Done! Content: {content[:100]}...")
            print(f"Tokens used: {tokens}")
            return content
        elif status == "failed":
            error = data.get("status_details", {}).get("error", {})
            raise Exception(f"Error: {error.get('message', 'Unknown error')}")
        elif status == "cancelled":
            raise Exception("Response was cancelled")
        elif status == "in_progress":
            time.sleep(2)
        else:
            raise Exception(f"Unknown status: {status}")
```

## TypeScript/JavaScript Migration

### Before

```typescript
interface OldPollingResponse {
  polling_id: string;
  object: "response.polling";
  status: "pending" | "streaming" | "completed" | "error" | "cancelled";
  content: string;
  content_length: number;
  chunk_count: number;
  error?: string;
  metadata?: Record<string, any>;
}

// Usage
const data: OldPollingResponse = await response.json();
console.log(data.content);
```

### After

```typescript
interface OpenAIResponseObject {
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
    role?: "assistant";
    status?: "in_progress" | "completed";
    content?: Array<{
      type: "text";
      text: string;
    }>;
  }>;
  usage: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  } | null;
  metadata: Record<string, any>;
  created_at: number;
}

// Helper function
function extractTextContent(response: OpenAIResponseObject): string {
  let text = "";
  for (const item of response.output) {
    if (item.type === "message" && item.content) {
      for (const part of item.content) {
        if (part.type === "text") {
          text += part.text;
        }
      }
    }
  }
  return text;
}

// Usage
const data: OpenAIResponseObject = await response.json();
const content = extractTextContent(data);
console.log(content);
```

## Configuration Changes

### litellm_config.yaml

**No changes required!** The configuration format remains the same:

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: "127.0.0.1"
    port: "6379"
  responses:
    background_mode:
      polling_via_cache: true
      polling_ttl: 7200
```

## Validation Checklist

Use this checklist to ensure your migration is complete:

- [ ] Updated field names (`polling_id` → `id`)
- [ ] Updated status checks (`pending`/`streaming` → `in_progress`)
- [ ] Updated error handling (`error` → `status_details.error`)
- [ ] Implemented content extraction from `output` array
- [ ] Added usage tracking (optional but recommended)
- [ ] Updated TypeScript interfaces (if applicable)
- [ ] Tested with actual API calls
- [ ] Updated documentation/comments in code
- [ ] Verified backward compatibility isn't assumed

## Common Pitfalls

### 1. Assuming Flat Content

❌ **Wrong:**
```python
content = data.get("content", "")  # This field no longer exists!
```

✅ **Correct:**
```python
content = extract_text_content(data)
```

### 2. Old Status Values

❌ **Wrong:**
```python
if status == "pending" or status == "streaming":
    # Will never match!
```

✅ **Correct:**
```python
if status == "in_progress":
    # Correct!
```

### 3. Simple Error Messages

❌ **Wrong:**
```python
error = data.get("error")  # No longer exists at top level
```

✅ **Correct:**
```python
error = data.get("status_details", {}).get("error", {}).get("message")
```

### 4. Ignoring Output Item Types

❌ **Wrong:**
```python
# Assuming all output is text
for item in data["output"]:
    text = item["content"]  # Might not be text!
```

✅ **Correct:**
```python
for item in data["output"]:
    if item.get("type") == "message":
        for part in item.get("content", []):
            if part.get("type") == "text":
                text = part.get("text", "")
```

## Testing Your Migration

Use this simple test to verify your migration:

```python
import requests

url = "http://localhost:4000"
api_key = "sk-test-key"

# Start background response
response = requests.post(
    f"{url}/v1/responses",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "gpt-4o",
        "input": "Say hello",
        "background": True
    }
)

data = response.json()

# Verify new format
assert "id" in data, "Missing 'id' field"
assert data["object"] == "response", f"Wrong object type: {data['object']}"
assert data["status"] == "in_progress", f"Wrong initial status: {data['status']}"
assert "output" in data, "Missing 'output' field"
assert isinstance(data["output"], list), "output should be a list"

print("✅ Migration successful! Your code is using the new format.")
```

## Getting Help

- **Documentation**: See `OPENAI_RESPONSE_FORMAT.md` for complete format specification
- **Examples**: Check `test_polling_feature.py` for working examples
- **OpenAI Docs**: https://platform.openai.com/docs/api-reference/responses/object

## Timeline

- **Old Format**: Deprecated
- **New Format**: Current (OpenAI compatible)
- **Breaking Change**: Yes - requires code updates

We recommend migrating as soon as possible to ensure compatibility with future updates.

