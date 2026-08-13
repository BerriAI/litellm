# A2A Protocol Guardrail Translation Handler

Handler for processing A2A (Agent-to-Agent) Protocol messages with guardrails.

## Overview

This handler processes A2A JSON-RPC 2.0 input/output by:
1. Extracting text from message parts (`kind: "text"`)
2. Applying guardrails to text content
3. Mapping guardrailed text back to original structure

## A2A Protocol Format

### Input Format (JSON-RPC 2.0)

```json
{
  "jsonrpc": "2.0",
  "id": "request-id",
  "method": "message/send",
  "params": {
    "message": {
      "kind": "message",
      "messageId": "...",
      "role": "user",
      "parts": [
        {"kind": "text", "text": "Hello, my SSN is 123-45-6789"}
      ]
    },
    "metadata": {
      "guardrails": ["block-ssn"]
    }
  }
}
```

### Output Formats

The handler supports multiple A2A response formats:

**Direct message:**
```json
{
  "result": {
    "kind": "message",
    "parts": [{"kind": "text", "text": "Response text"}]
  }
}
```

**Nested message:**
```json
{
  "result": {
    "message": {
      "parts": [{"kind": "text", "text": "Response text"}]
    }
  }
}
```

**Task with artifacts:**
```json
{
  "result": {
    "kind": "task",
    "artifacts": [
      {"parts": [{"kind": "text", "text": "Artifact text"}]}
    ]
  }
}
```

**Task with status message:**
```json
{
  "result": {
    "kind": "task",
    "status": {
      "message": {
        "parts": [{"kind": "text", "text": "Status message"}]
      }
    }
  }
}
```

**Streaming artifact-update:**
```json
{
  "result": {
    "kind": "artifact-update",
    "artifact": {
      "parts": [{"kind": "text", "text": "Streaming text"}]
    }
  }
}
```

## Usage

The handler is automatically discovered and applied when guardrails are used with A2A endpoints.

### Via LiteLLM Proxy

```bash
curl -X POST 'http://localhost:4000/a2a/my-agent' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "message/send",
  "params": {
    "message": {
      "kind": "message",
      "messageId": "msg-1",
      "role": "user",
      "parts": [{"kind": "text", "text": "Hello, my SSN is 123-45-6789"}]
    },
    "metadata": {
      "guardrails": ["block-ssn"]
    }
  }
}'
```

### Specifying Guardrails

Guardrails can be specified in the A2A request via the `metadata.guardrails` field:

```json
{
  "params": {
    "message": {...},
    "metadata": {
      "guardrails": ["block-ssn", "pii-filter"]
    }
  }
}
```

## Extension

Override these methods to customize behavior:

- `_extract_texts_from_result()`: Custom text extraction from A2A responses
- `_extract_texts_from_parts()`: Custom text extraction from message parts
- `_apply_text_to_path()`: Custom application of guardrailed text

## Call Types

This handler is registered for:
- `CallTypes.send_message`: Synchronous A2A message sending
- `CallTypes.asend_message`: Asynchronous A2A message sending
