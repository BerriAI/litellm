# Using Anthropic File API with LiteLLM Proxy

## Overview

This tutorial shows how to create and analyze files with Claude-4 on Anthropic via LiteLLM Proxy.

## Prerequisites

- LiteLLM Proxy running
- Anthropic API key

Add the following to your `.env` file:
```
ANTHROPIC_API_KEY=sk-1234
```

## Usage

### 1. Setup config.yaml

```yaml
model_list:
  - model_name: claude-opus
    litellm_params:
      model: anthropic/claude-opus-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
```

## 2. Create a file 

Use the `/anthropic` passthrough endpoint to create a file.

```bash
curl -L -X POST 'http://0.0.0.0:4000/anthropic/v1/files' \
-H 'x-api-key: sk-1234' \
-H 'anthropic-version: 2023-06-01' \
-H 'anthropic-beta: files-api-2025-04-14' \
-F 'file=@"/path/to/your/file.csv"'
```

Expected response:

```json
{
  "created_at": "2023-11-07T05:31:56Z",
  "downloadable": false,
  "filename": "file.csv",
  "id": "file-1234",
  "mime_type": "text/csv",
  "size_bytes": 1,
  "type": "file"
}
```


## 3. Analyze the file with Claude-4 via `/chat/completions`


```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer $LITELLM_API_KEY' \
-d '{
    "model": "claude-opus",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this sheet?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "file-1234",
                        "format": "text/csv" # 👈 IMPORTANT: This is the format of the file you want to analyze
                    }
                }
            ]
        }
    ]
}'
```