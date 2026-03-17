import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# File Search in the Responses API — E2E Testing Guide

This tutorial walks you through end-to-end testing of the `file_search` tool in LiteLLM's Responses API.
Two paths are covered:

| Path | When it runs | What LiteLLM does |
|---|---|---|
| **Native passthrough** | Provider natively supports `file_search` (OpenAI, Azure) | Decodes unified vector store ID → forwards to provider as-is |
| **Emulated fallback** | Provider doesn't support `file_search` (Anthropic, Bedrock, etc.) | Converts to a function tool → intercepts tool call → runs vector search → synthesizes OpenAI-format output |

---

## Prerequisites

```bash
pip install 'litellm[proxy]'
export OPENAI_API_KEY="sk-..."          # for native path
export ANTHROPIC_API_KEY="sk-ant-..."  # for emulated path
```

---

## Path 1: Native Passthrough (OpenAI)

OpenAI natively handles `file_search`. LiteLLM decodes any unified vector store ID and forwards the request unchanged.

### Step 1 — Create a vector store and upload a file

```python
from openai import OpenAI

client = OpenAI()  # direct OpenAI call to set up test data

# Upload a file
with open("knowledge.txt", "w") as f:
    f.write("LiteLLM is a unified interface for 100+ LLM providers. "
            "It supports chat completions, responses API, embeddings, and more.")

file = client.files.create(file=open("knowledge.txt", "rb"), purpose="assistants")
print("file_id:", file.id)

# Create a vector store and attach the file
vs = client.vector_stores.create(name="litellm-test-store")
client.vector_stores.files.create(vector_store_id=vs.id, file_id=file.id)
print("vector_store_id:", vs.id)
```

### Step 2 — Run file search via LiteLLM Python SDK

```python showLineNumbers title="Native file_search via LiteLLM SDK"
import litellm

response = litellm.responses(
    model="openai/gpt-4.1",
    input="What does LiteLLM support?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_abc123"]  # replace with your vector_store_id
    }],
)

for item in response.output:
    if item.type == "file_search_call":
        print("Queries run:", item.queries)
        print("Status:", item.status)
    elif item.type == "message":
        for block in item.content:
            print("\nAnswer:", block.text)
            for ann in block.annotations:
                print(f"  ↳ Citation: {ann.filename} (file_id={ann.file_id})")
```

**Expected output:**
```
Queries run: ['What does LiteLLM support?']
Status: completed

Answer: LiteLLM is a unified interface for 100+ LLM providers...
  ↳ Citation: knowledge.txt (file_id=file-xxxx)
```

### Step 3 — Run via LiteLLM Proxy

Start the proxy:

```bash title="config.yaml"
# config.yaml
model_list:
  - model_name: gpt-4.1
    litellm_params:
      model: openai/gpt-4.1
      api_key: os.environ/OPENAI_API_KEY
```

```bash
litellm --config config.yaml
```

Call the proxy:

```python showLineNumbers title="Native file_search via LiteLLM Proxy"
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000", api_key="any")

response = client.responses.create(
    model="gpt-4.1",
    input="What does LiteLLM support?",
    tools=[{"type": "file_search", "vector_store_ids": ["vs_abc123"]}],
)

for item in response.output:
    print(item.type, getattr(item, "queries", getattr(item, "content", "")))
```

---

## Path 2: Emulated Fallback (Anthropic / any non-native provider)

When you use a provider that doesn't natively support `file_search`, LiteLLM:
1. Converts the `file_search` tool to a function tool (`litellm_file_search`).
2. Lets the provider call the function with a natural-language query.
3. Runs your vector store search internally.
4. Feeds results back and makes a follow-up call.
5. Returns the final answer in OpenAI's `file_search_call` + `message` format.

### Step 1 — Register a LiteLLM-managed vector store

LiteLLM's vector store registry lets you configure any supported vector store backend (OpenAI, Pinecone, Milvus, Qdrant, etc.):

```python showLineNumbers title="Register vector store via LiteLLM Proxy API"
import requests

# Register the vector store with LiteLLM Proxy
resp = requests.post(
    "http://localhost:4000/v1/vector_stores/new",
    headers={"Authorization": "Bearer sk-your-proxy-key"},
    json={
        "vector_store_id": "my-openai-vs",        # your logical name
        "custom_llm_provider": "openai",
        "vector_store_name": "litellm-test-store",
        "litellm_params": {
            "api_key": "sk-..."  # provider API key (or use credentials in config.yaml)
        },
    },
)
print(resp.json())
# Returns: {"vector_store_id": "bGl0ZWxsbV9wcm94eToB..."}  ← LiteLLM unified ID
```

:::tip
Save the returned `vector_store_id` — this is the **LiteLLM-managed unified ID** that encodes the provider routing. Pass this in `vector_store_ids` and LiteLLM will decode it automatically.
:::

### Step 2 — Run file search via LiteLLM SDK (emulated)

```python showLineNumbers title="Emulated file_search with Anthropic"
import litellm

# Use the unified vector_store_id returned by /v1/vector_stores/new
UNIFIED_VS_ID = "bGl0ZWxsbV9wcm94eToB..."

response = litellm.responses(
    model="anthropic/claude-sonnet-4-5",
    input="What does LiteLLM support?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": [UNIFIED_VS_ID]
    }],
)

for item in response.output:
    if item.type == "file_search_call":
        print("Queries run:", item.queries)
    elif item.type == "message":
        for block in item.content:
            print("\nAnswer:", block.text)
            for ann in block.annotations:
                print(f"  ↳ Citation: {ann.filename}")
```

LiteLLM automatically detects that Anthropic doesn't support `file_search` natively and routes through the emulated handler.

### Step 3 — Run via LiteLLM Proxy (emulated)

```bash title="config.yaml"
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY
```

```python showLineNumbers title="Emulated file_search via LiteLLM Proxy"
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000", api_key="sk-your-proxy-key")

response = client.responses.create(
    model="claude-sonnet",
    input="What does LiteLLM support?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["bGl0ZWxsbV9wcm94eToB..."]  # unified ID
    }],
)

for item in response.output:
    if hasattr(item, "type"):
        if item.type == "file_search_call":
            print("Queries:", item.queries)
        elif item.type == "message":
            print("Answer:", item.content[0].text)
```

---

## Validating the Output Format

Regardless of which path ran, the response always follows the OpenAI Responses API format:

```json
{
  "output": [
    {
      "type": "file_search_call",
      "id": "fs_abc123",
      "status": "completed",
      "queries": ["What does LiteLLM support?"],
      "search_results": null
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "LiteLLM is a unified interface...",
          "annotations": [
            {
              "type": "file_citation",
              "index": 150,
              "file_id": "file-xxxx",
              "filename": "knowledge.txt"
            }
          ]
        }
      ]
    }
  ]
}
```

**Validation script:**

```python showLineNumbers title="Validate response structure"
def validate_file_search_response(response):
    """Assert that response follows OpenAI file_search output format."""
    output = response.output
    assert len(output) >= 2, "Expected at least 2 output items"

    # First item: file_search_call
    fs_call = output[0]
    fs_type = fs_call["type"] if isinstance(fs_call, dict) else fs_call.type
    assert fs_type == "file_search_call", f"Expected file_search_call, got {fs_type}"

    fs_status = fs_call["status"] if isinstance(fs_call, dict) else fs_call.status
    assert fs_status == "completed"

    # Second item: message
    msg = output[1]
    msg_type = msg["type"] if isinstance(msg, dict) else msg.type
    assert msg_type == "message"

    content = msg["content"] if isinstance(msg, dict) else msg.content
    assert len(content) > 0
    text_block = content[0]
    text = text_block["text"] if isinstance(text_block, dict) else text_block.text
    assert isinstance(text, str) and len(text) > 0

    print("✅ Response structure valid")
    print(f"   Queries: {fs_call['queries'] if isinstance(fs_call, dict) else fs_call.queries}")
    print(f"   Answer length: {len(text)} chars")
    annotations = text_block["annotations"] if isinstance(text_block, dict) else text_block.annotations
    print(f"   Citations: {len(annotations)}")

validate_file_search_response(response)
```

---

## Troubleshooting

### `UnsupportedParamsError` is raised

This means `file_search` was passed to a provider that doesn't support it natively, but the emulated fallback couldn't route either. Check:
- The model string is correct (e.g. `anthropic/claude-sonnet-4-5`, not just `claude-sonnet-4-5`)
- The `custom_llm_provider` is resolved — LiteLLM needs it to look up the provider config

### Vector store search returns no results

- Confirm the vector store ID exists and has files attached
- For LiteLLM-managed stores, ensure the file has finished processing (`status: completed`)
- Try a broader query string

### `403 Access denied` on vector store

The calling team doesn't have access to the vector store. Either:
- The vector store was created by a different team
- Use a proxy admin key to bypass team-scoped access control

### Empty `annotations` in emulated mode

The emulated path adds `file_citation` annotations only when the vector store search result includes a `file_id`. If your vector store provider doesn't return file-level metadata in search results, annotations will be empty — the answer text will still be populated.

---

## What to check next

- [File Search reference in Responses API docs](/docs/response_api#file-search-vector-stores) — full API reference
- [Vector Store management](/docs/vector_store_files) — create and manage vector stores
- [Managed vector stores](/docs/providers/bedrock_vector_store) — provider-specific setup
