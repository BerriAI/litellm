---
slug: responses_api_file_search
title: "Generic file_search in the Responses API: RAG for Any Model"
date: 2026-03-16T10:00:00
authors:
  - name: Sameer Kankute
    title: SWE @ LiteLLM (LLM Translation)
    url: https://www.linkedin.com/in/sameer-kankute/
    image_url: https://pbs.twimg.com/profile_images/2001352686994907136/ONgNuSk5_400x400.jpg
description: "Use the standard OpenAI Responses API file_search tool with Claude, Gemini, GPT-4, and any LiteLLM-supported model. LiteLLM intercepts for non-native providers, searches your vector stores, and injects context automatically."
tags: [responses api, file_search, vector stores, rag, openai]
hide_table_of_contents: false
---

# Generic file_search in the Responses API

The **`file_search` tool** in the OpenAI Responses API lets models query vector stores for retrieval-augmented generation (RAG). Until now, only OpenAI and Azure natively supported it. LiteLLM now extends this to **all providers**—Claude, Gemini, Vertex AI, Bedrock, and more—via a single, unified contract.

Pass `tools=[{"type": "file_search", "vector_store_ids": ["vs_abc123"]}]` and LiteLLM handles the rest.

## Architecture

```mermaid
flowchart TD
    Request["tools=[{type: file_search, vector_store_ids: [...]}]"]
    Request --> CheckProvider["check supports_native_file_search(provider)"]
    CheckProvider --> HasFileSearch{"has file_search tools?"}
    HasFileSearch -->|no| Handler["existing handler"]
    HasFileSearch -->|yes| Native{"native provider?"}
    Native -->|"YES: OpenAI / Azure"| PassThrough["pass file_search through unchanged"]
    PassThrough --> Handler
    Native -->|"NO: Claude / Gemini / etc."| Intercept["FileSearchResponsesAPIUtils"]
    Intercept --> ExtractQuery["extract query from input"]
    ExtractQuery --> Asearch["litellm.vector_stores.asearch per vector_store_id"]
    Asearch --> InjectContext["inject context into input"]
    InjectContext --> StripTool["strip file_search from tools"]
    StripTool --> StoreResults["store results for post-call hook"]
    StoreResults --> Handler
    Handler --> Route["base_llm_http_handler OR litellm_completion_transformation_handler"]
```

**Native path (OpenAI, Azure):** The `file_search` tool is forwarded to the provider as-is. The provider performs retrieval and returns citations in its response.

**Non-native path (Claude, Gemini, etc.):** LiteLLM intercepts the request before routing:

1. Extracts the query from the last user message in `input`
2. Calls `litellm.vector_stores.asearch()` for each `vector_store_id`
3. Builds a context string from the retrieved chunks
4. Injects the context as a user message before the original input
5. Strips the `file_search` tool from `tools` (other tools are preserved)
6. Forwards the enriched request to the model
7. Stores search results in `model_call_details["search_results"]` for logging and citations

## Usage

```python
import litellm

# Works with any provider—Claude, Gemini, GPT-4, etc.
response = litellm.responses(
    model="anthropic/claude-opus-4-5",
    input="What does our docs say about testing?",
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": ["vs_abc123"],
        }
    ],
)
print(response)
```

Via the LiteLLM Proxy with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-proxy-api-key",
)

response = client.responses.create(
    model="anthropic/claude-opus-4-5",
    input="Summarise the company handbook.",
    tools=[{"type": "file_search", "vector_store_ids": ["vs_handbook_abc"]}],
)
```

## Prerequisites

- **Vector stores** must be created and populated (e.g. via [Create a Vector Store](/docs/vector_stores/create)).
- **`vector_store_registry`** must be configured in the proxy `config.yaml` or via the Python SDK so LiteLLM can resolve each `vector_store_id` to the correct provider and credentials.

## Learn more

- [Responses API file_search docs](/docs/response_api#file_search-vector-store-rag)
- [Vector Store Create](/docs/vector_stores/create)
- [Vector Store Search](/docs/vector_stores/search)
