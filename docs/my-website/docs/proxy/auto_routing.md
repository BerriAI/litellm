# Auto Routing

LiteLLM can auto select the best model for a request based on rules you define.

## LiteLLM Python SDK

Auto routing allows you to define routing rules that automatically select the best model for a request based on the input content. This is useful for directing different types of queries to specialized models.

### Setup

1. **Create a router configuration file** (e.g., `router.json`):

```json
{
    "encoder_type": "openai",
    "encoder_name": "text-embedding-3-large",
    "routes": [
        {
            "name": "litellm-gpt-4.1",
            "utterances": [
                "litellm is great"
            ],
            "description": "positive affirmation",
            "function_schemas": null,
            "llm": null,
            "score_threshold": 0.5,
            "metadata": {}
        },
        {
            "name": "litellm-claude-35",
            "utterances": [
                "how to code a program in [language]"
            ],
            "description": "coding assistant",
            "function_schemas": null,
            "llm": null,
            "score_threshold": 0.5,
            "metadata": {}
        }
    ]
}
```

2. **Configure the Router with auto routing models**:

```python
from litellm import Router
import os

router = Router(
    model_list=[
        # Embedding models for routing
        {
            "model_name": "custom-text-embedding-model",
            "litellm_params": {
                "model": "text-embedding-3-large",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        # Your target models
        {
            "model_name": "litellm-gpt-4.1",
            "litellm_params": {
                "model": "gpt-4.1",
            },
            "model_info": {"id": "openai-id"},
        },
        {
            "model_name": "litellm-claude-35",
            "litellm_params": {
                "model": "claude-3-5-sonnet-latest",
            },
            "model_info": {"id": "claude-id"},
        },
        # Auto router configuration
        {
            "model_name": "auto_router1",
            "litellm_params": {
                "model": "auto_router/auto_router_1",
                "auto_router_config_path": "router.json",
                "auto_router_default_model": "gpt-4o-mini",
                "auto_router_embedding_model": "custom-text-embedding-model",
            },
        },
    ],
)
```

### Usage

Once configured, use the auto router by calling it with your auto router model name:

```python
# This request will be routed to gpt-4.1 based on the utterance match
response = await router.acompletion(
    model="auto_router1",
    messages=[{"role": "user", "content": "litellm is great"}],
)

# This request will be routed to claude-3-5-sonnet-latest for coding queries
response = await router.acompletion(
    model="auto_router1",
    messages=[{"role": "user", "content": "how to code a program in python"}],
)
```

### Configuration Parameters

- **auto_router_config_path**: Path to your router.json configuration file
- **auto_router_default_model**: Fallback model when no route matches
- **auto_router_embedding_model**: Model used for generating embeddings to match against utterances

### Router Configuration Schema

The `router.json` file supports the following structure:

- **encoder_type**: Type of encoder (e.g., "openai")
- **encoder_name**: Name of the embedding model
- **routes**: Array of routing rules with:
  - **name**: Target model name (must match a model in your model_list)
  - **utterances**: Example phrases/patterns to match against
  - **description**: Human-readable description of the route
  - **score_threshold**: Minimum similarity score to trigger this route (0.0-1.0)
  - **metadata**: Additional metadata for the route

## How It Works

1. When a request comes in, LiteLLM generates embeddings for the input message
2. It compares these embeddings against the utterances defined in your routes
3. If a route's similarity score exceeds the threshold, the request is routed to that model
4. If no route matches, the request goes to the default model

