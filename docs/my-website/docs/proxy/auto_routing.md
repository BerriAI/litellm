import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Auto Routing

LiteLLM can auto select the best model for a request based on rules you define.

<Image alt="Auto Routing" img={require('../../img/auto_router.png')} style={{ borderRadius: '8px', marginBottom: '1em', maxWidth: '100%' }} />

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


## LiteLLM Proxy Server

### Setup

Navigate to the LiteLLM UI and go to **Models+Endpoints** > **Add Model** > **Auto Router Tab**.

Configure the following required fields:

- **Auto Router Name** - The model name that developers will use when making LLM API requests to LiteLLM
- **Default Model** - The fallback model used when no route is matched (e.g., if set to "gpt-4o-mini", unmatched requests will be routed to gpt-4o-mini)
- **Embedding Model** - The model used to generate embeddings for input messages. These embeddings are used to semantically match input against the utterances defined in your routes

#### Route Configuration

<Image alt="Auto Router Setup" img={require('../../img/auto_router2.png')} style={{ borderRadius: '8px', marginBottom: '1em', maxWidth: '100%' }} />

<br />

<br />

Click **Add Route** to create a new routing rule. Each route consists of utterances that are matched against input messages to determine the target model.

Configure each route with:

- **Utterances** - Example phrases that will trigger this route. Use placeholders in brackets for variables:

```json
"how to code a program in [language]",
"can you explain this [language] code",
"can you explain this [language] script",
"can you convert this [language] code to [target_language]"
```

- **Description** - A human-readable description of what this route handles
- **Score Threshold** - The minimum similarity score (0.0-1.0) required to trigger this route


### Usage

Once added developers need to select the model=`auto_router1` in the `model` field of the LLM API request.

<Tabs>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="sk-1234", # replace with your LiteLLM API key
    base_url="http://localhost:4000"
)

# This request will be auto-routed based on the content
response = client.chat.completions.create(
    model="auto_router1",
    messages=[
        {
            "role": "user",
            "content": "how to code a program in python"
        }
    ]
)

print(response)
```
</TabItem>

<TabItem value="curl" label="Curl Request">

```shell
curl -X POST http://localhost:4000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $LITELLM_API_KEY" \
-d '{
    "model": "auto_router1",
    "messages": [{"role": "user", "content": "how to code a program in python"}]
}'
```
</TabItem>
</Tabs>



## How It Works

1. When a request comes in, LiteLLM generates embeddings for the input message
2. It compares these embeddings against the utterances defined in your routes
3. If a route's similarity score exceeds the threshold, the request is routed to that model
4. If no route matches, the request goes to the default model

