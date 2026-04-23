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

---

## Complexity Router

The Complexity Router provides an alternative to semantic routing that uses **rule-based scoring** to classify requests by complexity and route them to appropriate models — with **zero external API calls** and **sub-millisecond latency**.

### When to Use

| Feature | Semantic Auto Router | Complexity Router |
|---------|---------------------|-------------------|
| Classification | Embedding-based matching | Rule-based scoring |
| Latency | ~100-500ms (embedding API) | &lt;1ms |
| API Calls | Requires embedding model | None |
| Training | Requires utterance examples | Works out of the box |
| Best For | Intent-based routing | Cost optimization |

Use **Complexity Router** when you want to:
- Route simple queries to cheaper/faster models (e.g., gpt-4o-mini)
- Route complex queries to more capable models (e.g., claude-sonnet-4)
- Minimize latency overhead from routing decisions
- Avoid additional API costs for embeddings

### LiteLLM Python SDK

```python
from litellm import Router

router = Router(
    model_list=[
        # Target models for each tier
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "gpt-4o-mini"},
        },
        {
            "model_name": "gpt-4o",
            "litellm_params": {"model": "gpt-4o"},
        },
        {
            "model_name": "claude-sonnet",
            "litellm_params": {"model": "claude-sonnet-4-20250514"},
        },
        {
            "model_name": "o1-preview",
            "litellm_params": {"model": "o1-preview"},
        },
        # Complexity router configuration
        {
            "model_name": "smart-router",
            "litellm_params": {
                "model": "auto_router/complexity_router",
                "complexity_router_config": {
                    "tiers": {
                        "SIMPLE": "gpt-4o-mini",
                        "MEDIUM": "gpt-4o",
                        "COMPLEX": "claude-sonnet",
                        "REASONING": "o1-preview",
                    },
                },
                "complexity_router_default_model": "gpt-4o",
            },
        },
    ],
)
```

#### Usage

```python
# Simple query → routes to gpt-4o-mini
response = await router.acompletion(
    model="smart-router",
    messages=[{"role": "user", "content": "What is 2+2?"}],
)

# Complex technical query → routes to claude-sonnet or higher
response = await router.acompletion(
    model="smart-router",
    messages=[{"role": "user", "content": "Design a distributed microservice architecture with Kubernetes orchestration"}],
)

# Reasoning request → routes to o1-preview
response = await router.acompletion(
    model="smart-router",
    messages=[{"role": "user", "content": "Think step by step and reason through this problem carefully..."}],
)
```

### LiteLLM Proxy Server

Add the complexity router to your `config.yaml`:

```yaml
model_list:
  # Target models
  - model_name: gpt-4o-mini
    litellm_params:
      model: gpt-4o-mini
      
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      
  - model_name: claude-sonnet
    litellm_params:
      model: claude-sonnet-4-20250514
      
  - model_name: o1-preview
    litellm_params:
      model: o1-preview

  # Complexity router
  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        tiers:
          SIMPLE: gpt-4o-mini
          MEDIUM: gpt-4o
          COMPLEX: claude-sonnet
          REASONING: o1-preview
      complexity_router_default_model: gpt-4o
```

### Configuration Options

#### Tier Boundaries

Customize the score thresholds for each tier:

```yaml
complexity_router_config:
  tiers:
    SIMPLE: gpt-4o-mini
    MEDIUM: gpt-4o
    COMPLEX: claude-sonnet
    REASONING: o1-preview
  tier_boundaries:
    simple_medium: 0.15    # Below 0.15 → SIMPLE
    medium_complex: 0.35   # 0.15-0.35 → MEDIUM
    complex_reasoning: 0.60  # 0.35-0.60 → COMPLEX, above → REASONING
```

#### Token Thresholds

Adjust when prompts are considered "short" or "long":

```yaml
complexity_router_config:
  token_thresholds:
    simple: 15   # Prompts under 15 tokens are penalized (simple indicator)
    complex: 400 # Prompts over 400 tokens get complexity boost
```

#### Dimension Weights

Customize how much each signal contributes to the complexity score:

```yaml
complexity_router_config:
  dimension_weights:
    tokenCount: 0.10        # Prompt length
    codePresence: 0.30      # Code-related keywords
    reasoningMarkers: 0.25  # "step by step", "think through", etc.
    technicalTerms: 0.25    # Domain-specific complexity
    simpleIndicators: 0.05  # "what is", "define", greetings
    multiStepPatterns: 0.03 # "first...then", numbered steps
    questionComplexity: 0.02 # Multiple questions
```

### How Complexity Routing Works

The router scores each request across 7 dimensions:

| Dimension | What It Detects | Effect |
|-----------|-----------------|--------|
| Token Count | Short (&lt;15) or long (&gt;400) prompts | Short = simple, long = complex |
| Code Presence | "function", "class", "api", "database", etc. | Increases complexity |
| Reasoning Markers | "step by step", "think through", "analyze" | Triggers REASONING tier |
| Technical Terms | "architecture", "distributed", "encryption" | Increases complexity |
| Simple Indicators | "what is", "define", "hello" | Decreases complexity |
| Multi-Step Patterns | "first...then", "1. 2. 3." | Increases complexity |
| Question Complexity | Multiple question marks | Increases complexity |

**Special behavior:** If 2+ reasoning markers are detected in the user message, the request automatically routes to the REASONING tier regardless of the weighted score.

