# Using Knowledge Bases with LiteLLM

LiteLLM integrates with AWS Bedrock Knowledge Bases, allowing your models to access your organization's data for more accurate and contextually relevant responses.

## Quick Start

In order to use a Bedrock Knowledge Base with LiteLLM, you need to pass `knowledge_bases` as a parameter to the completion request. Where `knowledge_bases` is a list of Bedrock Knowledge Base IDs.

### LiteLLM Python SDK

```python showLineNumbers title="Basic Bedrock Knowledge Base Usage"
import os
import litellm


# Make a completion request with knowledge_bases parameter
response = await litellm.acompletion(
    model="anthropic/claude-3-5-sonnet", 
    messages=[{"role": "user", "content": "What is litellm?"}],
    knowledge_bases=["YOUR_KNOWLEDGE_BASE_ID"]  # e.g., "T37J8R4WTM"
)

print(response.choices[0].message.content)
```

### LiteLLM Proxy

#### 1. Configure your proxy

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet
      api_key: os.environ/ANTHROPIC_API_KEY

```

#### 2. Make a request with knowledge_bases parameter

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="curl" label="Curl">

```bash showLineNumbers title="Curl Request to LiteLLM Proxy"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "What is litellm?"}],
    "knowledge_bases": ["YOUR_KNOWLEDGE_BASE_ID"]
  }'
```

</TabItem>

<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python showLineNumbers title="OpenAI Python SDK Request"
from openai import OpenAI

# Initialize client with your LiteLLM proxy URL
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-litellm-api-key"
)

# Make a completion request with knowledge_bases parameter
response = client.chat.completions.create(
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "What is litellm?"}],
    extra_body={"knowledge_bases": ["YOUR_KNOWLEDGE_BASE_ID"]}
)

print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

## How It Works

LiteLLM implements a `BedrockKnowledgeBaseHook` that intercepts your completion requests for handling the integration with Bedrock Knowledge Bases.

1. You make a completion request with the `knowledge_bases` parameter
2. LiteLLM automatically:
   - Uses your last message as the query to retrieve relevant information from the Knowledge Base
   - Adds the retrieved context to your conversation
   - Sends the augmented messages to the model

### Example Transformation

When you pass `knowledge_bases=["YOUR_KNOWLEDGE_BASE_ID"]`, your request flows through these steps:

**1. Original Request to LiteLLM:**
```json
{
    "model": "anthropic/claude-3-5-sonnet",
    "messages": [
        {"role": "user", "content": "What is litellm?"}
    ],
    "knowledge_bases": ["YOUR_KNOWLEDGE_BASE_ID"]
}
```

**2. Request to AWS Bedrock Knowledge Base:**
```json
{
    "retrievalQuery": {
        "text": "What is litellm?"
    }
}
```
This is sent to: `https://bedrock-agent-runtime.{aws_region}.amazonaws.com/knowledgebases/YOUR_KNOWLEDGE_BASE_ID/retrieve`

**3. Final Request to LiteLLM:**
```json
{
    "model": "anthropic/claude-3-5-sonnet",
    "messages": [
        {"role": "user", "content": "What is litellm?"},
        {"role": "user", "content": "Context: \n\nLiteLLM is an open-source SDK to simplify LLM API calls across providers (OpenAI, Claude, etc). It provides a standardized interface with robust error handling, streaming, and observability tools."}
    ]
}
```

This process happens automatically whenever you include the `knowledge_bases` parameter in your request.

## API Reference

### LiteLLM Completion Knowledge Base Parameters

When using the Knowledge Base integration with LiteLLM, you can include the following parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `knowledge_bases` | List[str] | List of Bedrock Knowledge Base IDs to query |
