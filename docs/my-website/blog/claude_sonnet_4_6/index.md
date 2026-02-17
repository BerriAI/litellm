---
slug: claude_sonnet_4_6
title: "Day 0 Support: Claude Sonnet 4.6"
date: 2026-02-17T10:00:00
authors:
  - name: Ishaan Jaff
    title: "CTO, LiteLLM"
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
  - name: Krrish Dholakia
    title: "CEO, LiteLLM"
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
description: "Day 0 support for Claude Sonnet 4.6 on LiteLLM AI Gateway - use across Anthropic, Azure, Vertex AI, and Bedrock."
tags: [anthropic, claude, sonnet 4.6]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

LiteLLM now supports Claude Sonnet 4.6 on Day 0. Use it across Anthropic, Azure, Vertex AI, and Bedrock through the LiteLLM AI Gateway.

## Docker Image

```bash
docker pull ghcr.io/berriai/litellm:v1.81.3-stable.sonnet-4-6
```

## Usage - Anthropic

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-sonnet-4-6
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/config.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:v1.81.3-stable.sonnet-4-6 \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-sonnet-4-6",
  "messages": [
    {
      "role": "user",
      "content": "what llm are you"
    }
  ]
}'
```

</TabItem>

<TabItem value="sdk" label="LiteLLM SDK">

```python
from litellm import completion

response = completion(
    model="anthropic/claude-sonnet-4-6",
    messages=[{"role": "user", "content": "what llm are you"}]
)
print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

## Usage - Azure

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-sonnet-4-6
    litellm_params:
      model: azure_ai/claude-sonnet-4-6
      api_key: os.environ/AZURE_AI_API_KEY
      api_base: os.environ/AZURE_AI_API_BASE  # https://<resource>.services.ai.azure.com
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e AZURE_AI_API_KEY=$AZURE_AI_API_KEY \
  -e AZURE_AI_API_BASE=$AZURE_AI_API_BASE \
  -v $(pwd)/config.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:v1.81.3-stable.sonnet-4-6 \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-sonnet-4-6",
  "messages": [
    {
      "role": "user",
      "content": "what llm are you"
    }
  ]
}'
```

</TabItem>

<TabItem value="sdk" label="LiteLLM SDK">

```python
from litellm import completion

response = completion(
    model="azure_ai/claude-sonnet-4-6",
    api_key="your-azure-api-key",
    api_base="https://<resource>.services.ai.azure.com",
    messages=[{"role": "user", "content": "what llm are you"}]
)
print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

## Usage - Vertex AI

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-sonnet-4-6
    litellm_params:
      model: vertex_ai/claude-sonnet-4-6
      vertex_project: os.environ/VERTEX_PROJECT
      vertex_location: us-east5
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e VERTEX_PROJECT=$VERTEX_PROJECT \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/credentials.json:/app/credentials.json \
  ghcr.io/berriai/litellm:v1.81.3-stable.sonnet-4-6 \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-sonnet-4-6",
  "messages": [
    {
      "role": "user",
      "content": "what llm are you"
    }
  ]
}'
```

</TabItem>

<TabItem value="sdk" label="LiteLLM SDK">

```python
from litellm import completion

response = completion(
    model="vertex_ai/claude-sonnet-4-6",
    vertex_project="your-project-id",
    vertex_location="us-east5",
    messages=[{"role": "user", "content": "what llm are you"}]
)
print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

## Usage - Bedrock

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-sonnet-4-6
    litellm_params:
      model: bedrock/anthropic.claude-sonnet-4-6-v1
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -v $(pwd)/config.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:v1.81.3-stable.sonnet-4-6 \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "claude-sonnet-4-6",
  "messages": [
    {
      "role": "user",
      "content": "what llm are you"
    }
  ]
}'
```

</TabItem>

<TabItem value="sdk" label="LiteLLM SDK">

```python
from litellm import completion

response = completion(
    model="bedrock/anthropic.claude-sonnet-4-6-v1",
    aws_access_key_id="your-access-key",
    aws_secret_access_key="your-secret-key",
    aws_region_name="us-east-1",
    messages=[{"role": "user", "content": "what llm are you"}]
)
print(response.choices[0].message.content)
```

</TabItem>
</Tabs>
