---
slug: gemini_embedding_2_ga
title: "Gemini Embedding 2 (GA): Multimodal Embeddings on LiteLLM"
date: 2026-04-24T10:00:00
authors:
  - sameer
description: "Use generally available gemini-embedding-2 for multimodal embeddings on LiteLLM via Gemini API and Vertex AI—the same flows as preview, stable model id."
tags: [gemini, embeddings, multimodal, vertex ai]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Gemini Embedding 2 (GA): Multimodal Embeddings

Litellm now fully supports Gemini Embedding 2 GA.

:::info
For end-to-end behavior, input shapes, and MIME types, see the [Gemini Embedding 2 Preview walkthrough](/blog/gemini_embedding_2_multimodal). This post focuses on **GA naming**, **cost map** coverage.
:::

{/* truncate */}

## Supported Input Types

| Modality | Supported Formats |
|----------|-------------------|
| **Text** | Plain text |
| **Image** | PNG, JPEG |
| **Audio** | MP3, WAV |
| **Video** | MP4, MOV |
| **Documents** | PDF |

## Input Formats

LiteLLM accepts three input formats for multimodal content:

1. **Data URIs** – Base64-encoded inline: `data:image/png;base64,<encoded_data>`
2. **GCS URLs** – Cloud Storage paths (Vertex AI): `gs://bucket/path/to/file.png`
3. **Gemini File References** – Pre-uploaded files (Gemini API): `files/abc123`

## Quick Start

<Tabs>
<TabItem value="gemini" label="Gemini API">

```python
from litellm import embedding
import os

os.environ["GEMINI_API_KEY"] = "your-api-key"

# Text + Image (base64)
response = embedding(
    model="gemini/gemini-embedding-2",
    input=[
        "The food was delicious and the waiter...",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII"
    ],
)
print(response)
```

</TabItem>

<TabItem value="vertex" label="Vertex AI">

```python
import litellm
from litellm import embedding

litellm.vertex_project = "your-project-id"
litellm.vertex_location = "us-central1"

# Text + Image (GCS URL)
response = embedding(
    model="vertex_ai/gemini-embedding-2",
    input=[
        "Describe this image",
        "gs://my-bucket/images/photo.png"
    ],
)
print(response)
```

</TabItem>

<TabItem value="proxy" label="LiteLLM Proxy">

**1. Config (config.yaml)**

```yaml
model_list:
  - model_name: gemini-embedding-2
    litellm_params:
      model: gemini/gemini-embedding-2
      api_key: os.environ/GEMINI_API_KEY
  - model_name: vertex-gemini-embedding-2
    litellm_params:
      model: vertex_ai/gemini-embedding-2
      vertex_project: os.environ/VERTEXAI_PROJECT
      vertex_location: global

general_settings:
  master_key: sk-1234
```

**2. Start proxy**

```bash
litellm --config config.yaml
```

**3. Call embeddings** (OpenAI-compatible **`POST /v1/embeddings`** on the proxy)

```bash
curl -sS -X POST http://localhost:4000/v1/embeddings \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-embedding-2",
    "input": [
      "The food was delicious and the waiter...",
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII"
    ]
  }'
```

</TabItem>
</Tabs>

## Input Format Examples

| Format | Example | Provider |
|--------|---------|----------|
| **Data URI** | `data:image/png;base64,...` | Gemini, Vertex AI |
| **GCS URL** | `gs://bucket/path/image.png` | Vertex AI |
| **File reference** | `files/abc123` | Gemini API only |

### Supported MIME Types for Data URIs

- **Images:** `image/png`, `image/jpeg`
- **Audio:** `audio/mpeg`, `audio/wav`
- **Video:** `video/mp4`, `video/quicktime`
- **Documents:** `application/pdf`

### GCS URL MIME Inference

For Vertex AI, MIME types are inferred from file extensions:

- `.png` → `image/png`
- `.jpg` / `.jpeg` → `image/jpeg`
- `.mp3` → `audio/mpeg`
- `.wav` → `audio/wav`
- `.mp4` → `video/mp4`
- `.mov` → `video/quicktime`
- `.pdf` → `application/pdf`

## Optional Parameters

| Parameter | Description | Maps to |
|-----------|-------------|---------|
| `dimensions` | Output embedding size | `outputDimensionality` |

```python
response = embedding(
    model="gemini/gemini-embedding-2",
    input=["text to embed"],
    dimensions=768,  # Optional: control output vector size
)
```
