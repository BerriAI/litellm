---
slug: video_characters_api
title: "New Video Characters, Edit and Extension API support"
date: 2026-03-16T10:00:00
authors:
  - name: Sameer Kankute
    title: SWE @ LiteLLM
    url: https://www.linkedin.com/in/sameer-kankute/
    image_url: https://pbs.twimg.com/profile_images/2001352686994907136/ONgNuSk5_400x400.jpg
  - name: Krrish Dholakia
    title: "CEO, LiteLLM"
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
description: "LiteLLM now supports creating, retrieving, and managing reusable video characters across multiple video generations."
tags: [videos, characters, proxy, routing]
hide_table_of_contents: false
---

LiteLLM now supoports videos character, edit and extension apis.

## What's New

Four new endpoints for video character operations:
- **Create character** - Upload a video to create a reusable asset
- **Get character** - Retrieve character metadata
- **Edit video** - Modify generated videos
- **Extend video** - Continue clips with character consistency

**Available from:** LiteLLM v1.83.0+

## Quick Example

```python
import litellm

# Create character from video
character = litellm.avideo_create_character(
    name="Luna",
    video=open("luna.mp4", "rb"),
    custom_llm_provider="openai",
    model="sora-2"
)
print(f"Character: {character.id}")

# Use in generation
video = litellm.avideo(
    model="sora-2",
    prompt="Luna dances through a magical forest.",
    characters=[{"id": character.id}],
    seconds="8"
)

# Get character info
fetched = litellm.avideo_get_character(
    character_id=character.id,
    custom_llm_provider="openai"
)

# Edit with character preserved
edited = litellm.avideo_edit(
    video_id=video.id,
    prompt="Add warm golden lighting"
)

# Extend sequence
extended = litellm.avideo_extension(
    video_id=video.id,
    prompt="Luna waves goodbye",
    seconds="5"
)
```

## Via Proxy

```bash
# Create character
curl -X POST "http://localhost:4000/v1/videos/characters" \
  -H "Authorization: Bearer sk-litellm-key" \
  -F "video=@luna.mp4" \
  -F "name=Luna"

# Get character
curl -X GET "http://localhost:4000/v1/videos/characters/char_abc123def456" \
  -H "Authorization: Bearer sk-litellm-key"

# Edit video
curl -X POST "http://localhost:4000/v1/videos/edits" \
  -H "Authorization: Bearer sk-litellm-key" \
  -H "Content-Type: application/json" \
  -d '{
    "video": {"id": "video_xyz789"},
    "prompt": "Add warm golden lighting and enhance colors"
  }'

# Extend video
curl -X POST "http://localhost:4000/v1/videos/extensions" \
  -H "Authorization: Bearer sk-litellm-key" \
  -H "Content-Type: application/json" \
  -d '{
    "video": {"id": "video_xyz789"},
    "prompt": "Luna waves goodbye and walks into the sunset",
    "seconds": "5"
  }'
```

## Managed Character IDs

LiteLLM automatically encodes provider and model metadata into character IDs:

**What happens:**
```
Upload character "Luna" with model "sora-2" on OpenAI
  ↓
LiteLLM creates: char_abc123def456 (contains provider + model_id)
  ↓
When you reference it later, LiteLLM decodes automatically
  ↓
Router knows exactly which deployment to use
```

**Behind the scenes:**
- Character ID format: `character_<base64_encoded_metadata>`
- Metadata includes: provider, model_id, original_character_id
- Transparent to you - just use the ID, LiteLLM handles routing