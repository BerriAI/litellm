---
slug: video_characters_api
title: "Reusable Video Characters with LiteLLM"
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

Upload a video character once, reference it across unlimited generations. LiteLLM now handles character management with full router support.

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
curl -X GET "http://localhost:4000/v1/videos/characters/char_xyz123" \
  -H "Authorization: Bearer sk-litellm-key"
```

## Key Features

✅ **Full Router Support** - Load balance across multiple model deployments  
✅ **Character Encoding** - Automatic provider/model tracking in character IDs  
✅ **Error Handling** - Proper HTTP status checks before response parsing  
✅ **Backward Compatible** - External providers receive NotImplementedError, not instantiation errors  
✅ **Multi-Deployment** - Router picks optimal deployment when target_model_names is set

## Best Practices

**Character uploads:**
- 2-4 seconds optimal
- Match target resolution (16:9, 9:16, or 1:1)
- 720p-1080p
- Clear character isolation

**Prompting:**
```
✅ "Luna the fox dances through a cosmic forest, stars trailing her movement"
❌ "A character that looks like Luna"
```

Always mention character name verbatim in prompt.

## Implementation Notes

All four handler methods now include:
- `response.raise_for_status()` - Proper error detection before model parsing
- Router-first dispatch - Consistent with avideo_edit/extension
- Async support - Full async/await pattern

## What's Inside

- 8 handler methods (sync + async pairs)
- Character transformation classes
- SDK functions + Router wiring
- Full test coverage
- Comprehensive error handling

## Common Issues

**Character doesn't appear?**
- Include character ID in `characters` array
- Use character name in prompt (exact match)
- Ensure character occupies meaningful screen space

**Distorted character?**
- Character video aspect ratio must match target resolution
- Upload again with matching dimensions

**Want to edit with character?**
- Use avideo_edit (currently no character support in extensions)
- Edit preserves original composition

## Next Steps

- Try the [examples](https://docs.litellm.ai/video_characters)
- Check out [character best practices](https://docs.litellm.ai/docs/video_characters#best-practices)
- Deploy the proxy and start routing

**Resources:**
- [Docs](https://docs.litellm.ai/docs/video_characters)
- [SDK Reference](https://github.com/BerriAI/litellm)
- [Support](https://github.com/BerriAI/litellm/issues)
