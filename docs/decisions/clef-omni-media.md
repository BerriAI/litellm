# Clef-omni audio and video

Configure a model with `model: cloudflare/clef-omni` and set `CLOUDFLARE_API_KEY` and `CLOUDFLARE_ACCOUNT_ID`. An explicit `api_base` works as well. The fully qualified `cloudflare/@cf/cloudflare/clef-omni` model name is supported

Send embedded clips in the top-level `audio` and `videos` lists. Each entry is either a base64 data URL or an object with `content_type` and `base64`. LiteLLM validates the shape, media type and clip count before sending a request. Cloudflare validates the decoded bytes, sizes and durations

```python
import base64
from pathlib import Path
import litellm

result = litellm.decisions(
    model="cloudflare/clef-omni",
    state="Evaluate these clips.",
    questions={
        "speech": {"type": "noul", "instructions": "Is there speech?"},
        "red": {"type": "noul", "instructions": "Does the video show red?"},
    },
    audio=[{
        "content_type": "audio/wav",
        "base64": base64.b64encode(Path("clip.wav").read_bytes()).decode(),
    }],
    videos=["data:video/mp4;base64," + base64.b64encode(Path("clip.mp4").read_bytes()).decode()],
)
print(result.answers)
```

The same fields work with `await litellm.adecisions(...)` and the proxy routes `/v1/systemone` and `/systemone`

For `/v1/decisions` and `/decisions`, use the OpenAI-style `input` and `questions` fields, with the same top-level `audio` and `videos` extensions. These are LiteLLM extensions for Clef-omni, not OpenAI upstream media support. `input_audio` and `input_video` content parts are not introduced by this change

```json
{
  "model": "your-clef-omni-alias",
  "input": "Evaluate the clips.",
  "questions": [{"type": "predicate", "name": "speech", "instructions": "Is there speech?"}],
  "audio": [{"content_type": "audio/wav", "base64": "<base64-encoded WAV bytes>"}],
  "videos": ["data:video/mp4;base64,<base64-encoded MP4 bytes>"]
}
```

Remote URLs are rejected. Empty lists behave like text-only requests. Nonempty audio or video lists are rejected for other Decisions models, including Clef and Clef-flash, rather than silently discarded

Cloudflare's [Clef-omni documentation](https://developers.cloudflare.com/workers-ai/models/clef-omni/) allows four audio clips (8 MiB and 300 seconds each), two videos (16 MiB and 60 seconds each), and at most 16 MiB of audio and video combined. Videos are sampled at two frames per second. Media consumes input tokens at the model's input price. Cloudflare handles decoding, duration limits and context-window limits

This change is independent of the image support in PR #45762. It does not add native image handling.
