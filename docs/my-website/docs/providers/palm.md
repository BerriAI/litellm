# Google PaLM

:::warning PaLM was decommissioned
Google decommissioned PaLM in October 2024.

LiteLLM keeps the `palm/` route only to surface a clear migration error. New requests should use the `gemini/` route instead.
:::

| Property | Details |
|-------|-------|
| Status | Deprecated / decommissioned |
| Legacy Provider Route on LiteLLM | `palm/` |
| Recommended Replacement | [`gemini/`](./gemini.md) |
| Deprecation Announcement | [Google PaLM deprecation notice](https://ai.google.dev/palm_docs/palm?hl=en) |

## What happens in LiteLLM

If you call LiteLLM with a `palm/` model, LiteLLM raises an error that instructs you to migrate to `gemini/`.

## Migration

Replace:

```python
from litellm import completion

completion(
    model="palm/chat-bison",
    messages=[{"role": "user", "content": "Hello"}],
)
```

With:

```python
import os
from litellm import completion

os.environ["GEMINI_API_KEY"] = "your-api-key"

completion(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "Hello"}],
)
```

:::tip Legacy API keys
When you migrate to `gemini/`, prefer `GEMINI_API_KEY`.

LiteLLM still falls back to `PALM_API_KEY` for compatibility, but `GEMINI_API_KEY` is the current env var.
:::
