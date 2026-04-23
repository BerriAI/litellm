# Gemini — Lyria (music generation)

Google Lyria 3 preview models are listed in LiteLLM’s [model cost map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) under the `gemini/` provider for metadata and spend tracking.

| Property | Details |
|----------|---------|
| Provider route | `gemini/` |
| Models | `gemini/lyria-3-clip-preview`, `gemini/lyria-3-pro-preview` |
| Provider docs | [Gemini API pricing / models ↗](https://ai.google.dev/gemini-api/docs/pricing) |

## Models

| Model | Notes |
|-------|--------|
| `gemini/lyria-3-clip-preview` | ~30s clip; paid tier listed as per generated song in Google’s pricing |
| `gemini/lyria-3-pro-preview` | Full song; paid tier listed as per generated song in Google’s pricing |

Input context limit in the cost map: **131,072** tokens. For modalities, limits, and features, see [Google’s Gemini API docs ↗](https://ai.google.dev/gemini-api/docs/models).

## LiteLLM behavior

- **Cost map**: Per-song paid pricing is stored as `output_cost_per_image` on those entries (flat per generation unit). Token-based completion cost may not reflect music billing until a dedicated path exists.
- **API calls**: Use the Gemini API as documented by Google. LiteLLM does not ship a separate `music_generation` helper like Veo’s `video_generation`.

## Auth

Same as other Gemini API models: `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

