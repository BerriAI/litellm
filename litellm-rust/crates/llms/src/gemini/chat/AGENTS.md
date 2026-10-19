# rules

- Gemini on Google AI Studio: reuses `vertex_ai/gemini` for params, the body, the response and the stream, with `GeminiSurface::AiStudio`
- The API key goes in the `x-goog-api-key` header, never in the URL
- Gemini 3 and newer call `v1alpha`, older models `v1beta`; a custom `api_base` is called at `{api_base}/models/{model}:generateContent`

# references

- https://ai.google.dev/api/generate-content
- litellm/llms/gemini/chat/transformation.py
- litellm/llms/vertex_ai/vertex_llm_base.py
