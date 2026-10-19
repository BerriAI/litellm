# rules

- Gemini `generateContent` for Chat Completions, shared by Vertex AI and Google AI Studio (`gemini/chat`) through `GeminiSurface`
- `transformation.rs` turns OpenAI messages into `contents` and `systemInstruction`, `schema.rs` turns JSON schemas into Gemini's subset, `vertex_and_google_ai_studio_gemini.rs` maps params, builds URLs and turns responses back, `handler.rs` turns `alt=sse` stream events into chunks
- Only Gemini models route here; Claude and partner models on Vertex keep their own configs
- Vertex authenticates with a Google access token, or forwards a caller's `Authorization` header

# references

- https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
- https://ai.google.dev/api/generate-content
- litellm/llms/vertex_ai/gemini/transformation.py
- litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py
