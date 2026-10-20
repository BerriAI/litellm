# rules

- `transformation.rs` sends the Mistral OCR request to Vertex's Mistral publisher through `rawPredict`
- `deepseek_transformation.rs` sends DeepSeek-OCR as a Chat Completions call to Vertex's OpenAPI endpoint and parses the transcript into OCR pages. Requests default to greedy decoding with a mild repetition penalty

# references

- https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/mistral
- https://cloud.google.com/vertex-ai/generative-ai/docs/maas/deepseek
