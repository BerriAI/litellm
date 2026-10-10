# rules

- Shared across Vertex AI's Messages and OCR adapters: the global host, the global and default locations, the `rawPredict` methods and the Anthropic API version Vertex expects in the body
- Location validation and host selection are in `llms/src/vertex_ai/common_utils.rs`

# references

- https://cloud.google.com/vertex-ai/docs/general/locations
