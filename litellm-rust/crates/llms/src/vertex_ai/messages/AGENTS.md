# rules

- Claude on Vertex AI: reuses `anthropic/messages` shaping, moves the model into the publisher URL, puts the Vertex `anthropic_version` in the body, and authenticates with a Google access token
- Applies Vertex's beta policy and strips `output_config.effort` for models that do not accept it

# references

- https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude
