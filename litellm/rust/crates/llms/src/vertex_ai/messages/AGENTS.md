# rules

- Claude on Vertex AI: reuses `anthropic/messages` shaping, moves the model into the publisher URL, and authenticates with a Google access token
- Vertex reads the API version only from the body's `anthropic_version` and answers 400 when an `anthropic-version` header names a different one, so the caller's header never reaches Vertex. A version Vertex accepts in the body is carried there, `2023-06-01`, `2023-01-01`, or no header get `vertex-2023-10-16`, and any other version drops under `drop_params` and is refused without it
- Applies Vertex's beta policy and strips `output_config.effort` for models that do not accept it

# references

- https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude
