# rules

- Claude on Vertex AI: reuses `anthropic/messages` shaping, moves the model into the publisher URL, and authenticates with a Google access token
- Vertex reads the API version only from the body's `anthropic_version` and answers 400 when an `anthropic-version` header names a different one, so the caller's header never reaches Vertex. A version Vertex accepts in the body is carried there, `2023-06-01`, `2023-01-01`, or no header get `vertex-2023-10-16`, and any other version drops under `drop_params` and is refused without it
- Filters the caller's and derived betas through the `vertex_ai` row of `anthropic_beta_headers_config.json`, never derives `structured-outputs`, and strips `output_config.effort` for models that do not accept it
- Logs in through `litellm-auth-gcp`'s `vertex` login, built on Google's Rust SDK (a service account's token is its own self-signed JWT), and sets the token under Python's exact `Authorization` key, so a caller `Authorization` is replaced and an `authorization` in another casing is sent beside it
- Folds an object `output_format` into `output_config.format` after the version check and refuses any other value with Python's `UnsupportedParamsError` text
- Builds a custom `api_base` URL with a narrow splitter that matches Python's URL for every realistic base (pinned by `testdata/python_urls.json`) and deliberately keeps the doubled version for a base ending in `/v1` or `/v1beta1`. Bases with `;params` or control characters differ from Python by owner ruling

# references

- Python counterpart `litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/messages/transformation.py` (`VertexAIPartnerModelsAnthropicMessagesConfig`)
- https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude
