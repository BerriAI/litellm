# rules

- Chat Completions translated onto Anthropic Messages. Both format specs live under `llms-types/src/formats/`
- `top_k` is not forwarded, because Python gates it per model inside `transform_request`
