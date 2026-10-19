# rules

- Mistral at `https://api.mistral.ai/v1`; a base without `/v1` gets it appended
- `tool_choice` narrows to `auto`/`any`/`none`, `seed` goes as `random_seed`, magistral reasoning adds Mistral's system prompt
- Content blocks in answers and chunks become text plus `reasoning_content`
- Shared flow and response handling: `../../openai/chat/compatible.rs`

# references

- https://docs.mistral.ai/api/#tag/chat
- litellm/llms/mistral/chat/transformation.py
