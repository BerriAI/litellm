# rules

- Send the OpenAI Responses payload to `{api_base}/responses`, defaulting to `https://api.fireworks.ai/inference/v1`, with a bearer token from `api_key` or the Fireworks key variables
- Send the model as its Fireworks resource name, such as `accounts/fireworks/models/<id>`
- Fold leading text-only system and developer items into `instructions`, and replay other developer items as `system` messages, since Fireworks has no developer role
- Session affinity and forwarded user ids stay on Python
- No native WebSocket mode

# references

- https://docs.fireworks.ai/api-reference/post-responses
