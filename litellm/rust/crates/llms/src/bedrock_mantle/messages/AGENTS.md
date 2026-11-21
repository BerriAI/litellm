# rules

- The `bedrock_mantle` provider reuses the `bedrock/mantle/` route's shaping in `bedrock/messages/mantle/`
- Both routes run the Bedrock Invoke request transform with their `InvokeRequestSettings`, then move the betas into the header and put `model` and `stream` back in the body
- Auth, region and URL order copy Python per route, quirks included, and forwarded `Authorization` / `x-api-key` follow Python's `_sign_request`
- `mantle_capability_lookup` names the keys Python probes on both routes: the model id the route sends, under the route's provider for the shared Anthropic transform and under `bedrock` for the Invoke helpers. The Python host and the router both read them from the Rust catalog through `messages_route_capabilities`
- `bedrock/mantle/` is slated for deprecation in favour of `bedrock_mantle/`. Folding it in must keep its defaults (us-west-2, no BEDROCK_MANTLE_API_KEY, its URL building, no claude gate) or ship with a deprecation notice

# references

- Python counterpart `litellm/llms/bedrock_mantle/messages/transformation.py` (`BedrockMantleAnthropicMessagesConfig`)
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
