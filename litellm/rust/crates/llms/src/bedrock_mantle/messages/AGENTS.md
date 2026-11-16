# rules

- Mirrors Python's `BedrockMantleAnthropicMessagesConfig` (`bedrock_mantle/`). `bedrock/mantle/` is `bedrock/messages/mantle_transformation.rs`, as in Python
- Both routes run the Bedrock Invoke request transform with their `InvokeRequestSettings`, then move the betas into the header and put `model` and `stream` back in the body
- Auth, region and URL order copy Python per route, quirks included, and forwarded `Authorization` / `x-api-key` follow Python's `_sign_request`
- `bedrock/mantle/` is slated for deprecation in favour of `bedrock_mantle/`. Folding it in must keep its defaults (us-west-2, no BEDROCK_MANTLE_API_KEY, its URL building, no claude gate) or ship with a deprecation notice

# references

- Python counterparts `litellm/llms/bedrock_mantle/messages/transformation.py` and `litellm/llms/bedrock/messages/mantle_transformation.py`
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
