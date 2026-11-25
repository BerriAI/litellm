# rules

- `BedrockMantleMessagesConfig` serves both Mantle routes, `BEDROCK_MANTLE_MESSAGES_CONFIG` for the `bedrock_mantle` provider and `BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG` for the `bedrock/mantle/` route. The routes differ only through `MantleRoute` and their `InvokeRequestSettings`, never through a second config type
- Both routes run the Bedrock Invoke request transform with their `InvokeRequestSettings`, then move the betas into the header and put `model` and `stream` back in the body
- Auth, region and URL order copy Python per route, quirks included, and forwarded `Authorization` / `x-api-key` follow Python's `_sign_request`
- `bedrock/mantle/` keeps its own defaults: Bedrock's region order and us-west-2 default, only `AWS_BEARER_TOKEN_BEDROCK`, `build_mantle_messages_url`, and no claude gate in `messages_provider`
- `mantle_capability_lookup` names the keys Python probes on both routes: the model id the route sends, under the route's provider for the shared Anthropic transform and under `bedrock` for the Invoke helpers. The Python host and the router both read them from the Rust catalog through `messages_route_capabilities`

# references

- Python counterpart of `BEDROCK_MANTLE_MESSAGES_CONFIG` `litellm/llms/bedrock_mantle/messages/transformation.py` (`BedrockMantleAnthropicMessagesConfig`)
- Python counterpart of `BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG` `litellm/llms/bedrock/messages/mantle_transformation.py` (`AmazonMantleMessagesConfig`)
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
