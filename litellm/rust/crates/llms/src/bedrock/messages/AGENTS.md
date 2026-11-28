# rules

- Bedrock serves Messages on three routes: `invoke/` (InvokeModel), `claude_platform/` (the `bedrock/claude_platform/` route) and the `bedrock/mantle/` route, which `BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG` in `bedrock_mantle/messages/` serves next to the `bedrock_mantle` provider
- Helpers Python keeps in `bedrock/common_utils.py` and `bedrock/request_metadata.py` live in the matching Rust modules
- The Invoke helpers read only `capabilities.bedrock`, which Python resolves under provider `bedrock`. The shared Anthropic transform reads the rest, which the Mantle routes resolve under their own provider
- `invoke/request.rs::transform_anthropic_messages_request` is shared by Bedrock Invoke and both Mantle routes. Mantle's differences go through `InvokeRequestSettings`, never through route checks inside the transform

# references

- Each route's upstream docs and Python counterpart are in its own AGENTS.md, the `bedrock/mantle/` route's in `bedrock_mantle/messages/AGENTS.md`
