# rules

- Claude on Bedrock InvokeModel: reuses `anthropic/messages` shaping, moves the model into the URL, and decodes the AWS event stream through `chat/invoke_handler.rs`
- `invoke_transformations/` holds only what Python's `AmazonAnthropicClaudeMessagesConfig` holds, under its method names. Helpers Python keeps in `bedrock/common_utils.py` and `bedrock/request_metadata.py` live in the matching Rust modules
- Invocation metrics in the response map onto Messages usage here
- The Invoke helpers read only `capabilities.bedrock`, which Python resolves under provider `bedrock`. The shared Anthropic transform reads the rest, which the Mantle routes resolve under their own provider
- `invoke_transformations/request.rs::transform_anthropic_messages_request` is shared by Bedrock Invoke and both Mantle routes. Mantle's differences go through `InvokeRequestSettings`, never through route checks inside the transform
- `mantle_transformation.rs` mirrors Python's `AmazonMantleMessagesConfig` (`bedrock/mantle/`) and holds the Mantle shaping that `bedrock_mantle/` reuses

# references

- https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
