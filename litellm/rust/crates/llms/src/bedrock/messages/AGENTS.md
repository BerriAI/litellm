# rules

- Claude on Bedrock InvokeModel: reuses `anthropic/messages` shaping, moves the model into the URL, and decodes the AWS event stream through `chat/invoke_handler.rs`
- `invoke_transformations/` holds only what Python's `AmazonAnthropicClaudeMessagesConfig` holds, under its method names. Helpers Python keeps in `bedrock/common_utils.py` and `bedrock/request_metadata.py` live in the matching Rust modules
- Invocation metrics in the response map onto Messages usage here

# references

- https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
