# rules

- Claude on Bedrock InvokeModel: reuses `anthropic/messages` shaping, moves the model into the URL, and decodes the AWS event stream through `bedrock/chat/invoke_handler.rs`
- `transformation.rs` is the config, `request.rs` the request transform and `InvokeRequestSettings`, `betas.rs` the Bedrock beta selection and `stream.rs` the event stream to Anthropic SSE relay
- InvokeModel reads betas from the body, so the selected betas go into the body's `anthropic_beta` list, not only the header
- Invocation metrics in the response map onto Messages usage here

# references

- Python counterpart `litellm/llms/bedrock/messages/invoke_transformations/anthropic_claude_transformation.py` (`AmazonAnthropicClaudeMessagesConfig`)
- https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
