# rules

- Claude on Bedrock InvokeModel: reuses `anthropic/messages` shaping, moves the model into the URL, and decodes the AWS event stream
- Invocation metrics in the response map onto Messages usage here
- The wire body is cut to Python's `BedrockInvokeAnthropicMessagesRequest` allowlist, and betas travel in the body's `anthropic_beta` (caller header values join through `request_with_headers`)

# references

- https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
