# rules

- Claude on Bedrock InvokeModel: reuses `anthropic/messages` shaping, moves the model into the URL, signs with SigV4 or a bearer token, and decodes the AWS event stream
- Invocation metrics in the response map onto Messages usage here
- Keep Bedrock differences here, not in the Anthropic helpers

# references

- https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
