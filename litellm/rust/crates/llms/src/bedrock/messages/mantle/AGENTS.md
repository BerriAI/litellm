# rules

- The `bedrock/mantle/` route: the Invoke request transform with Mantle's `InvokeRequestSettings`, the betas moved into a header, `model` and `stream` back in the body, the Mantle URL and zero-filled usage
- Holds the Mantle shaping the `bedrock_mantle` provider reuses. Rules for both Mantle configs are in `bedrock_mantle/messages/AGENTS.md`

# references

- Python counterpart `litellm/llms/bedrock/messages/mantle_transformation.py` (`AmazonMantleMessagesConfig`)
- Upstream docs in `bedrock_mantle/messages/AGENTS.md`
