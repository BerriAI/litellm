# rules

- Reuse Anthropic Messages shaping and reasoning translation
- Use bearer credentials or AWS SigV4 for the Mantle Messages endpoint
- Filter and rename caller betas using the Mantle beta policy

# references

- Python counterpart `litellm/llms/bedrock/messages/mantle_transformation.py`
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
