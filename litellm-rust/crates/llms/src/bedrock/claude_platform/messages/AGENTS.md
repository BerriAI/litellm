# rules

- Reuse Anthropic Messages shaping, reasoning translation, and beta headers
- Require a workspace header and use an API key or AWS SigV4
- Keep connection fields out of the upstream request body
- Drop `context_management` by default, with per-request `claude_platform_unsupported_params` overrides

# references

- Python counterpart `litellm/llms/bedrock/claude_platform/messages_transformation.py`
- https://platform.claude.com/docs/en/build-with-claude/claude-platform-on-aws
