# structure

- Messages adapter for Claude Platform on AWS, the Anthropic-operated Messages API behind an AWS gateway, reached as `bedrock/claude_platform/<model>`

# invariants

- The `claude_platform/` prefix selects this adapter before any converse or invoke check, with no base-model or capability gating, and is stripped once before the model goes on the wire
- A workspace ID is required, and a missing one is an authentication error raised before any request is sent
- An api key means `x-api-key` auth with no signing. Without one the request is SigV4 signed with the Bedrock credential chain, and the signature must cover the final serialized body
- Workspace and override keys, every `aws_*` key, and gateway-rejected params never reach the body
- Automatic betas are computed on the filtered body, so a dropped param never pulls in a beta

# gotchas

- User `anthropic-beta` values are forwarded as is, unlike plain Bedrock, because this gateway serves the full Anthropic platform and Python turns beta filtering off here
- `claude_platform_unsupported_params` replaces the default rejected set (`context_management`) instead of extending it, and a non-list value is ignored with a warning
- Responses and streams are native Anthropic Messages with no Bedrock event-stream framing, and only error mapping reports them as Bedrock errors

- Anthropic payload, beta and response policy and Bedrock region validation and SigV4 signing are reused explicitly, so this folder only holds the workspace, URL, auth mode and body filtering differences
- The workspace ID, the `aws_*` params and the `claude_platform_unsupported_params` override reach the config as an injected `ClaudePlatformSettings`, because the Messages trait gives `validate_environment` no request or deployment params. `ClaudePlatformSettings::from_params(request_extra, deployment_params)` holds Python's key precedence, and the const config carries empty settings, so only the environment applies
- A forwarded `x-api-key` with no resolved key is sent unsigned, where Python would SigV4 sign it and keep the header
- SigV4 signing needs the same validated region as the default URL and fails without one, instead of falling back to a default region

# known gaps

- Core does not route `bedrock/claude_platform/<model>` to this config yet, and it still has no per-call way to build `ClaudePlatformSettings` from the request and deployment params
- A non-list override comes back as `UnsupportedParams::Ignored` for the caller to log, since this crate has no logger
- Error mapping does not yet report upstream failures as Bedrock errors
- The global `litellm.api_base` fallback is left to the caller's `api_base`

# references

## python

- `litellm/llms/bedrock/claude_platform/messages_transformation.py` (`BedrockClaudePlatformMessagesConfig`)
- `litellm/llms/bedrock/claude_platform/common_utils.py`
- `litellm/llms/bedrock/claude_platform/transformation.py`, the chat config, only as a cross-check of the same auth and body filtering

## docs

- https://platform.claude.com/docs/en/build-with-claude/claude-platform-on-aws
- https://platform.claude.com/docs/en/build-with-claude/claude-platform-on-aws.md
- https://platform.claude.com/llms.txt
- https://docs.aws.amazon.com/claude-platform/latest/userguide/making-requests.html
- https://docs.aws.amazon.com/claude-platform/latest/userguide/making-requests.md
- https://docs.aws.amazon.com/claude-platform/latest/userguide/llms.txt
- https://platform.claude.com/docs/en/api/http/messages/create
