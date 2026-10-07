# structure

- The `bedrock/mantle/<model>` route: the wire adapter for the bedrock-mantle endpoint's native Anthropic Messages API
- The `bedrock_mantle` provider reuses this adapter for its Claude models by composition from `bedrock_mantle/messages`. Everything that differs between the two spellings is settings and auth, never wire format

# boundaries

- Credential and settings lookup for the `bedrock_mantle` spelling lives in `bedrock_mantle/messages`. This folder only reads the `bedrock` provider's `aws_*` settings
- Never import `bedrock_mantle` from here; the dependency only points from `bedrock_mantle` to `bedrock`

# invariants

- The model goes in the body with one leading `mantle/` segment removed, unlike Invoke, which puts it in the URL
- `anthropic_version` and `anthropic_beta` never appear in the body; send them as headers, and drop `anthropic-beta` when no beta survives
- On the SigV4 path, any caller copy of a header the signer computes (`authorization`, `x-amz-date`, `x-amz-security-token`, `date`) is removed before signing, because the signer refuses a request that already carries one. A bearer token leaves forwarded headers alone
- Streams are Anthropic SSE, not the AWS eventstream that Invoke returns
- Beta filtering happens once, in this adapter, against the mantle column of the Anthropic beta map

# gotchas

- `BEDROCK_MANTLE_API_BASE` is shared with the OpenAI-compatible mantle routes, so an override can carry `/v1` or `/openai/v1`. Strip any known base suffix before appending the messages path exactly once
- When no API key resolves, SigV4 signs for service `bedrock` and the credential scope must use the region in the URL host, or a stale `aws_region_name` breaks the signature

- One adapter for both spellings. Python's two paths drifted (accepted `api_base` suffixes, and `clear_thinking_20251015` accepted only on `bedrock_mantle`). Rust implements the union for both instead of copying the drift
- Settings are data, not code: `MantleSettings` names the bearer token envs, the `api_base` env, the region envs, the default region and the secret names. `bedrock_mantle` builds its own `AmazonMantleMessagesConfig { settings }` and delegates to it
- A `clear_thinking_20251015` edit with no thinking requested gets the minimum legacy budget before the Anthropic thinking translation runs, so adaptive models end up with adaptive thinking the same way caller-sent legacy thinking does

# known gaps

- `validate_environment` never sees the caller's `api_base`, so SigV4 takes its region from a public mantle host only when the host comes from the env override. A caller `api_base` pointing at another region's public host signs with the configured region until the trait passes `api_base` through
- No litellm params reach the trait, so `aws_region_name`, `aws_bedrock_runtime_endpoint`, `aws_bedrock_project_id` (sent as `anthropic-workspace-id`) and the per-call `aws_*` credentials are not read yet
- The Bedrock Invoke shaping Python inherits is not ported, because the Rust Invoke request transform is not either: cache-control `scope`/`ttl` stripping, structured output routing and `output_config` key stripping, the effort ceiling clamp, tool `custom`/schema/name normalization, `eager_input_streaming` removal, the top-level field allowlist, the web search tool rejection and the `tool-examples`, fine-grained streaming, `dangerous-tool-use` and interleaved thinking betas Python adds itself
- The bedrock request metadata headers Python merges in `validate_anthropic_messages_environment` are not sent

# references

## python

- `litellm/llms/bedrock/messages/mantle_transformation.py`
- `build_mantle_messages_url` in `litellm/llms/bedrock/common_utils.py`

## docs

- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.md
- https://docs.aws.amazon.com/bedrock/latest/userguide/apis.md
- https://docs.aws.amazon.com/bedrock/latest/userguide/llms.txt
