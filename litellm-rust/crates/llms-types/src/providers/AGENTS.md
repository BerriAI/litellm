# Provider wire contracts

This directory owns shared provider-specific wire data and extensions: request and response envelopes, header values, invocation metrics, and protocol status values. Organize them by provider, adding format submodules when needed to keep distinct API contracts readable

## Placement

Use a provider-specific type when it represents that provider's protocol rather than a shared API format. `AnthropicBeta` describes the `anthropic-beta` header, and `BedrockInvocationMetrics` describes Bedrock invocation metadata. Messages content blocks belong in `formats::messages`, even if only one provider currently supports a variant

Reuse format contracts inside provider envelopes where their semantics match. Dependency direction is `providers` importing `formats`, never the reverse. Keep one authoritative type and import path. Do not copy a shared format model to add a provider field

A wire contract may live here independently of how many callers currently use it. A decoding or rendering projection tailored to one transformation stays in `llms`, such as `InvokeChunkPayload` or `ReplayedWebSearchResult`. Do not promote every adapter struct unchanged into a public schema

## Data and policy

Parsing, formatting, deduplication, shape validation, and value equality belong with the represented data. Wire-equivalent header values must agree on equality, ordering, and hashing, including a known variant and an opaque value with the same wire string

Provider capability checks, runtime defaults, authentication, OAuth companions, required betas, header precedence, parameter mapping, response normalization, polling, and retry decisions belong in `llms` or the auth crates. An operation-status enum belongs here. Deciding whether to poll again does not

Recognizing a wire value does not establish model support, a runtime default, or equivalent behavior across providers

Preserve existing extension fields, unknown values, missing fields, explicit nulls, and numeric coercion when adding typed fields. Use `Recognized<T>` only for established permissive contracts. Do not duplicate typed fields in flattened maps or silently discard unknown data

## Verification

Test wire serialization, parsing, malformed-input handling, extension preservation, presence semantics, and value semantics here. Test authentication, header selection, response normalization, and provider execution in their owning crates. Fixtures verify LiteLLM's supported contract rather than asserting that an external provider's schema never changes
