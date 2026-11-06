# ADR 002 CustomLLM can handle native generateContent requests

Status Accepted

## Context

The generateContent completion adapter cannot preserve image-generation configuration or generated inline image parts. Custom provider names also cannot be converted to the built-in provider enum

## Decision

CustomLLM exposes separate synchronous and asynchronous non-streaming generateContent hooks. An overridden hook receives native contents, config and tools, with other request fields in optional_params and the usual logging and LiteLLM context. Dispatch happens before built-in provider config lookup. Calls without an override use the completion adapter, including arbitrary registered custom provider names

The hook returns the native response without completion conversion. Logging records a JSON representation through the existing generateContent response logging path so usageMetadata remains available to logging and cost normalization without changing the response sent to the caller

## Alternatives Considered

Registering a BaseGoogleGenAIGenerateContentConfig would reuse the HTTP transport but requires a different extension contract from existing CustomLLM handlers. Extending the completion adapter cannot represent every native field or image response without imposing another conversion contract

## Consequences

Sync and async hook overrides are independent. Streaming still uses the existing completion adapter for custom providers. Custom handlers own upstream transport and errors, while the SDK entrypoints retain exception mapping and success callbacks
