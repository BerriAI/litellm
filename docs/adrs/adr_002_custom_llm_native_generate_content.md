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

Sync and async hook overrides are independent. Streaming still uses the existing completion adapter for custom providers, and a handler without the matching streaming hook answers 400 before any upstream call. The completion adapter answers 400 for generationConfig keys it cannot translate unless drop_params is set. Custom handlers own upstream transport and errors, while the SDK entrypoints retain exception mapping and success callbacks

Adapter calls share the caller's logging object, so one request logs one success or failure under the caller's call id and call type, and the proxy's deferred logging covers them. Per-request prices apply on both paths
