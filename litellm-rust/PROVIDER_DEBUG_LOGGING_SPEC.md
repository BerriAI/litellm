# Rust provider debug logging

## Objective

When `litellm._turn_on_debug()` is enabled, a Python caller using the native Rust bridge should see the exact request Rust sends to the provider and the provider response that Rust receives

The logging contract is structured JSON first. Human-friendly terminal output is a renderer for that same JSON contract, not a separate logging format

## Output modes

The logger supports two renderers over the same event type

`json` emits one compact JSON object per line. This is the canonical machine-readable format for files, pipes, CI, and log collectors

`pretty` emits indented JSON with terminal colors. This is the default for an interactive terminal after `litellm._turn_on_debug()` is called

Use [`colored_json`](https://docs.rs/colored_json/latest/colored_json/) for the pretty renderer. It operates directly on `serde` values, supports automatic terminal detection, and honors color configuration without introducing an interactive terminal framework

Do not use a prompt or progress-bar library such as `cliclack` or `indicatif`. Debug events are logs, not an interactive UI. Do not use an immature tree formatter when the data already has a useful JSON structure

`JSON_LOGS=true`, redirected output, or a non-interactive terminal selects compact JSON. `NO_COLOR` disables ANSI color while preserving indentation in pretty mode

## Request event

The event is emitted after provider transformation and authentication, immediately before the HTTP request is sent

```json
{
  "event": "provider.request",
  "source": "litellm-rust",
  "call_id": "call_01",
  "provider": "bedrock",
  "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
  "stream": false,
  "method": "POST",
  "url": "https://bedrock-runtime.us-west-2.amazonaws.com/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/invoke",
  "headers": {
    "authorization": "[REDACTED]",
    "content-type": "application/json",
    "x-amz-date": "20260729T034800Z"
  },
  "body": {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 20,
    "messages": [
      {
        "role": "user",
        "content": "Say hello"
      }
    ]
  }
}
```

The event must represent the final provider request. Logging the original LiteLLM input would hide transformation bugs and would not answer what was sent over the network

## Non-streaming response event

The event is emitted after the complete provider response is read and before provider response transformation

```json
{
  "event": "provider.response",
  "source": "litellm-rust",
  "call_id": "call_01",
  "provider": "bedrock",
  "status": 200,
  "duration_ms": 842,
  "headers": {
    "content-type": "application/json",
    "x-amzn-requestid": "request_01"
  },
  "body": {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "Hello"
      }
    ]
  }
}
```

Keeping this event before response transformation makes provider protocol errors visible. A future `litellm.response` event may show the normalized result, but it is outside this feature

## Streaming events

Streaming must not buffer the response or emit one log entry per token. The first event records the accepted upstream response

```json
{
  "event": "provider.stream.started",
  "source": "litellm-rust",
  "call_id": "call_02",
  "provider": "bedrock",
  "status": 200,
  "content_type": "application/vnd.amazon.eventstream"
}
```

The terminal event summarizes what the decoder observed

```json
{
  "event": "provider.stream.completed",
  "source": "litellm-rust",
  "call_id": "call_02",
  "provider": "bedrock",
  "duration_ms": 1240,
  "bytes_received": 4832,
  "frames_received": 12,
  "events_decoded": 12
}
```

An interrupted or invalid stream emits `provider.error` with the counters collected before failure

## Error event

```json
{
  "event": "provider.error",
  "source": "litellm-rust",
  "call_id": "call_03",
  "provider": "bedrock",
  "duration_ms": 311,
  "status": 403,
  "kind": "http_error",
  "message": "provider returned HTTP 403",
  "body": {
    "message": "The security token included in the request is invalid"
  }
}
```

Errors remain values in Rust. Logging an error does not replace or alter the existing typed error returned to the caller

## Hook contract

Use one immutable tagged event union and one observation-only hook

```rust
#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "event")]
pub enum ProviderDebugEvent {
    #[serde(rename = "provider.request")]
    Request(ProviderRequestEvent),
    #[serde(rename = "provider.response")]
    Response(ProviderResponseEvent),
    #[serde(rename = "provider.stream.started")]
    StreamStarted(ProviderStreamStartedEvent),
    #[serde(rename = "provider.stream.completed")]
    StreamCompleted(ProviderStreamCompletedEvent),
    #[serde(rename = "provider.error")]
    Error(ProviderErrorEvent),
}

pub trait ProviderDebugHook: Send + Sync {
    fn emit(&self, event: &ProviderDebugEvent);
}
```

Each request receives `Option<Arc<dyn ProviderDebugHook>>`. `None` is the zero-cost disabled path. Avoid a global mutable logger flag in Rust because the native bridge can serve concurrent Python callers with different logging configuration

The hook cannot mutate requests, responses, or errors. Hook failures are swallowed after a bounded diagnostic because debug output must never change provider-call behavior

## Activation path

`litellm._turn_on_debug()` remains the user-facing switch

The Python bridge wrapper checks `litellm._logging._is_debugging_on()` for each call. When enabled, it passes a debug hook into the PyO3 `messages` or `amessages` entry point. PyO3 only adapts that hook into the Rust trait and does not contain provider logic

The standalone Axum server installs the same console hook when its debug log level is enabled. Both hosts therefore share event construction, redaction, and rendering

No debug state belongs in `litellm-core`. Core remains pure and returns transformed values. The `ai-gateway` I/O boundary is the only place that knows the final URL, signed headers, serialized body, timing, status, and stream counters

## Module structure

Keep the first implementation to two focused modules

```text
crates/ai-gateway/src/integrations/provider_debug/
  mod.rs       typed events, hook trait, safe event constructors, redaction
  console.rs   compact JSON and colored pretty JSON renderers
```

The Python bridge gets a thin adapter beside its existing route adapters

```text
crates/python-bridge/src/provider_debug.rs
```

Do not add provider-specific loggers. Bedrock, Anthropic, and Azure produce the same event union at the shared messages HTTP boundary

## Code style

Construct complete immutable events in one expression. Do not seed mutable maps and append fields over time

Use typed structs for every event payload. Do not pass untyped maps between the request handler, redactor, and renderer

Keep serialization in the renderer. Request execution emits typed values and does not call `println!`, `eprintln!`, or `serde_json::to_string_pretty`

Keep redaction in safe event constructors. Renderers must never receive an unredacted authorization value, which prevents a future renderer from accidentally leaking credentials

Use early returns for the disabled path and hook failures. Logging should not add nesting to request execution

Measure duration with `std::time::Instant`. Wall-clock timestamps may be added by the renderer, but elapsed request time must not depend on system-clock changes

Do not clone response bodies solely for logging. Non-streaming responses already require a bounded body read. Streaming paths record counters while forwarding existing chunks

Start with these modules and split further only if they exceed a clear responsibility. A separate file per event type would create file sprawl without improving safety or readability

## Redaction and bounds

The following request headers are always replaced with `[REDACTED]`, case-insensitively

```text
authorization
proxy-authorization
x-api-key
api-key
x-amz-security-token
cookie
set-cookie
```

Credential query parameters and body fields with known secret names are also redacted. AWS signatures, bearer tokens, and session credentials must never reach a hook

Prompt and response content remains visible because this mode explicitly exists to inspect provider traffic. The debug documentation must warn users that prompts may contain sensitive application data

Bodies are limited to 64 KiB of serialized JSON. A truncated event includes `body_truncated: true` and `body_original_bytes`. Binary data is represented by media type and byte count rather than copied into JSON

## Verification

Unit tests use an injected recording hook rather than capturing global stdout. They prove the request event contains the transformed Bedrock URL and body, every credential header is redacted, disabled logging emits nothing, response timing and status are present, and stream completion reports counters without buffering

Renderer tests compare compact JSON as parsed values rather than fragile string ordering. Pretty renderer tests disable color and compare indented output

The live Python test calls `litellm._turn_on_debug()`, invokes `litellm.anthropic.messages.acreate(..., rust=True)`, and verifies both the provider result and the Rust response marker. A manual run confirms the colored request and response events are readable in a terminal

## Non-goals

This feature does not implement production callback delivery, request mutation, distributed tracing export, per-token stream logging, persistent log storage, or normalized LiteLLM response logging
