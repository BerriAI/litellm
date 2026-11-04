# QCA managed interaction transport

## Status

Accepted.

## Context

QCA Cloud Agents exposes a session and event API rather than a single model inference request. A client creates a
session, appends a user message, then follows the session event stream until the agent returns to an idle or terminal
state. The Interactions API expects one logical request with a stable interaction ID, normalized steps, and an optional
stream.

The generic Interactions provider transformation supports one upstream HTTP request. Making it orchestrate several QCA
requests would mix transport state with request and response transformation. Routing QCA through the Responses API
bridge would also lose session continuation and event resume semantics.

## Decision

QCA uses a dedicated Interactions handler. It owns session creation, user event submission, SSE resume with
`Last-Event-ID`, terminal state detection, and normalization into LiteLLM Interactions responses.

An interaction ID contains the QCA session ID and accepted input event ID. Continuation reuses the session and submits a
new user event. The input event ID is also the SSE resume cursor so events emitted between event acceptance and stream
connection are not missed.

QCA credit fields remain provider usage extras under the normalized response usage object. LiteLLM does not convert
credits into a currency cost because QCA does not publish a stable conversion owned by LiteLLM.

The first implementation accepts async text interactions only. Unsupported tools, files, background execution, and
lifecycle operations fail before any upstream request.

## Alternatives considered

Extending the generic transformation was rejected because its one-request contract cannot represent session creation,
event submission, and streaming without provider-specific branches throughout the shared handler.

Mapping QCA to chat completions or Responses was rejected because neither surface preserves the accepted event cursor or
the managed session lifecycle.

## Consequences

The provider has a small dedicated transport module while the public Interactions entry point keeps one provider
dispatch branch. Router deployments can expose QCA through an ordinary model alias with server-owned credentials and an
environment ID.

The interaction ID is an opaque continuation token for clients. QCA lifecycle endpoints remain unsupported until their
semantics can be mapped without pretending a session delete or cancel is an interaction operation.

## References

QCA documents the session creation contract at <https://docs.qoder.com/cloud-agents/api/sessions/create> and event
submission at <https://docs.qoder.com/cloud-agents/api/sessions/events/send>.

The event stream and resume cursor are documented at <https://docs.qoder.com/cloud-agents/api/sessions/events/stream>
and <https://docs.qoder.com/cloud-agents/sse-initial-connection-behavior-change>.
