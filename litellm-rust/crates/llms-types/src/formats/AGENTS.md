# API format ownership

This directory owns shared API request and response bodies, messages, content blocks, usage records, tool-call chunks, stream-event payloads, and protocol error bodies, grouped by API format

A format is independent of the provider that originated it. Name contracts for the format, without a provider prefix solely because that provider designed it. Format fields stay here even when provider support varies. Representing a field does not promise provider support

Format types must not depend on `providers`. Provider wire envelopes and extensions belong there and may reuse format contracts. Adapter-only decoding or rendering projections stay in `llms` until there is a shared data contract to expose

Keep distinct API formats distinct. Similar fields do not justify merging Chat Completions messages and choices with Responses input/output items. Reuse existing format contracts inside batch and token-counting payloads instead of copying them

These contracts include LiteLLM's normalized responses and supported extensions. An upstream schema must not silently replace the contract handed to the host or impose one provider's restrictions on every implementation

Shape validation belongs here. Model support, thinking budgets, beta requirements, content rewriting, finish-reason mapping, usage normalization, and cross-format conversions belong in provider adapters or shared transformation machinery. Messages web-search and encrypted-content fields remain format data regardless of which providers implement them

Stream events are data. Decoding, framing, buffering, live streams, and lifecycle decisions belong in execution crates. A transformation result does not become a wire event because it contains one

Follow the parent crate's contract-preservation rules. Keep serialization, rejection, passthrough, and presence-semantics tests here. Test transformations where they are implemented
