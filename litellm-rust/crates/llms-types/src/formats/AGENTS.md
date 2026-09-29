# API format contracts

This directory owns shared request and response bodies, messages, content blocks, usage records, tool-call chunks, stream-event payloads, and protocol error bodies. Organize them by API format, such as `messages`, `chat_completions`, `responses`, `ocr`, `audio_transcription`, and `batches`

## Format ownership

A format is independent of the provider that originated it. Use names such as `MessagesRequest`, without an Anthropic prefix solely because Anthropic designed Messages. A field belonging to the format stays here even when provider support varies. Representing it does not promise that every provider supports it

Keep format types independent of `providers`. Provider-specific wire envelopes and extensions belong there and may reuse these contracts. Adapter-only decoding or rendering projections stay in `llms` until there is a shared API data contract to expose

Keep Chat Completions messages and choices distinct from Responses input/output items. A Responses message is one item kind alongside reasoning, function calls, and function-call outputs. Similar fields do not justify one message or content-block model across different formats. Reuse an existing format contract inside batch and token-counting payloads instead of copying it

## Serialization contract

These types include LiteLLM's normalized response contracts and extensions, not just upstream schemas. `ChatCompletionsResponse` represents the response handed to the host. Replacing it with a more complete upstream schema must not silently change that contract

Represent known discriminators such as `tool_use` and `server_tool_use` with enum variants. Add typed item and content variants as consumers need them, preserving existing opaque passthrough. Follow the crate's rules for unknown data, malformed input, missing fields, explicit nulls, and numeric coercion. Do not turn a partial projection into a stricter public schema as part of a type move

Messages web-search result/error schemas, encrypted-content fields, and thinking configuration are format data. Flattening search results, removing encrypted content, choosing thinking budgets, requiring beta headers, and checking model support are provider policy and belong in `llms`

Stream-event payloads belong here. Live streams, decoders, framing, buffering, and lifecycle decisions belong in execution crates. For example, `ResponsesWsEvent` is event data, while `ResponsesWsTransformResult` belongs in `llms::base_llm::responses::transformation`

## Verification and references

Test the serialized contract consumers observe, including extensions and presence semantics. Add rejection cases for shapes the contract rejects and preservation cases for data it passes through. Transformation and cross-format conversion tests belong with their implementation

Chat Completions references: [OpenAI overview](https://developers.openai.com/api/reference/chat-completions/overview.md) and [MiniMax compatibility](https://platform.minimax.io/docs/api-reference/text-chat-openai.md)

Responses references: [OpenAI migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses.md) and [MiniMax create response](https://platform.minimax.io/docs/api-reference/responses-create.md)

References describe format shapes and individual implementations. They do not require modeling every field or imposing one provider's restrictions on shared types. Check the current provider reference when changing its adapter
