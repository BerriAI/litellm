# structure

Paths below are relative to the repository root. These are ownership rules, not a claim that every legacy item already follows them

| Contract or responsibility | Owner |
| --- | --- |
| Requests, responses, content blocks, usage, tool-call chunks, stream events, protocol error bodies | `llms-types/formats/<format>` |
| Shared provider wire values and extensions, such as `AnthropicBeta` and `BetaSet` | `llms-types/providers` |
| Adapter traits, transform contexts, capabilities, thinking budgets, stream transformer state | `llms/base_llm/<format>` |
| Capability policy, defaults, effort clamping, header/auth selection, provider rewriting | `llms/<provider>/<format>` |
| Calls, shaping envelopes, prepared requests, timeout/routing inputs, live response streams | `inference-<format>` |
| Pricing and catalog records | `model-catalog` |
| SSE/AWS framing, provider decoding, orchestration | `framer`, `llms`, `inference-<format>`, respectively |
| Python hooks and objects, credential/client mechanics, legacy logging selection | Their host, auth, transport and callback owners |

`formats/chat` means Chat Completions, matching `llms/base_llm/chat`, provider `chat` adapters and `inference-chat`. A format's origin does not change its ownership: Messages web-search schemas and encrypted-content fields remain format data

Provider extensions may reuse format contracts, but formats must not depend on providers. Add a typed extension when a consumer interprets or constructs it. Adapter-only projections remain local until they are genuinely shared public API data

# examples

## Data helpers versus execution policy

`ContentBlock::text`, `ResponsesWsEvent::model`, `Recognized::known` and exact `EffortLevel` conversions are deterministic data operations. Clamping effort, choosing a thinking budget, mapping finish reasons, computing normalized usage and translating API formats are transformations even when pure

`ProviderSpecificHeader` and `ProviderSpecificHeaders` are shared LiteLLM input data. Parsing and deduplicating `BetaSet` belong with its value type; provider-scoped header selection belongs in `core-utils`, required betas and precedence in `llms`, and applying headers in `http`

Stream events describe wire data; `ResponsesWsTransformResult` describes provider transformation output and belongs in `llms::base_llm::responses::transformation`. A live stream, decoder or buffering decision is never a shared payload schema

## Partial contracts and adapter projections

`ChatCompletionsResponse` is the normalized response handed to the host, not necessarily a complete upstream response. `BatchResponse` is a partial normalized LiteLLM batch job consumed by the Anthropic adapter

Batch submission, polling, result retrieval and count/status normalization remain in the provider's `batches` adapter. Reuse the enclosed request/result format types without inventing duplicate schemas

`InvokeChunkPayload` is a decoding envelope and `ReplayedWebSearchResult` is a permissive rendering projection. Do not expose them as complete public contracts. Reuse shared messages in batch and token-count adapters without promoting every helper struct

The prompt factory's `Conversation` is a normalization intermediate, so it stays with its algorithm in `core-utils`

## Missing, null and unknown values

For a deliberately permissive field, the existing Serde helpers retain omitted, explicit-null, known and unknown values separately

```rust
#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct Example {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub mode: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
```

This excerpt uses this crate's local macro and helpers. It is a pattern only for a field whose existing acceptance is permissive; `Recognized<String>` also retains a number or object and therefore cannot replace a strict field indiscriminately

| Input | Representation | Required serialized result |
| --- | --- | --- |
| `{}` | `mode: None` | Field stays omitted |
| `{"mode": null}` | `Some(Unrecognized(Null))` | Explicit null survives |
| `{"mode": "future"}` | `Some(Known("future"))` | String survives |
| `{"mode": {"opaque": 1}}` | `Some(Unrecognized(...))` | Opaque object survives |
| `{"new_field": [1, null]}` | Stored in `extra` | Unknown field survives |

For `Option<Nullable<T>>`, omission is `None`, explicit null is `Some(Nullable::Null)`, and a present typed value is `Some(Nullable::Value(...))`. Plain `Option<T>` commonly collapses missing and null; do not substitute it without checking the contract

Typing an opaque field must neither reject previously accepted inputs nor accept malformed inputs previously rejected. Known discriminators get enum variants while the existing unknown-variant behavior stays intact

## Schema shape versus schema interpretation

```json
{
  "type": "object",
  "properties": {
    "payload": {"const": {"type": "custom"}}
  }
}
```

The nested `const` is a literal object requiring the value `{"type":"custom"}`. It is not a child schema. `JsonSchemaObject.extra` preserves such values; `properties`, `items` and combinators identify typed schema positions

This crate represents both data and schema shape. A Bedrock adapter may normalize a schema type to `"object"`, but it must preserve the literal object. Keep that regression in the provider crate and wire round-trip tests here

## One authoritative field

When promoting a field from `extra` to a typed member, deserialize and serialize through the typed field alone. Do not leave the old map entry alongside it or add a second public alias to ease consumer migration

Move canonical definitions and update imports together. A public or serializable execution type such as `MessagesTransformContext`, `MessagesCall` or `StreamShape` remains with its execution owner

# validation

Use named cases to verify missing/null distinctions, unknown fields and discriminators, accepted malformed passthrough where intentional, and strict rejection elsewhere. Compare round-trip values and exact omission rather than only successful deserialization

Example test body for the permissive field above:

```rust
#[rstest]
#[case::missing(json!({}))]
#[case::null(json!({"mode": null}))]
#[case::opaque(json!({"mode": {"future": [1, null]}}))]
#[case::extension(json!({"new_field": [1, null]}))]
fn round_trip(#[case] input: Value) {
    let parsed: Example = serde_json::from_value(input.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), input);
}
```

Use the actual owning type in production tests. Schema generation and data-shape checks belong here; provider support, credential precedence and stream execution tests belong in their owning crates

# references

- `litellm-rust/crates/llms-types/src/lib.rs` (`wire_type`)
- `litellm-rust/crates/llms-types/src/serde_compat.rs`
- `litellm-rust/crates/llms-types/src/recognized.rs`
- `litellm-rust/crates/llms-types/src/json_schema.rs`
- `litellm-rust/crates/inference/AGENTS.md`
