# structure

These examples cover the complete Messages stack. Some provider implementations and test ports land after the shared contracts; inspect the current branch before assuming support

Paths below are relative to the repository root

The base OCR and Mistral OCR pairs are the layout reference. Keep corresponding operation and parameter names when responsibilities match, using Rust acronym casing: `BaseOCRConfig` becomes `BaseOcrConfig`. Private Python helpers may drop the leading underscore. Give Rust helpers responsibility names rather than duplicating trait method names

For OCR, order operations as supported parameters, credential metadata and connection resolution, health-check input, parameter mapping, environment validation, URL construction, request transformation, async request transformation, response transformation, async response transformation and error conversion. Constants and data types precede the config; private helpers follow it in operation order; tests come last. Rust-only hooks follow corresponding Python methods

For other formats, order parameter support/mapping, environment validation, URL construction, request transformation and response transformation, then Rust-only runtime hooks. Trait defaults cover unchanged inherited behavior; explicit delegation shares provider policy without reproducing Python inheritance

`BaseOcrConfig::prepare_request` corresponds to Python HTTP-handler preparation, while `validate_request_body`, `base_llm/ocr/error.rs` and `document.rs` are Rust-only. Mistral uses the default async request transform; `resolve_headers`, `build_ocr_url` and `normalize_response` cover the typed environment, URL and response operations. Preserve existing credential/header precedence and response extras when aligning layout

Azure Messages pairs with `litellm/llms/azure_ai/anthropic/messages_transformation.py`; Bedrock Converse pairs with `litellm/llms/bedrock/chat/converse_transformation.py`. Chat's `supported_openai_param_mappings` describes name pairs, while Python returns a supported-name list; audio `map_transcription_params` remains a filtering helper

`AnthropicConfig`, `AmazonConverseConfig` and non-OCR base traits are partial ports. OpenAI Responses supports WebSocket and direct HTTP transformations, but HTTP does not implement Python's model-specific rewrites or Responses-to-Chat emulation. Preserve their admission and fallback gates

# examples

## Schema nodes and literal data

Bedrock changes schema `type: "custom"` to `"object"`. A generic recursive JSON walk corrupts tool contracts because objects in `const`, `enum`, `examples` and opaque extensions are data

```json
{
  "type": "custom",
  "properties": {
    "type": {
      "type": "custom",
      "const": {"type": "custom"},
      "examples": [{"type": "custom"}]
    }
  }
}
```

The schema's two `type` values become `"object"`; the property name `"type"` and literal values remain unchanged. In `normalized_schema`, recurse through typed `properties`, `$defs`, `items`, `additionalProperties` and combinators. Retain `..*schema` to preserve other fields and leave `Recognized::Unrecognized` values unchanged

```rust
Recognized::Known(JsonSchema::Object(Box::new(JsonSchemaObject {
    properties: schema.properties.map(map_properties),
    items: schema.items.map(map_box),
    any_of: schema.any_of.map(map_schemas),
    ..*schema
})))
```

This excerpt shows traversal boundaries; the existing helper also normalizes `schema_type` and the other typed schema fields. Do not move this provider normalization into `llms-types`

Regression: deserialize a real `ToolDefinition`, run `bedrock_tool_definition`, and compare the whole serialized result. Include a property named `type`, literal objects and an unrecognized schema field

## Typed requests and provider wire requests

Bedrock Invoke carries `anthropic_beta` in the body, removes the HTTP beta header, and omits body `model` and `stream` because the endpoint identifies both. Vertex also omits body `model`

The `inference-messages` handler serializes `MessagesRequest` with `messages_request_body` before request interceptors and forwards authenticated headers separately. Provider-specific wire payload changes belong in the provider request transformation

For Invoke streaming, retain the typed request's stream intent even though wire serialization removes `stream`. Test request preparation and streaming selection together; checking only the serialized body misses that dependency

## Validate Python boundary inputs before projection

The bridge projects Python settings and request kwargs into typed Rust inputs; it does not own provider execution. A wrong projection key or silent fallback cannot be caught by Rust-only transformation tests

`aws_region_name=3` must be rejected, rather than becoming `None` and selecting an ambient region. `""`, `"US-EAST-1"` and strings containing path or hostname punctuation must also fail the legacy region gate. Reuse `BaseAWSLLM._validate_aws_region_name` after strict boundary typing

```python
region: Final = _BEDROCK_REGION.validate_python(
    kwargs.get("aws_region_name"), strict=True
)
BaseAWSLLM._validate_aws_region_name(region)
```

This is a follow-up example, not an implemented Python change in the Rust-only stack. The unchanged bridge does not project these Bedrock kwargs. A future boundary change must keep the specific type-check suppression and reason required by repository rules and test `route_host.shaping`, not the shared validator in isolation

## Metadata and connection ownership

Bedrock request metadata needs both a host projection of configured fields/sources and a Rust resolver. Preserve identity precedence, reserved prefixes, caller-header replacement and signing coverage. The Rust-only stack implements the resolver and typed inputs; the unchanged Python bridge does not supply them. Passing native tests do not establish Python SDK integration

Invoke currently applies `aws_bedrock_project_id` as a workspace header even though the Python workspace contract is Mantle's. This is an additional behavior to report, not a demonstrated provider failure. Do not treat Mantle tests as proof of Invoke parity

Per-call AWS access keys, roles, profiles and sessions are still not projected into Messages signing inputs. This predates the implemented Invoke transform; report it as a known gap rather than a new regression or silently expanding a test-port task to fix it

The unchanged Python bridge also omits the newer mid-conversation-system, cache-TTL, structured-output, tool-search and effort-ceiling capabilities. They retain Rust defaults until the host projects them. Compatible Rust adapters intentionally preserve native thinking and sampling fields; Python configs that still inherit Claude shaping may behave differently. Document these differences instead of editing the Python reference

## System turns and credential precedence

Azure and Vertex follow Python's shared system normalization: hoist only leading system turns, preserve later turns when supported, otherwise convert them to user turns with the operator note. Move a converted reminder behind an immediately following tool-result turn. String top-level `system` normalizes to blocks

Updating older Rust expectations to this behavior is a parity correction, not inherently a weakened test. Preserve full content and turn ordering in assertions

Vertex always replaces forwarded `Authorization` with the explicit/configured token or minted Google credential, matching Python's stale-token refresh test. A caller that previously used only a forwarded valid bearer loses that path when no configured credential is available. Report this compatibility consequence explicitly; token-looking fixture names do not prove token validity

## Stream failures and upstream status

AWS exception frames are not base64 chunk payloads. Decode error headers first, preserve their status/message in `ErrorDetail::Http`, then let the inference boundary map them to transport HTTP errors. Generic invalid-response errors keep their existing mapping

For a combined regression, feed an encoded throttling frame through the Messages decoder and route boundary and assert upstream status 429. Place that test downstream in inference; never import inference back into `llms`

A preserved upstream error status does not change the status of an HTTP response whose streaming headers were already sent. The Chat helper shares the decoder, but helper linkage alone does not establish a live Chat route selecting it

# references

- `litellm/llms/bedrock/common_utils.py`
- `litellm/llms/bedrock/base_aws_llm.py`
- `litellm/llms/anthropic/pass_through/messages/transformation.py`
- `litellm/rust_bridge/messages/route_host.py`
- `litellm-rust/crates/llms/src/bedrock/messages/invoke_transformations/anthropic_claude3_transformation.rs`
- `litellm-rust/crates/inference/src/error.rs`
