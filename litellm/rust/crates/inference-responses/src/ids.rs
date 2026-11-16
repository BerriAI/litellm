//! LiteLLM-managed identifiers on Responses payloads, fully compatible with `litellm/responses/utils.py`
//!
//! LiteLLM rewrites some ids in a Responses API result before handing it back, so that a later
//! request carrying those ids can be routed to the provider and deployment that produced them,
//! and then restored to what the provider originally issued. The native and Python paths serve
//! the same traffic, so an id minted by either one must decode on the other.
//!
//! Each managed id is a fixed prefix followed by standard, padded base64 of a `;`-separated
//! plaintext:
//!
//! | Id | Applied to | Decoded plaintext |
//! |---|---|---|
//! | `resp_<b64>` | the response `id` | `litellm:custom_llm_provider:{provider};model_id:{model_id};response_id:{id}` |
//! | `encitem_<b64>` | the `id` of a reasoning or compaction output item, or of any item with `encrypted_content` | `litellm:model_id:{model_id};item_id:{id}` |
//! | `litellm_enc:<b64>;{content}` | that item's `encrypted_content` | `<b64>` is `model_id:{model_id}`, and the original content follows the `;` unencoded |
//! | `cntr_<b64>` | `container_id` on output items, `code_interpreter_call`, and message annotations | `litellm:custom_llm_provider:{provider};model_id:{model_id};container_id:{id}` |
//!
//! For example, OpenAI's `resp_0a1b2c3d4e5f` served without a deployment id becomes
//! `resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOnJlc3BfMGExYjJjM2Q0ZTVm`,
//! which decodes to `litellm:custom_llm_provider:openai;model_id:None;response_id:resp_0a1b2c3d4e5f`.
//!
//! How they fit together:
//! - `resp_` lets a follow-up's `previous_response_id` find the provider and deployment again.
//!   [`encode_response`] stamps it on every response, unless the id is already managed.
//! - `encitem_` and `litellm_enc:` carry encrypted-content affinity: encrypted reasoning only
//!   decrypts under the API key that produced it, so the router pins replays to that deployment.
//!   They are stamped only when the router sets `encrypted_content_affinity_enabled` and a
//!   deployment `model_id` is known (see [`IdScope`]). Reasoning and compaction items get an
//!   `encitem_` id by type, so it matches the one Python streams before their content arrives.
//!   Items without an `id`, such as Codex's, carry the deployment in the wrapped content alone.
//! - `cntr_` lets container calls find the provider and deployment that own a container. A
//!   container id that already names a provider is left as it is.
//! - On the way in, [`restore_input`] turns `encitem_` ids and `litellm_enc:` content in the
//!   request input back into the provider's own bytes before the request is sent. The router
//!   pins by the deployments [`previous_response_deployment`] and [`encrypted_content_origin`]
//!   read.
//!
//! Python's quirks are kept on purpose:
//! - A missing provider or `model_id` is written as `None` in `resp_` ids (an f-string of
//!   `None`), but as an empty string in `cntr_` ids. [`decode_container_id`] reads both as absent.
//! - Decoding follows Python's non-strict `base64.b64decode`: characters outside the alphabet
//!   are skipped, trailing bits are allowed, and padding stripped in transit is added back.
//! - A `resp_` id counts as managed when it decodes to at least three `;`-separated parts.

use base64::{
    Engine, alphabet,
    engine::{GeneralPurpose, general_purpose::PAD},
};
use litellm_llms_types::formats::responses::ResponsesApiResponse;
use serde_json::{Map, Value};

const RESPONSE_PREFIX: &str = "resp_";
const ITEM_PREFIX: &str = "encitem_";
const CONTAINER_PREFIX: &str = "cntr_";
const WRAPPED_CONTENT_PREFIX: &str = "litellm_enc:";
const ENCRYPTED_ITEM_TYPES: [&str; 2] = ["reasoning", "compaction"];
const REASONING_FIELDS_ONLY_ITS_ORIGIN_RESOLVES: [&str; 3] = ["encrypted_content", "id", "status"];
const OUTPUT_ITEM_EVENT_TYPES: [&str; 2] =
    ["response.output_item.added", "response.output_item.done"];
const REASONING_ITEM_EVENT_TYPES: [&str; 6] = [
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.reasoning_text.delta",
    "response.reasoning_text.done",
];
const CONTENT_PART_EVENT_TYPES: [&str; 2] =
    ["response.content_part.added", "response.content_part.done"];

/// Python's non-strict `base64.b64decode`: characters outside the alphabet are skipped and
/// non-zero trailing bits are accepted
const LENIENT: GeneralPurpose = GeneralPurpose::new(
    &alphabet::STANDARD,
    PAD.with_decode_allow_trailing_bits(true),
);

/// Where a response came from, as encoded into the ids LiteLLM hands back
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IdScope {
    pub provider: Option<String>,
    pub model_id: Option<String>,
    pub encrypted_content_affinity: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedContainerId {
    pub provider: Option<String>,
    pub model_id: Option<String>,
    pub container_id: String,
}

/// Rewrites the response id, encrypted-content items and container ids the way the Python
/// Responses path does before returning to the caller. An already managed id is left untouched
pub fn encode_response(response: &mut ResponsesApiResponse, scope: &IdScope) {
    if is_managed_response_id(&response.id) {
        return;
    }
    response.id = response_id(scope, &response.id);
    encode_output(&mut response.output, scope);
}

/// [`encode_response`] on a response a stream event carries as JSON
fn encode_response_object(response: &mut Map<String, Value>, scope: &IdScope) {
    let Some(id) = response.get("id").and_then(Value::as_str) else {
        return;
    };
    if is_managed_response_id(id) {
        return;
    }
    let encoded = response_id(scope, id);
    response.insert("id".to_owned(), Value::String(encoded));
    if let Some(Value::Array(output)) = response.get_mut("output") {
        encode_output(output, scope);
    }
}

fn affinity_model_id(scope: &IdScope) -> Option<&str> {
    scope
        .model_id
        .as_deref()
        .filter(|_| scope.encrypted_content_affinity)
        .filter(|model_id| !model_id.is_empty())
}

fn encode_output(output: &mut [Value], scope: &IdScope) {
    if let Some(model_id) = affinity_model_id(scope) {
        output
            .iter_mut()
            .filter_map(Value::as_object_mut)
            .for_each(|item| encode_encrypted_item(item, model_id));
    }
    output
        .iter_mut()
        .for_each(|item| encode_containers(item, scope));
}

/// One Responses stream event with the ids [`encode_response`] gives the final response
/// (`ResponsesAPIStreamingIterator._process_chunk`): the response it carries, its output item and
/// the reasoning item its deltas name, and the container ids on items and annotations
pub fn encode_stream_event(event: &mut Map<String, Value>, scope: &IdScope) {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if let Some(model_id) = affinity_model_id(scope) {
        encode_stream_affinity(event, &event_type, model_id);
    }
    if let Some(Value::Object(response)) = event.get_mut("response") {
        encode_response_object(response, scope);
    }
    match event_type.as_str() {
        "response.output_item.added" | "response.output_item.done" => {
            if let Some(item) = event.get_mut("item") {
                encode_containers(item, scope);
            }
        }
        "response.output_text.annotation.added" => {
            if let Some(annotation) = event.get_mut("annotation") {
                encode_containers(annotation, scope);
            }
        }
        "response.content_part.done" => {
            if let Some(Value::Array(annotations)) = event
                .get_mut("part")
                .and_then(|part| part.get_mut("annotations"))
            {
                annotations
                    .iter_mut()
                    .for_each(|annotation| encode_containers(annotation, scope));
            }
        }
        _ => {}
    }
}

/// `encode_stream_event_for_affinity`
fn encode_stream_affinity(event: &mut Map<String, Value>, event_type: &str, model_id: &str) {
    if OUTPUT_ITEM_EVENT_TYPES.contains(&event_type) {
        if let Some(Value::Object(item)) = event.get_mut("item") {
            encode_encrypted_item(item, model_id);
        }
        return;
    }
    let part_type = event
        .get("part")
        .and_then(|part| part.get("type"))
        .and_then(Value::as_str);
    let names_reasoning = REASONING_ITEM_EVENT_TYPES.contains(&event_type)
        || (CONTENT_PART_EVENT_TYPES.contains(&event_type) && part_type == Some("reasoning_text"));
    if !names_reasoning {
        return;
    }
    if let Some(Value::String(id)) = event.get_mut("item_id")
        && !id.is_empty()
        && decode_encrypted_item_id(id).is_none()
    {
        *id = encrypted_item_id(model_id, id);
    }
}

/// A `resp_` id taken apart (`decode_responses_api_response_id`); an id LiteLLM did not mint
/// names neither a provider nor a deployment
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedResponseId {
    pub provider: Option<String>,
    pub model_id: Option<String>,
    pub response_id: String,
}

pub fn decode_response_id(id: &str) -> DecodedResponseId {
    let unmanaged = || DecodedResponseId {
        provider: None,
        model_id: None,
        response_id: id.to_owned(),
    };
    let Some(decoded) = decode_text(&id.replace(RESPONSE_PREFIX, "")) else {
        return unmanaged();
    };
    let parts: Vec<&str> = decoded.split(';').collect();
    let [provider, model_id, response_id, ..] = parts.as_slice() else {
        return unmanaged();
    };
    DecodedResponseId {
        provider: Some(provider.replace("litellm:custom_llm_provider:", "")),
        model_id: Some(model_id.replace("model_id:", "")),
        response_id: response_id.replace("response_id:", ""),
    }
}

/// The deployment a `previous_response_id` names (`get_model_id_from_response_id`)
pub fn previous_response_deployment(previous_response_id: &str) -> Option<String> {
    decode_response_id(previous_response_id)
        .model_id
        .filter(|model_id| !model_id.is_empty())
}

/// The deployment the first encrypted-content marker in `input` names
/// (`_extract_model_id_from_input`)
pub fn encrypted_content_origin(input: &Value) -> Option<String> {
    input
        .as_array()?
        .iter()
        .filter_map(Value::as_object)
        .find_map(marked_origin)
        .filter(|model_id| !model_id.is_empty())
}

/// `_model_id_of_input_item`: an `encitem_` id names its deployment even when that is empty, a
/// wrapped `encrypted_content` only when it is not
fn marked_origin(item: &Map<String, Value>) -> Option<String> {
    if let Some(id) = item.get("id").and_then(Value::as_str)
        && let Some((model_id, _)) = decode_encrypted_item_id(id)
    {
        return Some(model_id);
    }
    item.get("encrypted_content")
        .and_then(Value::as_str)
        .and_then(wrapped_model_id)
        .filter(|model_id| !model_id.is_empty())
}

/// The deployment a wrapped `encrypted_content` names (`unwrap_encrypted_content_with_model_id`)
pub fn wrapped_model_id(content: &str) -> Option<String> {
    let (metadata, _) = content
        .strip_prefix(WRAPPED_CONTENT_PREFIX)?
        .split_once(';')?;
    Some(decode_text(&padded(metadata))?.replace("model_id:", ""))
}

/// `strip_reasoning_the_targets_cannot_decrypt` for the deployment a request is sent to: each
/// encrypted reasoning item whose origin (its own marker, else `unmarked_origin`) the target
/// cannot `decrypt` keeps only its readable summary, or is dropped when it has none
pub fn strip_undecryptable_reasoning(
    input: &mut Value,
    unmarked_origin: &str,
    decrypts: impl Fn(&str) -> bool,
) {
    let Some(items) = input.as_array_mut() else {
        return;
    };
    let kept = std::mem::take(items)
        .into_iter()
        .filter_map(|item| {
            let Value::Object(reasoning) = &item else {
                return Some(item);
            };
            if !is_encrypted_reasoning(reasoning) {
                return Some(item);
            }
            let origin = marked_origin(reasoning);
            if decrypts(origin.as_deref().unwrap_or(unmarked_origin)) {
                return Some(item);
            }
            readable_reasoning(reasoning)
        })
        .collect();
    *items = kept;
}

fn truthy(value: &Value) -> bool {
    litellm_python_compat::truthy::truthy(&litellm_python_compat::json::from_json(value.clone()))
}

/// `_encrypted_reasoning_item`
fn is_encrypted_reasoning(item: &Map<String, Value>) -> bool {
    item.get("type").and_then(Value::as_str) == Some("reasoning")
        && item.get("encrypted_content").is_some_and(truthy)
}

/// `readable_reasoning`: the item less what only its origin resolves, when its `summary` or
/// `content` carries text
fn readable_reasoning(reasoning: &Map<String, Value>) -> Option<Value> {
    let has_text = |value: Option<&Value>| match value {
        Some(Value::String(text)) => !text.trim().is_empty(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .any(|block| block.get("text").is_some_and(truthy)),
        _ => false,
    };
    if !has_text(reasoning.get("summary")) && !has_text(reasoning.get("content")) {
        return None;
    }
    Some(Value::Object(
        reasoning
            .iter()
            .filter(|(key, _)| !REASONING_FIELDS_ONLY_ITS_ORIGIN_RESOLVES.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    ))
}

/// Restores encrypted-content item ids and contents that LiteLLM encoded on an earlier response,
/// so the provider receives what it originally issued
pub fn restore_input(input: &mut Value) {
    let Some(items) = input.as_array_mut() else {
        return;
    };
    for item in items.iter_mut().filter_map(Value::as_object_mut) {
        if let Some(Value::String(id)) = item.get_mut("id")
            && let Some((_, original)) = decode_encrypted_item_id(id)
        {
            *id = original;
        }
        if let Some(Value::String(content)) = item.get_mut("encrypted_content")
            && let Some(original) = unwrap_encrypted_content(content)
        {
            *content = original;
        }
    }
}

pub fn response_id(scope: &IdScope, id: &str) -> String {
    let assembled = format!(
        "litellm:custom_llm_provider:{};model_id:{};response_id:{id}",
        python_str(scope.provider.as_deref()),
        python_str(scope.model_id.as_deref()),
    );
    format!("{RESPONSE_PREFIX}{}", LENIENT.encode(assembled))
}

pub fn is_managed_response_id(id: &str) -> bool {
    decode_text(&id.replace(RESPONSE_PREFIX, ""))
        .is_some_and(|decoded| decoded.split(';').count() >= 3)
}

pub fn encrypted_item_id(model_id: &str, id: &str) -> String {
    format!(
        "{ITEM_PREFIX}{}",
        LENIENT.encode(format!("litellm:model_id:{model_id};item_id:{id}"))
    )
}

pub fn decode_encrypted_item_id(id: &str) -> Option<(String, String)> {
    let decoded = decode_text(&padded(id.strip_prefix(ITEM_PREFIX)?))?;
    let (model_id, item_id) = decoded.split_once(';')?;
    Some((
        model_id.replace("litellm:model_id:", ""),
        item_id.replace("item_id:", ""),
    ))
}

pub fn wrap_encrypted_content(content: &str, model_id: &str) -> String {
    format!(
        "{WRAPPED_CONTENT_PREFIX}{};{content}",
        LENIENT.encode(format!("model_id:{model_id}"))
    )
}

/// The original content of a wrapped `encrypted_content`, or `None` when it is not wrapped
pub fn unwrap_encrypted_content(content: &str) -> Option<String> {
    if !content.starts_with(WRAPPED_CONTENT_PREFIX) {
        return None;
    }
    let (metadata, original) = content.split_once(';')?;
    decode_text(&padded(&metadata.replace(WRAPPED_CONTENT_PREFIX, "")))?;
    Some(original.to_owned())
}

pub fn container_id(scope: &IdScope, id: &str) -> String {
    let assembled = format!(
        "litellm:custom_llm_provider:{};model_id:{};container_id:{id}",
        scope.provider.as_deref().unwrap_or_default(),
        scope.model_id.as_deref().unwrap_or_default(),
    );
    format!("{CONTAINER_PREFIX}{}", LENIENT.encode(assembled))
}

pub fn decode_container_id(id: &str) -> DecodedContainerId {
    let unmanaged = || DecodedContainerId {
        provider: None,
        model_id: None,
        container_id: id.to_owned(),
    };
    if !id.starts_with(CONTAINER_PREFIX) {
        return unmanaged();
    }
    let Some(decoded) = decode_text(&id.replace(CONTAINER_PREFIX, "")) else {
        return unmanaged();
    };
    let parsed = decoded
        .strip_prefix("litellm:custom_llm_provider:")
        .and_then(|rest| rest.split_once(';'))
        .and_then(|(provider, rest)| {
            let (model_id, rest) = rest.strip_prefix("model_id:")?.split_once(';')?;
            let container_id = rest.strip_prefix("container_id:")?;
            (!container_id.is_empty() && !container_id.contains('\n')).then_some((
                provider,
                model_id,
                container_id,
            ))
        });
    let Some((provider, model_id, container_id)) = parsed else {
        return unmanaged();
    };
    let present = |value: &str| (!matches!(value, "" | "None")).then(|| value.to_owned());
    DecodedContainerId {
        provider: present(provider),
        model_id: present(model_id),
        container_id: container_id.to_owned(),
    }
}

fn encode_encrypted_item(item: &mut Map<String, Value>, model_id: &str) {
    let wrapped = match item.get_mut("encrypted_content") {
        Some(Value::String(content)) if !content.is_empty() => {
            *content = wrap_encrypted_content(content, model_id);
            true
        }
        _ => false,
    };
    let encrypted_type = item
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|item_type| ENCRYPTED_ITEM_TYPES.contains(&item_type));
    if !wrapped && !encrypted_type {
        return;
    }
    if let Some(Value::String(id)) = item.get_mut("id")
        && !id.is_empty()
        && decode_encrypted_item_id(id).is_none()
    {
        *id = encrypted_item_id(model_id, id);
    }
}

fn encode_containers(item: &mut Value, scope: &IdScope) {
    let Some(item) = item.as_object_mut() else {
        return;
    };
    encode_container_field(item, scope);
    if let Some(nested) = item
        .get_mut("code_interpreter_call")
        .and_then(Value::as_object_mut)
    {
        encode_container_field(nested, scope);
    }
    if item.get("type").and_then(Value::as_str) != Some("message") {
        return;
    }
    let Some(content) = item.get_mut("content").and_then(Value::as_array_mut) else {
        return;
    };
    content
        .iter_mut()
        .filter_map(|part| part.get_mut("annotations").and_then(Value::as_array_mut))
        .flatten()
        .for_each(|annotation| encode_containers(annotation, scope));
}

fn encode_container_field(item: &mut Map<String, Value>, scope: &IdScope) {
    if let Some(Value::String(id)) = item.get_mut("container_id")
        && decode_container_id(id).provider.is_none()
    {
        *id = container_id(scope, id);
    }
}

fn python_str(value: Option<&str>) -> &str {
    value.unwrap_or("None")
}

fn padded(value: &str) -> String {
    let missing = value.len() % 4;
    let padding = if missing == 0 { 0 } else { 4 - missing };
    format!("{value}{}", "=".repeat(padding))
}

fn decode_text(value: &str) -> Option<String> {
    let alphabet: String = value
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '/' | '=')
        })
        .collect();
    String::from_utf8(LENIENT.decode(alphabet).ok()?).ok()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn scope(provider: Option<&str>, model_id: Option<&str>, affinity: bool) -> IdScope {
        IdScope {
            provider: provider.map(str::to_owned),
            model_id: model_id.map(str::to_owned),
            encrypted_content_affinity: affinity,
        }
    }

    fn response(id: &str, output: Value) -> ResponsesApiResponse {
        serde_json::from_value(json!({"id": id, "model": "gpt-6.1-sol", "output": output})).unwrap()
    }

    #[rstest]
    #[case::openai_without_model_id(
        Some("openai"),
        None,
        "resp_0a1b2c3d4e5f",
        "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOnJlc3BfMGExYjJjM2Q0ZTVm"
    )]
    #[case::openrouter_with_model_id(
        Some("openrouter"),
        Some("deploy-1"),
        "gen-123",
        "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5yb3V0ZXI7bW9kZWxfaWQ6ZGVwbG95LTE7cmVzcG9uc2VfaWQ6Z2VuLTEyMw=="
    )]
    fn response_ids_match_python(
        #[case] provider: Option<&str>,
        #[case] model_id: Option<&str>,
        #[case] id: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(response_id(&scope(provider, model_id, false), id), expected);
    }

    #[rstest]
    #[case::raw_short_id("resp_0a1b2c3d4e5f", false)]
    #[case::raw_long_id("resp_68a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3", false)]
    #[case::two_segments("resp_YTti", false)]
    #[case::three_segments("resp_YTtiO2M=", true)]
    #[case::managed(
        "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOng=",
        true
    )]
    fn managed_response_ids_match_python(#[case] id: &str, #[case] expected: bool) {
        assert_eq!(is_managed_response_id(id), expected);
    }

    #[test]
    fn encrypted_items_match_python() {
        assert_eq!(
            encrypted_item_id("deploy-1", "rs_0a1b"),
            "encitem_bGl0ZWxsbTptb2RlbF9pZDpkZXBsb3ktMTtpdGVtX2lkOnJzXzBhMWI="
        );
        assert_eq!(
            wrap_encrypted_content("gAAAA", "deploy-1"),
            "litellm_enc:bW9kZWxfaWQ6ZGVwbG95LTE=;gAAAA"
        );
    }

    #[rstest]
    #[case::padded("encitem_bGl0ZWxsbTptb2RlbF9pZDpkZXBsb3ktMTtpdGVtX2lkOnJzXzBhMWI=", Some(("deploy-1", "rs_0a1b")))]
    #[case::padding_stripped_in_transit("encitem_bGl0ZWxsbTptb2RlbF9pZDptO2l0ZW1faWQ6cnNfMQ", Some(("m", "rs_1")))]
    #[case::not_base64("encitem_notbase64!", None)]
    #[case::unmanaged("rs_0a1b", None)]
    fn encrypted_item_ids_decode_like_python(
        #[case] id: &str,
        #[case] expected: Option<(&str, &str)>,
    ) {
        assert_eq!(
            decode_encrypted_item_id(id),
            expected.map(|(model_id, item_id)| (model_id.to_owned(), item_id.to_owned()))
        );
    }

    #[rstest]
    #[case::unpadded_metadata("litellm_enc:bW9kZWxfaWQ6bQ;gAAAA", Some("gAAAA"))]
    #[case::no_separator("litellm_enc:nosemicolon", None)]
    #[case::unwrapped("gAAAA", None)]
    fn encrypted_content_unwraps_like_python(
        #[case] content: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(unwrap_encrypted_content(content).as_deref(), expected);
    }

    #[rstest]
    #[case::with_model_id(
        Some("deploy-1"),
        "cntr_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpkZXBsb3ktMTtjb250YWluZXJfaWQ6Y250cl9hYmM="
    )]
    #[case::without_model_id(
        None,
        "cntr_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDo7Y29udGFpbmVyX2lkOmNudHJfYWJj"
    )]
    fn container_ids_match_python(#[case] model_id: Option<&str>, #[case] expected: &str) {
        assert_eq!(
            container_id(&scope(Some("openai"), model_id, false), "cntr_abc"),
            expected
        );
    }

    #[rstest]
    #[case::raw("cntr_abc", None, None, "cntr_abc")]
    #[case::managed(
        "cntr_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDptO2NvbnRhaW5lcl9pZDpjbnRyXzE=",
        Some("openai"),
        Some("m"),
        "cntr_1"
    )]
    #[case::empty_segments(
        "cntr_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOjttb2RlbF9pZDo7Y29udGFpbmVyX2lkOmNudHJfMQ==",
        None,
        None,
        "cntr_1"
    )]
    #[case::python_none_provider(
        "cntr_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOk5vbmU7bW9kZWxfaWQ6O2NvbnRhaW5lcl9pZDp4",
        None,
        None,
        "x"
    )]
    fn container_ids_decode_like_python(
        #[case] id: &str,
        #[case] provider: Option<&str>,
        #[case] model_id: Option<&str>,
        #[case] container_id: &str,
    ) {
        assert_eq!(
            decode_container_id(id),
            DecodedContainerId {
                provider: provider.map(str::to_owned),
                model_id: model_id.map(str::to_owned),
                container_id: container_id.to_owned(),
            }
        );
    }

    #[test]
    fn encodes_response_items_under_affinity() {
        let mut encoded = response(
            "resp_1",
            json!([
                {"type": "reasoning", "id": "rs_0a1b", "encrypted_content": "gAAAA", "summary": []},
                {"type": "reasoning", "encrypted_content": "gBBBB", "summary": []},
                {"type": "code_interpreter_call", "id": "ci_1", "container_id": "cntr_abc"},
                {"type": "message", "id": "msg_1", "content": [{"type": "output_text", "text": "", "annotations": [
                    {"type": "container_file_citation", "container_id": "cntr_abc"}
                ]}]}
            ]),
        );
        let scope = scope(Some("openai"), Some("deploy-1"), true);
        encode_response(&mut encoded, &scope);
        let container = container_id(&scope, "cntr_abc");
        assert_eq!(encoded.id, response_id(&scope, "resp_1"));
        assert_eq!(
            Value::Array(encoded.output),
            json!([
                {
                    "type": "reasoning",
                    "id": "encitem_bGl0ZWxsbTptb2RlbF9pZDpkZXBsb3ktMTtpdGVtX2lkOnJzXzBhMWI=",
                    "encrypted_content": "litellm_enc:bW9kZWxfaWQ6ZGVwbG95LTE=;gAAAA",
                    "summary": []
                },
                {"type": "reasoning", "encrypted_content": "litellm_enc:bW9kZWxfaWQ6ZGVwbG95LTE=;gBBBB", "summary": []},
                {"type": "code_interpreter_call", "id": "ci_1", "container_id": container},
                {"type": "message", "id": "msg_1", "content": [{"type": "output_text", "text": "", "annotations": [
                    {"type": "container_file_citation", "container_id": container}
                ]}]}
            ])
        );
    }

    #[rstest]
    #[case::reasoning_without_ciphertext(
        json!({"type": "reasoning", "id": "rs_0a1b", "summary": []}),
        json!({"type": "reasoning", "id": "encitem_bGl0ZWxsbTptb2RlbF9pZDpkZXBsb3ktMTtpdGVtX2lkOnJzXzBhMWI=", "summary": []})
    )]
    #[case::compaction(
        json!({"type": "compaction", "id": "cmp_0a1b", "encrypted_content": "gAAAA"}),
        json!({
            "type": "compaction",
            "id": "encitem_bGl0ZWxsbTptb2RlbF9pZDpkZXBsb3ktMTtpdGVtX2lkOmNtcF8wYTFi",
            "encrypted_content": "litellm_enc:bW9kZWxfaWQ6ZGVwbG95LTE=;gAAAA"
        })
    )]
    #[case::already_encoded(
        json!({"type": "reasoning", "id": "encitem_bGl0ZWxsbTptb2RlbF9pZDpkZXBsb3ktMTtpdGVtX2lkOnJzXzBhMWI=", "summary": []}),
        json!({"type": "reasoning", "id": "encitem_bGl0ZWxsbTptb2RlbF9pZDpkZXBsb3ktMTtpdGVtX2lkOnJzXzBhMWI=", "summary": []})
    )]
    #[case::message(
        json!({"type": "message", "id": "msg_1", "content": []}),
        json!({"type": "message", "id": "msg_1", "content": []})
    )]
    fn encodes_item_ids_by_type_under_affinity(#[case] item: Value, #[case] expected: Value) {
        let mut encoded = response("resp_1", json!([item]));
        encode_response(&mut encoded, &scope(Some("openai"), Some("deploy-1"), true));
        assert_eq!(Value::Array(encoded.output), json!([expected]));
    }

    #[rstest]
    #[case::affinity_disabled(scope(Some("openai"), Some("deploy-1"), false))]
    #[case::no_model_id(scope(Some("openai"), None, true))]
    fn leaves_encrypted_items_without_affinity(#[case] scope: IdScope) {
        let output = json!([{"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA", "summary": []}]);
        let mut encoded = response("resp_1", output.clone());
        encode_response(&mut encoded, &scope);
        assert_eq!(Value::Array(encoded.output), output);
    }

    #[test]
    fn leaves_managed_responses_untouched() {
        let scope = scope(Some("openai"), Some("deploy-1"), true);
        let id = response_id(&scope, "resp_1");
        let output = json!([{"type": "code_interpreter_call", "container_id": "cntr_abc"}]);
        let mut encoded = response(&id, output.clone());
        encode_response(&mut encoded, &scope);
        assert_eq!((encoded.id, Value::Array(encoded.output)), (id, output));
    }

    #[test]
    fn restores_encrypted_items_in_input() {
        let mut input = json!([
            {
                "type": "reasoning",
                "id": "encitem_bGl0ZWxsbTptb2RlbF9pZDpkZXBsb3ktMTtpdGVtX2lkOnJzXzBhMWI=",
                "encrypted_content": "litellm_enc:bW9kZWxfaWQ6ZGVwbG95LTE=;gAAAA",
                "summary": []
            },
            {"role": "user", "content": "hi"}
        ]);
        restore_input(&mut input);
        assert_eq!(
            input,
            json!([
                {"type": "reasoning", "id": "rs_0a1b", "encrypted_content": "gAAAA", "summary": []},
                {"role": "user", "content": "hi"}
            ])
        );
    }

    fn managed(model_id: &str) -> String {
        response_id(
            &scope(Some("azure"), Some(model_id), false),
            "resp_upstream",
        )
    }

    #[rstest]
    #[case::managed(managed("deploy-1"), Some("deploy-1"), "resp_upstream")]
    #[case::minted_without_a_deployment(
        "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOnJlc3BfMGExYjJjM2Q0ZTVm".to_owned(),
        Some("None"),
        "resp_0a1b2c3d4e5f"
    )]
    #[case::raw("resp_0a1b2c3d4e5f".to_owned(), None, "resp_0a1b2c3d4e5f")]
    #[case::two_segments("resp_YTti".to_owned(), None, "resp_YTti")]
    fn response_ids_decode_like_python(
        #[case] id: String,
        #[case] deployment: Option<&str>,
        #[case] original: &str,
    ) {
        assert_eq!(
            (
                previous_response_deployment(&id),
                decode_response_id(&id).response_id
            ),
            (deployment.map(str::to_owned), original.to_owned())
        );
    }

    #[rstest]
    #[case::encoded_item_id(
        json!([{"type": "message"}, {"type": "reasoning", "id": encrypted_item_id("deploy-1", "rs_1")}]),
        Some("deploy-1")
    )]
    #[case::wrapped_content(
        json!([{"type": "reasoning", "encrypted_content": wrap_encrypted_content("gAAAA", "deploy-2")}]),
        Some("deploy-2")
    )]
    #[case::first_marker_wins(
        json!([
            {"encrypted_content": wrap_encrypted_content("gAAAA", "deploy-2")},
            {"id": encrypted_item_id("deploy-1", "rs_1")},
        ]),
        Some("deploy-2")
    )]
    #[case::an_empty_item_marker_stops_the_scan(
        json!([
            {"id": encrypted_item_id("", "rs_1")},
            {"encrypted_content": wrap_encrypted_content("gAAAA", "deploy-2")},
        ]),
        None
    )]
    #[case::unmarked(json!([{"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA"}]), None)]
    #[case::text_input(json!("hello"), None)]
    fn the_first_marker_in_the_input_names_its_origin(
        #[case] input: Value,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            encrypted_content_origin(&input),
            expected.map(str::to_owned)
        );
    }

    #[test]
    fn reasoning_the_target_cannot_decrypt_keeps_only_its_summary() {
        let summary = json!([{"type": "summary_text", "text": "thought"}]);
        let mut input = json!([
            {"type": "message", "content": "hi"},
            {"type": "reasoning", "id": encrypted_item_id("east", "rs_1"), "status": "completed",
             "encrypted_content": "gAAAA", "summary": summary},
            {"type": "reasoning", "id": encrypted_item_id("west", "rs_2"),
             "encrypted_content": "gBBBB", "summary": summary},
            {"type": "reasoning", "id": "rs_3", "encrypted_content": "gCCCC", "summary": []},
            {"type": "reasoning", "id": "rs_4", "summary": summary},
        ]);
        strip_undecryptable_reasoning(&mut input, "unmarked", |origin| origin == "west");
        assert_eq!(
            input,
            json!([
                {"type": "message", "content": "hi"},
                {"type": "reasoning", "summary": summary},
                {"type": "reasoning", "id": encrypted_item_id("west", "rs_2"),
                 "encrypted_content": "gBBBB", "summary": summary},
                {"type": "reasoning", "id": "rs_4", "summary": summary},
            ])
        );
    }

    #[test]
    fn unmarked_reasoning_belongs_to_the_request_s_origin() {
        let item =
            json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA", "summary": []});
        let mut kept = json!([item]);
        strip_undecryptable_reasoning(&mut kept, "east", |origin| origin == "east");
        assert_eq!(kept, json!([item]));
    }

    #[rstest]
    #[case::output_item(
        json!({"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA"}}),
        json!({"type": "response.output_item.added", "item": {
            "type": "reasoning",
            "id": encrypted_item_id("deploy-1", "rs_1"),
            "encrypted_content": wrap_encrypted_content("gAAAA", "deploy-1"),
        }})
    )]
    #[case::reasoning_delta(
        json!({"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": "x"}),
        json!({"type": "response.reasoning_summary_text.delta", "item_id": encrypted_item_id("deploy-1", "rs_1"), "delta": "x"})
    )]
    #[case::reasoning_text_part(
        json!({"type": "response.content_part.added", "item_id": "rs_1", "part": {"type": "reasoning_text"}}),
        json!({"type": "response.content_part.added", "item_id": encrypted_item_id("deploy-1", "rs_1"), "part": {"type": "reasoning_text"}})
    )]
    #[case::message_part(
        json!({"type": "response.content_part.added", "item_id": "msg_1", "part": {"type": "output_text"}}),
        json!({"type": "response.content_part.added", "item_id": "msg_1", "part": {"type": "output_text"}})
    )]
    #[case::message_item(
        json!({"type": "response.output_item.done", "item": {"type": "message", "id": "msg_1"}}),
        json!({"type": "response.output_item.done", "item": {"type": "message", "id": "msg_1"}})
    )]
    fn stream_events_carry_the_ids_of_the_final_response(
        #[case] event: Value,
        #[case] expected: Value,
    ) {
        let Value::Object(mut event) = event else {
            unreachable!()
        };
        encode_stream_event(&mut event, &scope(Some("azure"), Some("deploy-1"), true));
        assert_eq!(Value::Object(event), expected);
    }

    #[rstest]
    #[case::with_affinity(true)]
    #[case::without_affinity(false)]
    fn a_streamed_response_is_encoded_as_the_final_one(#[case] affinity: bool) {
        let scope = scope(Some("azure"), Some("deploy-1"), affinity);
        let output = json!([{"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA"}]);
        let mut whole = response("resp_upstream", output.clone());
        encode_response(&mut whole, &scope);
        let Value::Object(mut event) = json!({
            "type": "response.completed",
            "response": {"id": "resp_upstream", "output": output},
        }) else {
            unreachable!()
        };
        encode_stream_event(&mut event, &scope);
        assert_eq!(
            (&event["response"]["id"], &event["response"]["output"]),
            (&Value::String(whole.id), &Value::Array(whole.output))
        );
    }
}
