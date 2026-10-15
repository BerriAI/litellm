//! LiteLLM-managed identifiers on Responses payloads, byte-compatible with `litellm/responses/utils.py`

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
    if let Some(model_id) = scope
        .model_id
        .as_deref()
        .filter(|_| scope.encrypted_content_affinity)
        .filter(|model_id| !model_id.is_empty())
    {
        response
            .output
            .iter_mut()
            .filter_map(Value::as_object_mut)
            .for_each(|item| encode_encrypted_item(item, model_id));
    }
    response
        .output
        .iter_mut()
        .for_each(|item| encode_containers(item, scope));
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
    let Some(Value::String(content)) = item.get_mut("encrypted_content") else {
        return;
    };
    if content.is_empty() {
        return;
    }
    *content = wrap_encrypted_content(content, model_id);
    if let Some(Value::String(id)) = item.get_mut("id")
        && !id.is_empty()
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
}
