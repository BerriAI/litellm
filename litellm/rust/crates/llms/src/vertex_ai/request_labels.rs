//! Python's `litellm/llms/vertex_ai/request_labels.py` for partner models: once
//! `litellm.vertex_request_label_fields` is set, the proxy owns the `X-Vertex-AI-Labels` header
//! that Cloud Billing reads, so a caller can neither evict nor forge the identity labels in it.

use base64::{
    Engine,
    alphabet::STANDARD,
    engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig, general_purpose},
};
use litellm_python_compat::{Value as PyValue, json as python_json, unicode::isspace};
use litellm_router_types::ExtraParams;
use serde_json::{Map, Value};
use unicode_general_category::{GeneralCategory, get_general_category};

use crate::base_llm::auth::Headers;

pub const REQUEST_LABEL_FIELDS_PARAM: &str = "vertex_request_label_fields";
pub const LITELLM_METADATA_PARAM: &str = "litellm_metadata";
pub const LABELS_HEADER: &str = "X-Vertex-AI-Labels";

const PARTNER_REQUEST_LABELS_MAX: usize = 32;
const REQUEST_LABEL_MAX_CHARS: usize = 63;
const IDENTITY_PREFIX: &str = "user_api_key_";
const CLIENT_FIELD: &str = "spend_logs_metadata";

/// Python's `base64.b64decode(text, validate=True)`: canonical padding, and any trailing bits.
const STRICT_BASE64: GeneralPurpose = GeneralPurpose::new(
    &STANDARD,
    GeneralPurposeConfig::new()
        .with_decode_padding_mode(DecodePaddingMode::RequireCanonical)
        .with_decode_allow_trailing_bits(true),
);

type Label = (String, String);

fn is_label_character(character: char) -> bool {
    matches!(character, '_' | '-')
        || matches!(
            get_general_category(character),
            GeneralCategory::LowercaseLetter
                | GeneralCategory::OtherLetter
                | GeneralCategory::DecimalNumber
                | GeneralCategory::LetterNumber
                | GeneralCategory::OtherNumber
        )
}

fn normalized(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|character| match is_label_character(character) {
            true => character,
            false => '_',
        })
        .take(REQUEST_LABEL_MAX_CHARS)
        .collect()
}

fn normalized_label(key: &str, value: &str) -> Option<Label> {
    let key = normalized(key);
    let starts_with_a_letter = key.chars().next().is_some_and(|first| {
        matches!(
            get_general_category(first),
            GeneralCategory::LowercaseLetter | GeneralCategory::OtherLetter
        )
    });
    starts_with_a_letter.then(|| (key, normalized(value)))
}

fn is_reserved(key: &str) -> bool {
    key.trim_matches(isspace)
        .to_lowercase()
        .starts_with(IDENTITY_PREFIX)
}

fn allowed_fields(extra: &ExtraParams) -> Vec<&str> {
    match extra.get(REQUEST_LABEL_FIELDS_PARAM) {
        Some(Value::Array(fields)) => fields.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn string_pairs(source: Option<&Value>) -> Vec<(&str, &str)> {
    match source {
        Some(Value::Object(fields)) => fields
            .iter()
            .filter_map(|(key, value)| Some((key.as_str(), value.as_str()?)))
            .collect(),
        _ => Vec::new(),
    }
}

fn identity_labels(metadata: Option<&Map<String, Value>>, allowed: &[&str]) -> Vec<Label> {
    allowed
        .iter()
        .filter(|field| is_reserved(field))
        .filter_map(|field| {
            let value = metadata?.get(*field)?.as_str()?;
            normalized_label(field, value)
        })
        .collect()
}

fn spend_logs_labels(metadata: Option<&Map<String, Value>>) -> Vec<Label> {
    string_pairs(metadata.and_then(|metadata| metadata.get(CLIENT_FIELD)))
        .into_iter()
        .filter_map(|(key, value)| normalized_label(key, value))
        .collect()
}

/// Python's `_decoded_labels_header` then `string_labels`: the caller's header counts only when
/// it decodes to a JSON object, and only its string pairs are kept.
fn caller_header_labels(encoded: &str) -> Vec<Label> {
    let Some(PyValue::Dict(pairs)) = STRICT_BASE64
        .decode(encoded)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .and_then(|text| python_json::loads(&text).ok())
    else {
        return Vec::new();
    };
    pairs
        .into_iter()
        .filter_map(|pair| match pair {
            (PyValue::Str(key), PyValue::Str(value)) => Some((key, value)),
            _ => None,
        })
        .collect()
}

fn first_occurrences(labels: impl IntoIterator<Item = Label>) -> Vec<Label> {
    labels
        .into_iter()
        .fold(Vec::new(), |mut kept: Vec<Label>, (key, value)| {
            if !kept.iter().any(|(existing, _)| *existing == key) {
                kept.push((key, value));
            }
            kept
        })
        .into_iter()
        .take(PARTNER_REQUEST_LABELS_MAX)
        .collect()
}

fn header_value(labels: Vec<Label>) -> Option<String> {
    if labels.is_empty() {
        return None;
    }
    let object = PyValue::Dict(
        labels
            .into_iter()
            .map(|(key, value)| (PyValue::Str(key), PyValue::Str(value)))
            .collect(),
    );
    python_json::dumps(&object)
        .ok()
        .map(|json| general_purpose::STANDARD.encode(json))
}

fn is_labels_header(name: &str) -> bool {
    name.eq_ignore_ascii_case(LABELS_HEADER)
}

/// Python's `_labels_header_replacement` over the outbound headers: with label fields set, every
/// caller `X-Vertex-AI-Labels` header goes, and one built from the call's identity, the caller's
/// first decodable header and, when allowed, `spend_logs_metadata` takes its place.
pub(crate) fn with_partner_labels_header(headers: Headers, extra: &ExtraParams) -> Headers {
    let allowed = allowed_fields(extra);
    if allowed.is_empty() {
        return headers;
    }
    let metadata = extra.get(LITELLM_METADATA_PARAM).and_then(Value::as_object);
    let caller = headers
        .iter()
        .find(|(name, _)| is_labels_header(name))
        .map(|(_, value)| caller_header_labels(value))
        .unwrap_or_default();
    let spend_logs = match allowed.contains(&CLIENT_FIELD) {
        true => spend_logs_labels(metadata),
        false => Vec::new(),
    };
    let client = caller
        .into_iter()
        .chain(spend_logs)
        .filter(|(key, _)| !is_reserved(key));
    let labels = first_occurrences(
        identity_labels(metadata, &allowed)
            .into_iter()
            .chain(client),
    );
    headers
        .into_iter()
        .filter(|(name, _)| !is_labels_header(name))
        .chain(header_value(labels).map(|value| (LABELS_HEADER.to_string(), value)))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn extra(value: Value) -> ExtraParams {
        value
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }

    fn encoded(json: &str) -> String {
        general_purpose::STANDARD.encode(json)
    }

    fn sent_labels(headers: &[(&str, String)], params: Value) -> Option<String> {
        let headers = headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        let sent = with_partner_labels_header(headers, &extra(params));
        let labels: Vec<&(String, String)> = sent
            .iter()
            .filter(|(name, _)| is_labels_header(name))
            .collect();
        assert!(labels.len() <= 1);
        labels.first().map(|(name, value)| {
            assert_eq!(name, LABELS_HEADER);
            let decoded = general_purpose::STANDARD.decode(value).unwrap();
            String::from_utf8(decoded).unwrap()
        })
    }

    #[rstest]
    #[case::identity_is_normalized(
        vec![],
        json!({
            "vertex_request_label_fields": ["user_api_key_team_alias", "user_api_key_org_id"],
            "litellm_metadata": {"user_api_key_team_alias": "AI Platform", "user_api_key_org_id": "Org.1"}
        }),
        Some(r#"{"user_api_key_team_alias": "ai_platform", "user_api_key_org_id": "org_1"}"#)
    )]
    #[case::unreserved_fields_are_never_identity(
        vec![],
        json!({"vertex_request_label_fields": ["team"], "litellm_metadata": {"team": "t1"}}),
        None
    )]
    #[case::the_callers_header_follows_identity_and_cannot_forge_it(
        vec![("x-vertex-ai-labels", encoded(r#"{"env": "Prod", "user_api_key_hash": "forged", "n": 1}"#))],
        json!({
            "vertex_request_label_fields": ["user_api_key_hash"],
            "litellm_metadata": {"user_api_key_hash": "abc"}
        }),
        Some(r#"{"user_api_key_hash": "abc", "env": "Prod"}"#)
    )]
    #[case::an_undecodable_header_is_dropped(
        vec![("X-Vertex-AI-Labels", "not base64!".to_string())],
        json!({"vertex_request_label_fields": ["user_api_key_hash"]}),
        None
    )]
    #[case::spend_logs_only_when_allowed(
        vec![],
        json!({
            "vertex_request_label_fields": ["user_api_key_team_id"],
            "litellm_metadata": {"user_api_key_team_id": "t1", "spend_logs_metadata": {"project": "a"}}
        }),
        Some(r#"{"user_api_key_team_id": "t1"}"#)
    )]
    #[case::spend_logs_are_normalized_and_keep_their_first_value(
        vec![("x-vertex-ai-labels", encoded(r#"{"project": "Caller"}"#))],
        json!({
            "vertex_request_label_fields": ["spend_logs_metadata"],
            "litellm_metadata": {"spend_logs_metadata": {"Project": "B", "9lives": "x", "user_api_key_alias": "f", "cost center": "R&D"}}
        }),
        Some(r#"{"project": "Caller", "cost_center": "r_d"}"#)
    )]
    fn the_labels_header_matches_python(
        #[case] headers: Vec<(&str, String)>,
        #[case] params: Value,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(sent_labels(&headers, params).as_deref(), expected);
    }

    #[rstest]
    fn without_label_fields_the_callers_header_is_sent_as_is() {
        let headers = vec![("X-Vertex-AI-Labels".to_string(), "raw".to_string())];
        assert_eq!(
            with_partner_labels_header(headers.clone(), &extra(json!({"litellm_metadata": {}}))),
            headers
        );
    }

    #[rstest]
    fn at_most_32_labels_are_sent_identity_first() {
        let spend_logs: Map<String, Value> = (0..40)
            .map(|index| (format!("k{index:02}"), json!("v")))
            .collect();
        let sent = sent_labels(
            &[],
            json!({
                "vertex_request_label_fields": ["spend_logs_metadata", "user_api_key_user_id"],
                "litellm_metadata": {"user_api_key_user_id": "u", "spend_logs_metadata": spend_logs}
            }),
        )
        .unwrap();
        let labels: Map<String, Value> = serde_json::from_str(&sent).unwrap();
        assert_eq!(labels.len(), 32);
        assert_eq!(labels.keys().next().unwrap(), "user_api_key_user_id");
        assert_eq!(labels.keys().next_back().unwrap(), "k30");
    }

    #[rstest]
    #[case::long_values_are_cut(&"A".repeat(70), &"a".repeat(63))]
    #[case::non_latin_letters_are_kept("Équipe 東京", "équipe_東京")]
    fn values_are_lowercased_and_cut_to_63_characters(#[case] value: &str, #[case] expected: &str) {
        assert_eq!(
            normalized_label("user_api_key_alias", value),
            Some(("user_api_key_alias".to_string(), expected.to_string()))
        );
    }

    #[rstest]
    #[case::canonical("YQ==", true)]
    #[case::trailing_bits("YR==", true)]
    #[case::missing_padding("YQ", false)]
    #[case::whitespace(" YQ==", false)]
    fn the_header_decodes_as_strictly_as_python(#[case] encoded: &str, #[case] decodes: bool) {
        assert_eq!(STRICT_BASE64.decode(encoded).is_ok(), decodes);
    }
}
