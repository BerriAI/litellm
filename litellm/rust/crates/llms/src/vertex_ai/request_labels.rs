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
use serde::{
    Deserialize, Deserializer,
    de::{IgnoredAny, MapAccess, Visitor},
};
use serde_json::Value;
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

/// A JSON value the labels read only when it is text.
#[derive(Deserialize)]
#[serde(untagged)]
enum MaybeText {
    Text(String),
    Other(IgnoredAny),
}

impl MaybeText {
    fn text(self) -> Option<String> {
        match self {
            Self::Text(text) => Some(text),
            Self::Other(_) => None,
        }
    }
}

/// The text pairs of a JSON object in its order, skipping values that are not text.
#[derive(Clone, Debug, Default, PartialEq)]
struct TextPairs(Vec<Label>);

impl<'de> Deserialize<'de> for TextPairs {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Pairs;

        impl<'de> Visitor<'de> for Pairs {
            type Value = TextPairs;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<TextPairs, A::Error> {
                let mut pairs = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, MaybeText>()? {
                    pairs.extend(value.text().map(|value| (key, value)));
                }
                Ok(TextPairs(pairs))
            }
        }

        deserializer.deserialize_map(Pairs)
    }
}

/// The call's `litellm_metadata` as the labels read it: its text fields, and the text pairs of
/// its `spend_logs_metadata`.
#[derive(Clone, Debug, Default, PartialEq)]
struct LabelMetadata {
    fields: Vec<Label>,
    spend_logs: Vec<Label>,
}

impl<'de> Deserialize<'de> for LabelMetadata {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum SpendLogs {
            Pairs(TextPairs),
            Other(IgnoredAny),
        }

        struct Metadata;

        impl<'de> Visitor<'de> for Metadata {
            type Value = LabelMetadata;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<LabelMetadata, A::Error> {
                let mut metadata = LabelMetadata::default();
                while let Some(key) = map.next_key::<String>()? {
                    if key == CLIENT_FIELD {
                        if let SpendLogs::Pairs(pairs) = map.next_value()? {
                            metadata.spend_logs = pairs.0;
                        }
                        continue;
                    }
                    let value: MaybeText = map.next_value()?;
                    metadata
                        .fields
                        .extend(value.text().map(|value| (key, value)));
                }
                Ok(metadata)
            }
        }

        deserializer.deserialize_map(Metadata)
    }
}

/// `litellm.vertex_request_label_fields` and the call's `litellm_metadata`, read once from the
/// params the host passes.
#[derive(Clone, Debug, Default, PartialEq)]
struct VertexLabelParams {
    fields: Vec<String>,
    metadata: Option<LabelMetadata>,
}

impl VertexLabelParams {
    fn from_extra(extra: &ExtraParams) -> Self {
        let fields = extra
            .get(REQUEST_LABEL_FIELDS_PARAM)
            .and_then(|fields| Vec::<MaybeText>::deserialize(fields).ok())
            .unwrap_or_default()
            .into_iter()
            .filter_map(MaybeText::text)
            .collect();
        let metadata = extra
            .get(LITELLM_METADATA_PARAM)
            .and_then(|metadata| LabelMetadata::deserialize(metadata).ok());
        Self { fields, metadata }
    }
}

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

fn identity_labels(metadata: Option<&LabelMetadata>, allowed: &[String]) -> Vec<Label> {
    allowed
        .iter()
        .filter(|field| is_reserved(field))
        .filter_map(|field| {
            let (_, value) = metadata?.fields.iter().find(|(key, _)| key == field)?;
            normalized_label(field, value)
        })
        .collect()
}

fn spend_logs_labels(metadata: Option<&LabelMetadata>) -> Vec<Label> {
    metadata
        .map(|metadata| metadata.spend_logs.as_slice())
        .unwrap_or_default()
        .iter()
        .filter_map(|(key, value)| normalized_label(key, value))
        .collect()
}

/// The caller's header counts only when it is strict base64 of a JSON object, and only its string
/// pairs are kept. JSON that standard parsers refuse (`NaN`, `Infinity`) drops the header.
fn caller_header_labels(encoded: &str) -> Vec<Label> {
    let Some(Value::Object(pairs)) = STRICT_BASE64
        .decode(encoded)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
    else {
        return Vec::new();
    };
    pairs
        .into_iter()
        .filter_map(|(key, value)| match value {
            Value::String(value) => Some((key, value)),
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
    let VertexLabelParams {
        fields: allowed,
        metadata,
    } = VertexLabelParams::from_extra(extra);
    if allowed.is_empty() {
        return headers;
    }
    let metadata = metadata.as_ref();
    let caller = headers
        .iter()
        .find(|(name, _)| is_labels_header(name))
        .map(|(_, value)| caller_header_labels(value))
        .unwrap_or_default();
    let spend_logs = match allowed.iter().any(|field| field == CLIENT_FIELD) {
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
    use serde_json::{Map, json};

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
    #[case::metadata_that_is_not_text_is_skipped(
        vec![],
        json!({
            "vertex_request_label_fields": ["user_api_key_team_id", "user_api_key_org_id", 7, "spend_logs_metadata"],
            "litellm_metadata": {
                "user_api_key_team_id": 5, "user_api_key_org_id": "o1",
                "spend_logs_metadata": {"n": 1, "env": "Prod", "nested": {"a": "b"}}
            }
        }),
        Some(r#"{"user_api_key_org_id": "o1", "env": "prod"}"#)
    )]
    #[case::spend_logs_that_are_not_an_object_are_skipped(
        vec![],
        json!({
            "vertex_request_label_fields": ["user_api_key_org_id", "spend_logs_metadata"],
            "litellm_metadata": {"user_api_key_org_id": "o1", "spend_logs_metadata": "env=prod"}
        }),
        Some(r#"{"user_api_key_org_id": "o1"}"#)
    )]
    #[case::fields_that_are_not_a_list_set_nothing(
        vec![("X-Vertex-AI-Labels", encoded(r#"{"env": "prod"}"#))],
        json!({"vertex_request_label_fields": "user_api_key_org_id", "litellm_metadata": {"user_api_key_org_id": "o1"}}),
        Some(r#"{"env": "prod"}"#)
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

    #[rstest]
    #[case::string_pairs(r#"{"team": "a", "n": 1, "env": "prod"}"#, vec![("team", "a"), ("env", "prod")])]
    #[case::duplicate_key_keeps_first_position(r#"{"a": "1", "b": "2", "a": "3"}"#, vec![("a", "3"), ("b", "2")])]
    #[case::nan_drops_the_header(r#"{"team": "a", "x": NaN}"#, vec![])]
    #[case::infinity_drops_the_header(r#"{"team": "a", "x": Infinity}"#, vec![])]
    #[case::not_an_object(r#"["team"]"#, vec![])]
    fn the_callers_header_keeps_the_string_pairs_of_a_json_object(
        #[case] json: &str,
        #[case] expected: Vec<(&str, &str)>,
    ) {
        let labels = caller_header_labels(&general_purpose::STANDARD.encode(json));
        assert_eq!(
            labels,
            expected
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
                .collect::<Vec<_>>()
        );
    }
}
