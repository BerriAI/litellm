//! Python's `litellm/llms/vertex_ai/request_labels.py` for partner models: once
//! `litellm.vertex_request_label_fields` is set, the proxy owns the `X-Vertex-AI-Labels` header
//! that Cloud Billing reads, so a caller can neither evict nor forge the identity labels in it.

use base64::{Engine, engine::general_purpose};
use litellm_python_compat::{
    base64::b64decode_validate,
    unicode::{category, isspace},
};
use litellm_router_types::ExtraParams;
use serde::{
    Deserialize, Deserializer,
    de::{IgnoredAny, MapAccess, Visitor},
};
use serde_json::Value;

use crate::base_llm::auth::Headers;

pub const REQUEST_LABEL_FIELDS_PARAM: &str = "vertex_request_label_fields";
pub const LITELLM_METADATA_PARAM: &str = "litellm_metadata";
pub const LABELS_HEADER: &str = "X-Vertex-AI-Labels";

const PARTNER_REQUEST_LABELS_MAX: usize = 32;
const REQUEST_LABEL_MAX_CHARS: usize = 63;
const IDENTITY_PREFIX: &str = "user_api_key_";
const CLIENT_FIELD: &str = "spend_logs_metadata";

/// The `unicodedata` categories a label may hold besides `_` and `-`.
const LABEL_CHARACTER_CATEGORIES: [&str; 5] = ["Ll", "Lo", "Nd", "Nl", "No"];
/// The `unicodedata` categories a label key may start with.
const KEY_START_CATEGORIES: [&str; 2] = ["Ll", "Lo"];

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
    matches!(character, '_' | '-') || LABEL_CHARACTER_CATEGORIES.contains(&category(character))
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
    let starts_with_a_letter = key
        .chars()
        .next()
        .is_some_and(|first| KEY_START_CATEGORIES.contains(&category(first)));
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
    let Some(Value::Object(pairs)) =
        b64decode_validate(encoded).and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
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

/// The first label of each key, in order, stopping once the most Vertex AI keeps are found.
fn first_occurrences(labels: impl IntoIterator<Item = Label>) -> Vec<Label> {
    let mut seen = std::collections::HashSet::new();
    labels
        .into_iter()
        .filter(|(key, _)| seen.insert(key.clone()))
        .take(PARTNER_REQUEST_LABELS_MAX)
        .collect()
}

fn header_value(labels: Vec<Label>) -> Option<String> {
    if labels.is_empty() {
        return None;
    }
    let object: serde_json::Map<String, Value> = labels
        .into_iter()
        .map(|(key, value)| (key, Value::String(value)))
        .collect();
    Some(general_purpose::STANDARD.encode(Value::Object(object).to_string()))
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

    fn labels_in_order(json: &str) -> Vec<(String, Value)> {
        serde_json::from_str::<Map<String, Value>>(json)
            .unwrap()
            .into_iter()
            .collect()
    }

    /// The labels the sent header holds, in their order.
    fn sent_labels(headers: &[(&str, String)], params: Value) -> Option<Vec<(String, Value)>> {
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
            labels_in_order(std::str::from_utf8(&decoded).unwrap())
        })
    }

    #[rstest]
    fn first_occurrences_keep_each_key_once_in_order_up_to_the_limit() {
        let label = |key: usize, value: &str| (format!("k{key}"), value.to_owned());
        let labels = [label(1, "first"), label(0, "a"), label(1, "again")]
            .into_iter()
            .chain((0..200_000).map(|key| label(key, "late")));
        let kept = super::first_occurrences(labels);
        assert_eq!(kept.len(), super::PARTNER_REQUEST_LABELS_MAX);
        assert_eq!(
            kept[..3],
            [label(1, "first"), label(0, "a"), label(2, "late")]
        );
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
    #[case::a_header_python_cannot_encode_is_replaced(
        vec![("X-Vertex-AI-Labels", "ZWJ\u{200d}joined".to_string())],
        json!({
            "vertex_request_label_fields": ["user_api_key_hash"],
            "litellm_metadata": {"user_api_key_hash": "abc"}
        }),
        Some(r#"{"user_api_key_hash": "abc"}"#)
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
        assert_eq!(sent_labels(&headers, params), expected.map(labels_in_order));
    }

    #[rstest]
    fn the_header_is_compact_utf8_json() {
        let value = header_value(vec![
            ("team".into(), "équipe".into()),
            ("env".into(), "prod".into()),
        ])
        .unwrap();
        let decoded = general_purpose::STANDARD.decode(value).unwrap();
        assert_eq!(
            std::str::from_utf8(&decoded).unwrap(),
            r#"{"team":"équipe","env":"prod"}"#
        );
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
        assert_eq!(sent.len(), 32);
        assert_eq!(sent.first().unwrap().0, "user_api_key_user_id");
        assert_eq!(sent.last().unwrap().0, "k30");
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
