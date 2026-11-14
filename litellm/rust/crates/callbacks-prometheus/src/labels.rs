use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{
    pyfmt::{is_opaque, py_str, py_truthy, py_type_name},
    tracker::SeriesLimits,
};

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct Settings {
    #[serde(default)]
    pub emit_stream_label: bool,
    #[serde(default)]
    pub custom_tags: Vec<String>,
    #[serde(default)]
    pub disable_end_user_cost_tracking: bool,
    #[serde(default)]
    pub end_user_prometheus_only: bool,
    #[serde(default)]
    pub end_user_max_series: Option<i64>,
    #[serde(default)]
    pub end_user_ttl_seconds: Option<f64>,
    #[serde(default)]
    pub end_user_cleanup_interval_seconds: Option<f64>,
}

impl Settings {
    pub const fn end_user_limits(&self) -> SeriesLimits {
        SeriesLimits {
            max_series: self.end_user_max_series,
            ttl_seconds: self.end_user_ttl_seconds,
            cleanup_interval_seconds: self.end_user_cleanup_interval_seconds,
        }
    }

    fn resolved_end_user(&self, end_user: &Value) -> Option<String> {
        if self.disable_end_user_cost_tracking
            || !self.end_user_prometheus_only
            || !py_truthy(end_user)
        {
            return None;
        }
        Some(py_str(end_user))
    }
}

pub fn is_py_dict(value: &Value) -> Option<&Map<String, Value>> {
    match value {
        Value::Object(map) if !is_opaque(value) => Some(map),
        _ => None,
    }
}

pub fn sanitize_label_name(label: &str) -> String {
    if label.is_empty() {
        return "_".into();
    }
    let sanitized: String = label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.starts_with(|ch: char| ch.is_ascii_alphabetic() || ch == '_') {
        sanitized
    } else {
        format!("_{sanitized}")
    }
}

pub fn sanitize_label_value(value: &Value) -> Option<String> {
    if value.is_null() {
        return None;
    }
    Some(sanitize_text(&py_str(value)))
}

pub fn sanitize_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\n' => out.push(' '),
            '\r' | '\u{2028}' | '\u{2029}' => {}
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            other => out.push(other),
        }
    }
    out
}

fn wildcard_matches(pattern: &str, text: &str) -> bool {
    let mut segments = pattern.split('*');
    let first = segments.next().unwrap_or("");
    let Some(mut rest) = text.strip_prefix(first) else {
        return false;
    };
    for segment in segments {
        match rest.find(segment) {
            Some(at) if !rest[..at].contains('\n') => rest = &rest[at + segment.len()..],
            _ => return false,
        }
    }
    true
}

pub fn custom_labels_from_tags(tags: &[Value], configured: &[String]) -> Vec<(String, String)> {
    configured
        .iter()
        .map(|tag| {
            let label = sanitize_label_name(&format!("tag_{tag}"));
            let exact = tags
                .iter()
                .any(|candidate| candidate.as_str() == Some(tag.as_str()));
            let wildcard = tag.contains('*')
                && tags
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|candidate| wildcard_matches(tag, candidate));
            let matched = if exact || wildcard { "true" } else { "false" };
            (label, matched.to_owned())
        })
        .collect()
}

pub fn tag_match_error(tags: &[Value], configured: &[String]) -> Option<String> {
    configured
        .iter()
        .filter(|tag| {
            tag.contains('*')
                && !tags
                    .iter()
                    .any(|candidate| candidate.as_str() == Some(tag.as_str()))
        })
        .find_map(|tag| {
            tags.iter()
                .find(|candidate| {
                    candidate
                        .as_str()
                        .is_none_or(|text| wildcard_matches(tag, text))
                })
                .filter(|candidate| candidate.as_str().is_none())
                .map(|candidate| {
                    format!(
                        "TypeError: expected string or bytes-like object, got '{}'",
                        py_type_name(candidate)
                    )
                })
        })
}

#[derive(Clone, Debug, Default)]
pub struct LabelValues {
    pub fields: BTreeMap<&'static str, Value>,
    pub tags: Vec<Value>,
    pub custom_metadata_labels: Vec<(String, String)>,
}

pub const ENUM_FIELDS: [&str; 37] = [
    "end_user",
    "user",
    "user_email",
    "user_alias",
    "hashed_api_key",
    "api_key_alias",
    "team",
    "team_alias",
    "project_id",
    "project_alias",
    "model_group",
    "requested_model",
    "model",
    "litellm_model_name",
    "model_id",
    "api_base",
    "api_provider",
    "exception_status",
    "exception_class",
    "rate_limit_category",
    "rate_limit_type",
    "reason",
    "priority",
    "status_code",
    "fallback_model",
    "route",
    "client_ip",
    "user_agent",
    "stream",
    "org_id",
    "org_alias",
    "mcp_tool_name",
    "mcp_server_name",
    "service_tier",
    "input_sequence_length",
    "tags",
    "custom_metadata_labels",
];

impl LabelValues {
    pub fn from_enum(values: &Map<String, Value>) -> Self {
        let tags = match values.get("tags") {
            Some(Value::Array(items)) => items.clone(),
            _ => Vec::new(),
        };
        let custom_metadata_labels = values
            .get("custom_metadata_labels")
            .and_then(is_py_dict)
            .map(|map| {
                map.iter()
                    .map(|(key, value)| (key.clone(), py_str(value)))
                    .collect()
            })
            .unwrap_or_default();
        let fields = ENUM_FIELDS
            .iter()
            .filter(|name| !matches!(**name, "tags" | "custom_metadata_labels"))
            .map(|name| (*name, values.get(*name).cloned().unwrap_or(Value::Null)))
            .collect();
        Self {
            fields,
            tags,
            custom_metadata_labels,
        }
    }

    #[must_use]
    pub fn with(mut self, field: &'static str, value: Value) -> Self {
        self.fields.insert(field, value);
        self
    }

    pub fn get(&self, field: &str) -> &Value {
        self.fields.get(field).unwrap_or(&Value::Null)
    }

    pub fn checked_context(&self, settings: &Settings) -> Result<LabelContext, String> {
        tag_match_error(&self.tags, &settings.custom_tags)
            .map_or_else(|| Ok(self.context(settings)), Err)
    }

    pub fn context(&self, settings: &Settings) -> LabelContext {
        let mut custom: BTreeMap<String, Option<String>> = BTreeMap::new();
        for (key, value) in &self.custom_metadata_labels {
            custom.insert(sanitize_label_name(key), Some(sanitize_text(value)));
        }
        let tags = custom_labels_from_tags(&self.tags, &settings.custom_tags);
        LabelContext {
            sanitized: self
                .fields
                .iter()
                .map(|(name, value)| ((*name).to_owned(), sanitize_label_value(value)))
                .collect(),
            custom,
            tags: tags
                .into_iter()
                .map(|(key, value)| (key, Some(sanitize_text(&value))))
                .collect(),
            end_user: settings.resolved_end_user(self.get("end_user")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LabelContext {
    sanitized: BTreeMap<String, Option<String>>,
    custom: BTreeMap<String, Option<String>>,
    tags: BTreeMap<String, Option<String>>,
    end_user: Option<String>,
}

impl LabelContext {
    pub fn labels(&self, supported: &[String]) -> Vec<Option<String>> {
        supported
            .iter()
            .map(|label| {
                if let Some(value) = self.tags.get(label) {
                    return value.clone();
                }
                if let Some(value) = self.custom.get(label) {
                    return value.clone();
                }
                if label == "end_user" {
                    return self.end_user.clone();
                }
                self.sanitized.get(label).cloned().flatten()
            })
            .collect()
    }
}

pub fn label_strings(values: &[Option<String>]) -> Vec<String> {
    values
        .iter()
        .map(|value| value.clone().unwrap_or_else(|| "None".into()))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case("team.name", "team_name")]
    #[case("1abc", "_1abc")]
    #[case("", "_")]
    #[case("User-Agent: curl", "User_Agent__curl")]
    fn label_names_are_sanitized_like_python(#[case] raw: &str, #[case] expected: &str) {
        assert_eq!(sanitize_label_name(raw), expected);
    }

    #[rstest]
    #[case::int_before_a_match(json!([7, "prod-1"]), &["prod*"], Some("int"))]
    #[case::none_item(json!([null]), &["prod*"], Some("NoneType"))]
    #[case::match_before_an_int(json!(["prod-1", 7]), &["prod*"], None)]
    #[case::exact_match_skips_the_pattern(json!([7, "prod*"]), &["prod*"], None)]
    #[case::no_wildcard_compares_only(json!([7, {"$py": "['x']"}]), &["prod"], None)]
    #[case::stringified_item(json!([{"$py": "['x']", "$type": "list"}]), &["x*"], Some("list"))]
    fn a_wildcard_tag_raises_on_the_first_non_str_tag_python_matches(
        #[case] tags: Value,
        #[case] configured: &[&str],
        #[case] type_name: Option<&str>,
    ) {
        let tags = tags.as_array().cloned().unwrap_or_default();
        let configured: Vec<String> = configured.iter().map(|tag| (*tag).to_owned()).collect();
        assert_eq!(
            tag_match_error(&tags, &configured),
            type_name.map(|name| format!(
                "TypeError: expected string or bytes-like object, got '{name}'"
            ))
        );
    }

    #[rstest]
    #[case(json!("a\nb\r\"c\\"), Some("a b\\\"c\\\\"))]
    #[case(json!(5), Some("5"))]
    #[case(json!(null), None)]
    #[case(json!({"$py": "['a', 1]"}), Some("['a', 1]"))]
    #[case(json!({"$py": "[]", "$truthy": false}), Some("[]"))]
    fn label_values_are_sanitized_like_python(#[case] raw: Value, #[case] expected: Option<&str>) {
        assert_eq!(sanitize_label_value(&raw).as_deref(), expected);
    }

    #[rstest]
    #[case("User-Agent: curl/*", "User-Agent: curl/7.68", true)]
    #[case("User-Agent: curl/*", "x User-Agent: curl/7", false)]
    #[case("a*c", "abXc-tail", true)]
    #[case("a*c", "ab\nc", false)]
    #[case("prod*", "prod", true)]
    fn wildcard_tags_match_like_a_python_prefix_regex(
        #[case] pattern: &str,
        #[case] tag: &str,
        #[case] matched: bool,
    ) {
        assert_eq!(wildcard_matches(pattern, tag), matched);
    }
}
