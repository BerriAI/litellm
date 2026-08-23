//! Header and upstream-body helpers shared by every route module.

use serde_json::{Map, Value};

use crate::constants::UPSTREAM_ERROR_BODY_MAX_CHARS;
use crate::error::{CoreError, CoreResult, json_type_name};

/// Bound an upstream error body before it crosses a host boundary, so provider
/// bodies stay data-minimized.
pub fn truncate_error_body(body: &str) -> String {
    if body.chars().count() <= UPSTREAM_ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(UPSTREAM_ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
}

pub fn string_headers(
    context: &'static str,
    extra_headers: Option<Map<String, Value>>,
) -> CoreResult<Vec<(String, String)>> {
    extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    CoreError::InvalidRequest(format!(
                        "{context} extra_headers.{key} must be a string, got {}",
                        json_type_name(&value)
                    ))
                })
        })
        .collect()
}

pub fn has_header(headers: &[(String, String)], name: &str) -> bool {
    headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case(name))
}

pub fn has_bearer_auth(headers: &[(String, String)]) -> bool {
    headers.iter().any(|(name, value)| {
        if !name.eq_ignore_ascii_case("authorization") {
            return false;
        }
        let value = value.trim();
        value.len() > 7
            && value[..7].eq_ignore_ascii_case("bearer ")
            && !value[7..].trim().is_empty()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn truncate_leaves_short_bodies_untouched() {
        assert_eq!(truncate_error_body("short"), "short");
    }

    #[test]
    fn truncate_bounds_long_bodies_by_characters() {
        let body = "\u{00e9}".repeat(UPSTREAM_ERROR_BODY_MAX_CHARS + 10);
        let truncated = truncate_error_body(&body);
        assert!(truncated.ends_with("... (truncated)"));
        assert_eq!(
            truncated.chars().count(),
            UPSTREAM_ERROR_BODY_MAX_CHARS + "... (truncated)".chars().count()
        );
    }

    #[test]
    fn string_headers_rejects_non_string_values() {
        let headers = Map::from_iter([("x-trace".to_string(), json!(7))]);
        let err = string_headers("chat completions", Some(headers)).expect_err("non-string value");
        assert_eq!(
            err,
            CoreError::InvalidRequest(
                "chat completions extra_headers.x-trace must be a string, got number".to_string()
            )
        );
    }

    #[test]
    fn header_lookup_is_case_insensitive() {
        let headers = vec![("X-Api-Key".to_string(), "k".to_string())];
        assert!(has_header(&headers, "x-api-key"));
        assert!(!has_header(&headers, "authorization"));
    }

    #[test]
    fn bearer_detection_requires_a_non_empty_token() {
        assert!(has_bearer_auth(&[(
            "Authorization".to_string(),
            "Bearer abc".to_string()
        )]));
        assert!(!has_bearer_auth(&[(
            "Authorization".to_string(),
            "Bearer    ".to_string()
        )]));
        assert!(!has_bearer_auth(&[(
            "Authorization".to_string(),
            "Basic abc".to_string()
        )]));
    }
}
