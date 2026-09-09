//! Header and upstream-body helpers shared by every route module.

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value};

use crate::constants::UPSTREAM_ERROR_BODY_MAX_CHARS;
use crate::error::{Error, json_type_name};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn http_request(
    request: reqwest::RequestBuilder,
) -> Result<reqwest::Response, reqwest::Error> {
    request.send().await
}

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
) -> Result<Vec<(String, String)>, Error> {
    extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    Error::InvalidRequest(format!(
                        "{context} extra_headers.{key} must be a string, got {}",
                        json_type_name(&value)
                    ))
                })
        })
        .collect()
}

pub fn header_map(
    context: &'static str,
    extra_headers: Option<Map<String, Value>>,
) -> Result<HeaderMap, Error> {
    extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(name, value)| {
            let value = value.as_str().ok_or_else(|| Error::InvalidHeaderType {
                context,
                name: name.clone(),
                actual: json_type_name(&value),
            })?;
            let name =
                HeaderName::from_bytes(name.as_bytes()).map_err(|_| Error::InvalidHeaderName)?;
            let mut value =
                HeaderValue::from_str(value).map_err(|_| Error::InvalidHeaderValue {
                    name: name.to_string(),
                })?;
            if matches!(
                name.as_str(),
                "authorization"
                    | "proxy-authorization"
                    | "x-api-key"
                    | "api-key"
                    | "ocp-apim-subscription-key"
            ) {
                value.set_sensitive(true);
            }
            Ok((name, value))
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
    fn header_map_validates_wire_values_without_exposing_credentials() {
        for (input, expected) in [
            (json!({"invalid name": "secret"}), Error::InvalidHeaderName),
            (
                json!({"Authorization": "secret\r\nx-injected: true"}),
                Error::InvalidHeaderValue {
                    name: "authorization".into(),
                },
            ),
            (
                json!({"x-retry": 3}),
                Error::InvalidHeaderType {
                    context: "OCR",
                    name: "x-retry".into(),
                    actual: "number",
                },
            ),
        ] {
            let error = header_map("OCR", input.as_object().cloned())
                .expect_err("invalid header must fail before sending");
            assert_eq!(error, expected);
            assert!(!error.to_string().contains("secret"));
        }
        let headers = header_map(
            "OCR",
            json!({"Authorization": "Bearer secret", "X-Api-Key": "secret", "x-trace": "trace"})
                .as_object()
                .cloned(),
        )
        .unwrap();
        let request = reqwest::Client::new()
            .get("https://example.com")
            .headers(headers)
            .build()
            .unwrap();
        assert_eq!(request.headers()["authorization"], "Bearer secret");
        assert_eq!(request.headers()["x-api-key"], "secret");
        assert_eq!(request.headers()["x-trace"], "trace");
        assert!(!format!("{request:?}").contains("secret"));
    }

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
            Error::InvalidRequest(
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
    fn auth_header_detection_is_case_insensitive() {
        let headers = vec![
            ("x-trace-id".to_string(), "trace-1".to_string()),
            ("authorization".to_string(), "Bearer sk-test".to_string()),
        ];

        assert!(has_header(&headers, "authorization"));

        let headers = vec![("Authorization".to_string(), "Bearer sk-test".to_string())];

        assert!(has_header(&headers, "authorization"));

        let headers = vec![("x-trace-id".to_string(), "trace-1".to_string())];
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
