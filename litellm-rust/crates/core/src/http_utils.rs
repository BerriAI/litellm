//! Header and upstream-body helpers shared by every route module.

#![allow(
    clippy::disallowed_methods,
    reason = "this module owns the workspace's bounded, reusable HTTP clients"
)]

use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{Map, Value};

use crate::constants::{
    HTTP_CLIENT_CONNECT_TIMEOUT_SECS, HTTP_CLIENT_TIMEOUT_SECS, UPSTREAM_ERROR_BODY_MAX_CHARS,
};
use crate::error::{CoreError, CoreResult, json_type_name};

pub mod safe_fetch;

pub fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(HTTP_CLIENT_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(HTTP_CLIENT_CONNECT_TIMEOUT_SECS))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
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

/// Classify a failed `reqwest` send so the retry contract stays accurate.
///
/// Failing to establish the connection means the request never went out, so
/// the host can still serve it. Everything else here, a timeout above all,
/// may have reached the provider and been answered.
pub fn map_send_error(err: reqwest::Error) -> CoreError {
    if err.is_connect() || err.is_builder() {
        CoreError::connect(err.to_string())
    } else {
        CoreError::network(err.to_string())
    }
}

/// Build the canonical upstream-status error, truncating the provider body.
pub fn upstream_http(status: reqwest::StatusCode, body: &str) -> CoreError {
    CoreError::http(status.as_u16(), truncate_error_body(body))
}

pub async fn json_response(
    response: reqwest::Response,
    invalid_json_context: &'static str,
) -> CoreResult<Value> {
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .map_err(|error| CoreError::network(error.to_string()))?;
    if !status.is_success() {
        return Err(upstream_http(status, &String::from_utf8_lossy(&bytes)));
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| CoreError::invalid_response(format!("{invalid_json_context}: {error}")))
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
                    CoreError::invalid_request(format!(
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
            CoreError::invalid_request(
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

    #[test]
    fn upstream_http_truncates_the_provider_body() {
        let body = "x".repeat(UPSTREAM_ERROR_BODY_MAX_CHARS + 1);
        let err = upstream_http(reqwest::StatusCode::BAD_GATEWAY, &body);
        assert!(matches!(
            err,
            CoreError::Upstream(crate::UpstreamError::Http { status: 502, .. })
        ));
        match err {
            CoreError::Upstream(crate::UpstreamError::Http { body, .. }) => {
                assert_eq!(body, truncate_error_body(&body));
                assert!(body.ends_with("... (truncated)"));
            }
            other => panic!("expected an upstream http error, got {other:?}"),
        }
    }
}
