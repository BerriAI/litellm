use serde_json::{Map, Value};

use crate::constants::UPSTREAM_ERROR_BODY_MAX_CHARS;
use crate::error::{Error, json_type_name};

#[allow(
    dead_code,
    reason = "used by the OCR architecture in the next stacked PR"
)]
pub(crate) enum HeaderPolicy<'a> {
    All,
    Only(&'a [&'a str]),
    Except(&'a [&'a str]),
}

#[allow(
    dead_code,
    reason = "used by the OCR architecture in the next stacked PR"
)]
pub(crate) fn with_headers(
    builder: reqwest::RequestBuilder,
    headers: &[(String, String)],
    policy: HeaderPolicy<'_>,
) -> reqwest::RequestBuilder {
    headers
        .iter()
        .filter(|(name, _)| match policy {
            HeaderPolicy::All => true,
            HeaderPolicy::Only(names) => names
                .iter()
                .any(|allowed| name.eq_ignore_ascii_case(allowed)),
            HeaderPolicy::Except(names) => !names
                .iter()
                .any(|excluded| name.eq_ignore_ascii_case(excluded)),
        })
        .fold(builder, |builder, (name, value)| {
            builder.header(name, value)
        })
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn http_request(
    request: reqwest::RequestBuilder,
) -> Result<reqwest::Response, reqwest::Error> {
    request.send().await
}

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

#[allow(
    dead_code,
    reason = "used by the OCR architecture in the next stacked PR"
)]
pub(crate) fn deserialize_optional_param<'de, D, T>(
    deserializer: D,
) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    <Option<T> as serde::Deserialize>::deserialize(deserializer).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[rstest::rstest]
    #[case(HeaderPolicy::All, true, true)]
    #[case(HeaderPolicy::Only(&["authorization"]), true, false)]
    #[case(HeaderPolicy::Except(&["authorization"]), false, true)]
    fn forwarding_policy_preserves_matching_headers_and_duplicates(
        #[case] policy: HeaderPolicy<'_>,
        #[case] auth: bool,
        #[case] trace: bool,
    ) {
        let request = with_headers(
            reqwest::Client::new().get("https://example.com"),
            &[
                ("AuThOrIzAtIoN".into(), "Bearer token".into()),
                ("X-Trace".into(), "first".into()),
                ("x-trace".into(), "second".into()),
            ],
            policy,
        )
        .build()
        .unwrap();
        assert_eq!(request.headers().contains_key("authorization"), auth);
        let traces: Vec<_> = request.headers().get_all("x-trace").iter().collect();
        if trace {
            assert_eq!(traces, ["first", "second"]);
        } else {
            assert!(traces.is_empty());
        }
    }

    #[test]
    fn multipart_policy_leaves_content_headers_to_reqwest() {
        let request = with_headers(
            reqwest::Client::new()
                .post("https://example.com")
                .multipart(reqwest::multipart::Form::new().text("file", "abc")),
            &[
                ("Content-Type".into(), "application/json".into()),
                ("CONTENT-LENGTH".into(), "0".into()),
            ],
            HeaderPolicy::Except(&["content-type", "content-length"]),
        )
        .build()
        .unwrap();
        assert!(
            request.headers()["content-type"]
                .to_str()
                .unwrap()
                .starts_with("multipart/form-data; boundary=")
        );
        assert_ne!(request.headers()["content-length"], "0");
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
