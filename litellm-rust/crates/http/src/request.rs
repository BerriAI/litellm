use http::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use litellm_core_utils::core_helpers::json_type_name;
use litellm_error::Source;
use serde_json::{Map, Value};

use crate::Error;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum HeaderError {
    #[error("invalid request: {context} extra_headers.{name} must be a string, got {actual}")]
    NotAString {
        context: &'static str,
        name: String,
        actual: &'static str,
    },
    #[error("invalid request: {context} extra_headers.{name:?} is not a valid header name")]
    InvalidName {
        context: &'static str,
        name: String,
        #[source]
        source: Source,
    },
    #[error("invalid request: {context} extra_headers.{name} is not a valid header value")]
    InvalidValue {
        context: &'static str,
        name: HeaderName,
        #[source]
        source: Source,
    },
}

pub enum HeaderPolicy<'a> {
    All,
    Only(&'a [&'a str]),
    Except(&'a [&'a str]),
}

pub fn with_headers(
    builder: reqwest::RequestBuilder,
    headers: &HeaderMap,
    policy: HeaderPolicy<'_>,
) -> reqwest::RequestBuilder {
    headers
        .iter()
        .filter(|(name, _)| match policy {
            HeaderPolicy::All => true,
            HeaderPolicy::Only(names) => names
                .iter()
                .any(|allowed| name.as_str().eq_ignore_ascii_case(allowed)),
            HeaderPolicy::Except(names) => !names
                .iter()
                .any(|excluded| name.as_str().eq_ignore_ascii_case(excluded)),
        })
        .fold(builder, |builder, (name, value)| {
            builder.header(name, value)
        })
}

pub async fn http_request(request: reqwest::RequestBuilder) -> Result<reqwest::Response, Error> {
    Ok(request.send().await?)
}

pub async fn execute_http_request(
    client: &reqwest::Client,
    request: reqwest::Request,
) -> Result<reqwest::Response, Error> {
    Ok(client.execute(request).await?)
}

pub fn string_headers(
    context: &'static str,
    extra_headers: Option<Map<String, Value>>,
) -> Result<HeaderMap, HeaderError> {
    extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(name, value)| string_header(context, name, value))
        .collect()
}

fn string_header(
    context: &'static str,
    name: String,
    value: Value,
) -> Result<(HeaderName, HeaderValue), HeaderError> {
    let Value::String(value) = value else {
        return Err(HeaderError::NotAString {
            context,
            actual: json_type_name(&value),
            name,
        });
    };
    let name = match HeaderName::try_from(name.as_str()) {
        Ok(parsed) => parsed,
        Err(source) => {
            return Err(HeaderError::InvalidName {
                context,
                name,
                source: Source::new(source),
            });
        }
    };
    match HeaderValue::try_from(value) {
        Ok(value) => Ok((name, value)),
        Err(source) => Err(HeaderError::InvalidValue {
            context,
            name,
            source: Source::new(source),
        }),
    }
}

pub fn with_default_headers(
    headers: HeaderMap,
    defaults: &[(&'static str, &'static str)],
) -> HeaderMap {
    defaults
        .iter()
        .fold(headers, |mut headers, &(name, value)| {
            headers
                .entry(name)
                .or_insert(HeaderValue::from_static(value));
            headers
        })
}

pub fn has_bearer_auth(headers: &HeaderMap) -> bool {
    headers.get_all(AUTHORIZATION).iter().any(|value| {
        let Ok(value) = value.to_str() else {
            return false;
        };
        let value = value.trim();
        value.len() > 7
            && value
                .get(..7)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("bearer "))
            && !value[7..].trim().is_empty()
    })
}

#[cfg(any(test, feature = "test-support"))]
pub fn header_map(pairs: &[(&str, &str)]) -> HeaderMap {
    pairs
        .iter()
        .map(|&(name, value)| {
            (
                HeaderName::try_from(name).expect("test header name"),
                HeaderValue::try_from(value).expect("test header value"),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
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
            &header_map(&[
                ("AuThOrIzAtIoN", "Bearer token"),
                ("X-Trace", "first"),
                ("x-trace", "second"),
            ]),
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
            &header_map(&[
                ("Content-Type", "application/json"),
                ("CONTENT-LENGTH", "0"),
            ]),
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
    fn string_headers_are_matched_in_any_case() {
        let headers = string_headers(
            "chat completions",
            Some(Map::from_iter([("X-Api-Key".to_string(), json!("k"))])),
        )
        .unwrap();
        assert_eq!(headers, header_map(&[("x-api-key", "k")]));
        assert_eq!(headers["X-API-KEY"], "k");
    }

    #[test]
    fn string_headers_rejects_non_string_values() {
        let headers = Map::from_iter([("x-trace".to_string(), json!(7))]);
        let err = string_headers("chat completions", Some(headers)).expect_err("non-string value");
        assert_eq!(
            err,
            HeaderError::NotAString {
                context: "chat completions",
                name: "x-trace".into(),
                actual: "number"
            }
        );
    }

    #[rstest]
    #[case::space_in_name("x trace", "v", true)]
    #[case::empty_name("", "v", true)]
    #[case::newline_in_value("x-trace", "a\r\nx-injected: 1", false)]
    #[case::delete_in_value("x-trace", "\u{7f}", false)]
    fn string_headers_rejects_what_cannot_go_on_the_wire(
        #[case] name: &str,
        #[case] value: &str,
        #[case] bad_name: bool,
    ) {
        let headers = Map::from_iter([(name.to_string(), json!(value))]);
        let err = string_headers("messages", Some(headers)).expect_err("invalid header");
        match err {
            HeaderError::InvalidName {
                name: ref rejected, ..
            } => {
                assert!(bad_name);
                assert_eq!(rejected, name);
            }
            HeaderError::InvalidValue {
                name: ref rejected, ..
            } => {
                assert!(!bad_name);
                assert_eq!(rejected, name);
                let report = litellm_error::Report::new(&err).to_string();
                assert!(!report.contains(value), "{report}");
            }
            HeaderError::NotAString { .. } => panic!("value was a string"),
        }
    }

    #[rstest]
    #[case::nothing_forwarded(
        &[],
        &[("x-version", "1"), ("content-type", "application/json")],
        &[("x-version", "1"), ("content-type", "application/json")],
    )]
    #[case::forwarded_header_wins_in_any_case(
        &[("X-Version", "custom"), ("x-api-key", "k")],
        &[("x-version", "1"), ("content-type", "application/json")],
        &[("x-version", "custom"), ("x-api-key", "k"), ("content-type", "application/json")],
    )]
    #[case::repeated_forwarded_header_is_kept(
        &[("x-version", "a"), ("x-version", "b")],
        &[("x-version", "1")],
        &[("x-version", "a"), ("x-version", "b")],
    )]
    #[case::no_defaults(&[("x-api-key", "k")], &[], &[("x-api-key", "k")])]
    fn default_headers_fill_only_missing_names(
        #[case] forwarded: &[(&str, &str)],
        #[case] defaults: &[(&'static str, &'static str)],
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(
            with_default_headers(header_map(forwarded), defaults),
            header_map(expected)
        );
    }

    #[rstest]
    #[case::bearer("Authorization", "Bearer abc", true)]
    #[case::lowercase("authorization", "bearer abc", true)]
    #[case::blank_token("Authorization", "Bearer    ", false)]
    #[case::empty("authorization", "", false)]
    #[case::basic("Authorization", "Basic abc", false)]
    #[case::different_header("x-api-key", "abc", false)]
    fn bearer_detection_requires_a_non_empty_token(
        #[case] name: &str,
        #[case] value: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(has_bearer_auth(&header_map(&[(name, value)])), expected);
    }

    #[test]
    fn bearer_detection_skips_a_value_that_is_not_visible_ascii() {
        let headers = HeaderMap::from_iter([(
            AUTHORIZATION,
            HeaderValue::from_bytes("Bearer invali\u{00e9}auth".as_bytes()).unwrap(),
        )]);
        assert!(!has_bearer_auth(&headers));
    }
}
