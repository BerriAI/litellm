use std::borrow::Cow;

use bytes::Bytes;
use http::{HeaderMap, HeaderName, StatusCode};

/// Max characters of an upstream error body echoed across the call boundary
/// before truncation, so provider bodies are bounded and data-minimized.
const UPSTREAM_ERROR_BODY_MAX_CHARS: usize = 256;

fn truncate_error_body(body: &str) -> String {
    if body.chars().count() <= UPSTREAM_ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(UPSTREAM_ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
}

/// A reply the provider sent with a non-success status. It is an error to the route and a
/// response to the caller, so it keeps the whole reply, boxed to keep error enums small.
#[derive(Clone, Debug)]
pub struct Rejected(Box<http::Response<Bytes>>);

impl Rejected {
    pub fn new(response: http::Response<Bytes>) -> Self {
        Self(Box::new(response))
    }

    pub fn status(&self) -> StatusCode {
        self.0.status()
    }

    pub fn headers(&self) -> &HeaderMap {
        self.0.headers()
    }

    pub fn body(&self) -> &Bytes {
        self.0.body()
    }

    pub fn text(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(self.0.body())
    }

    /// Bounds the echoed body for the surfaces that report it as a message.
    pub fn truncated(self) -> Self {
        let truncated = truncate_error_body(&self.text());
        self.with_text(truncated)
    }

    pub fn with_text(self, text: String) -> Self {
        Self::new(self.0.map(|_| Bytes::from(text)))
    }

    pub fn into_response(self) -> http::Response<Bytes> {
        *self.0
    }
}

impl PartialEq for Rejected {
    fn eq(&self, other: &Self) -> bool {
        self.status() == other.status()
            && self.headers() == other.headers()
            && self.body() == other.body()
    }
}

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "upstream request failed with status {}: {}",
            self.status().as_u16(),
            self.text()
        )
    }
}

impl std::error::Error for Rejected {}

impl From<http::Response<Bytes>> for Rejected {
    fn from(response: http::Response<Bytes>) -> Self {
        Self::new(response)
    }
}

/// Headers we never relay from a provider: hop-by-hop fields, body framing the gateway
/// recomputes, provider cookies, and markers the gateway owns.
fn is_forwarded(name: &HeaderName, connection_tokens: &[&str]) -> bool {
    let name = name.as_str();
    !(connection_tokens
        .iter()
        .any(|token| name.eq_ignore_ascii_case(token))
        || name.starts_with("x-litellm-")
        || matches!(
            name,
            "connection"
                | "keep-alive"
                | "proxy-authenticate"
                | "proxy-authorization"
                | "te"
                | "trailer"
                | "transfer-encoding"
                | "upgrade"
                | "content-length"
                | "content-type"
                | "content-encoding"
                | "set-cookie"
                | "set-cookie2"
        ))
}

/// The subset of a provider's headers a caller may see, with duplicates kept in order.
pub fn forwarded(headers: &HeaderMap) -> HeaderMap {
    let connection_tokens = headers
        .get_all("connection")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .collect::<Vec<_>>();
    headers
        .iter()
        .filter(|(name, _)| is_forwarded(name, &connection_tokens))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

pub fn forward(target: &mut HeaderMap, headers: &HeaderMap) {
    target.extend(forwarded(headers));
}

/// `(name, value)` pairs for hosts without an `http` type, such as Python. Values that are
/// not UTF-8 are dropped.
pub fn header_pairs(headers: &HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_string())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;
    use rstest::rstest;

    fn header_map(pairs: &[(&str, &str)]) -> HeaderMap {
        pairs
            .iter()
            .map(|(name, value)| {
                (
                    HeaderName::from_bytes(name.as_bytes()).unwrap(),
                    HeaderValue::from_str(value).unwrap(),
                )
            })
            .collect()
    }

    #[rstest]
    fn forwarding_preserves_duplicate_headers_and_recomputes_transport_headers() {
        let provider = header_map(&[
            ("X-Provider-Trace", "first"),
            ("x-provider-trace", "second"),
            ("Retry-After", "7"),
            ("Connection", "x-private, Keep-Alive"),
            ("X-Private", "hop"),
            ("Keep-Alive", "timeout=5"),
            ("Content-Type", "application/vendor+json"),
            ("Content-Length", "999"),
            ("Content-Encoding", "gzip"),
            ("Transfer-Encoding", "chunked"),
            ("Set-Cookie", "secret=value"),
            ("X-LiteLLM-Rust", "false"),
        ]);
        let mut target = header_map(&[("content-type", "application/json")]);
        forward(&mut target, &provider);
        assert_eq!(
            target
                .get_all("x-provider-trace")
                .iter()
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert_eq!(target["retry-after"], "7");
        assert_eq!(target["content-type"], "application/json");
        assert_eq!(target.len(), 4);
    }

    #[rstest]
    fn header_pairs_keep_order_and_duplicates() {
        let headers = header_map(&[("a", "1"), ("b", "2"), ("a", "3")]);
        assert_eq!(
            header_pairs(&headers),
            [("a", "1"), ("a", "3"), ("b", "2")].map(|(n, v)| (n.to_owned(), v.to_owned()))
        );
    }

    #[rstest]
    #[case::short_body_is_untouched("short".to_owned(), "short".to_owned())]
    #[case::long_body_is_bounded_by_characters(
        "\u{00e9}".repeat(UPSTREAM_ERROR_BODY_MAX_CHARS + 10),
        format!("{}... (truncated)", "\u{00e9}".repeat(UPSTREAM_ERROR_BODY_MAX_CHARS))
    )]
    fn truncation_bounds_the_body_by_characters(#[case] body: String, #[case] expected: String) {
        assert_eq!(truncate_error_body(&body), expected);
    }

    #[rstest]
    fn rejected_reports_status_and_a_bounded_body() {
        let long = "x".repeat(400);
        let rejected = Rejected::new(
            http::Response::builder()
                .status(429)
                .header("retry-after", "7")
                .body(Bytes::from(long.clone()))
                .unwrap(),
        );
        assert_eq!(rejected.text(), long);
        let truncated = rejected.clone().truncated();
        assert!(truncated.text().ends_with("... (truncated)"));
        assert_eq!(truncated.status().as_u16(), 429);
        assert_eq!(truncated.headers()["retry-after"], "7");
        assert!(
            truncated
                .to_string()
                .starts_with("upstream request failed with status 429: x")
        );
        assert_ne!(truncated, rejected);
    }
}
