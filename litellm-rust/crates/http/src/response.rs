use std::ops::Deref;

use http::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProviderResponse<T> {
    pub body: T,
    #[serde(skip)]
    pub headers: Vec<(String, String)>,
}

impl<T> ProviderResponse<T> {
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> ProviderResponse<U> {
        ProviderResponse {
            body: map(self.body),
            headers: self.headers,
        }
    }
}

impl<T> From<T> for ProviderResponse<T> {
    fn from(body: T) -> Self {
        Self {
            body,
            headers: Vec::new(),
        }
    }
}

impl<T> Deref for ProviderResponse<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.body
    }
}

pub fn forwarded_headers(headers: &[(String, String)]) -> HeaderMap {
    let connection = headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("connection"))
        .flat_map(|(_, value)| value.split(','))
        .map(str::trim)
        .collect::<Vec<_>>();
    let mut forwarded = HeaderMap::new();
    for (name, value) in headers {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        if connection
            .iter()
            .any(|token| name.as_str().eq_ignore_ascii_case(token))
            || name.as_str().starts_with("x-litellm-")
            || matches!(
                name.as_str(),
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
            )
        {
            continue;
        }
        if let Ok(value) = HeaderValue::from_str(value) {
            forwarded.append(name, value);
        }
    }
    forwarded
}

pub fn append_provider_headers(target: &mut HeaderMap, headers: &[(String, String)]) {
    let forwarded = forwarded_headers(headers);
    for name in forwarded.keys() {
        for value in forwarded.get_all(name) {
            target.append(name.clone(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    fn forwarding_preserves_duplicate_headers_and_recomputes_transport_headers() {
        let pairs = [
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
        ]
        .map(|(name, value)| (name.to_owned(), value.to_owned()));
        let mut target = HeaderMap::new();
        target.insert("content-type", HeaderValue::from_static("application/json"));
        append_provider_headers(&mut target, &pairs);
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
    fn cached_response_contains_only_the_body() {
        let response = ProviderResponse {
            body: json!({"result": "hello"}),
            headers: vec![("x-request-id".into(), "live-request".into())],
        };
        let stored = serde_json::to_value(&response).unwrap();
        assert_eq!(stored, response.body);
        let replay: ProviderResponse<serde_json::Value> = serde_json::from_value(stored).unwrap();
        assert_eq!(replay.body, response.body);
        assert!(replay.headers.is_empty());
    }
}
