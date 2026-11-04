use http::HeaderMap;
use http::header::{ETAG, IF_NONE_MATCH, RETRY_AFTER};

use crate::{Fetch, FetchRequest, Reply};

/// A catalog over HTTP. The client never follows a redirect, as main's `httpx.get` does not
#[derive(Clone, Debug)]
pub struct HttpFetch {
    client: litellm_http::Client,
    url: String,
}

impl HttpFetch {
    pub fn new(client: litellm_http::Client, url: String) -> Self {
        Self { client, url }
    }
}

impl Fetch for HttpFetch {
    fn location(&self) -> &str {
        &self.url
    }

    async fn fetch(&self, request: &FetchRequest) -> Reply {
        let builder = self.client.get(&self.url).timeout(request.timeout);
        let builder = match &request.if_none_match {
            Some(etag) => builder.header(IF_NONE_MATCH, etag),
            None => builder,
        };
        let response = match builder.send().await {
            Ok(response) => response,
            Err(error) if error.is_builder() => {
                return Reply::Refused {
                    error: litellm_http::Error::from(error).into(),
                };
            }
            Err(error) => {
                return Reply::Transport {
                    error: litellm_http::Error::from(error).into(),
                };
            }
        };
        let status = response.status().as_u16();
        let etag = joined(response.headers(), &ETAG);
        let retry_after = joined(response.headers(), &RETRY_AFTER);
        match response.bytes().await {
            Ok(body) => Reply::Response {
                status,
                etag,
                retry_after,
                body: body.to_vec(),
            },
            Err(error) => Reply::Transport {
                error: litellm_http::Error::from(error).into(),
            },
        }
    }
}

/// Every value of `name` joined with `", "`, as httpx reads a repeated header
fn joined(headers: &HeaderMap, name: &http::HeaderName) -> Option<String> {
    let values: Vec<String> = headers
        .get_all(name)
        .iter()
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned())
        .collect();
    (!values.is_empty()).then(|| values.join(", "))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rstest::rstest;
    use wiremock::matchers::{header, method};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn request(if_none_match: Option<&str>) -> FetchRequest {
        FetchRequest {
            if_none_match: if_none_match.map(str::to_owned),
            timeout: Duration::from_secs(5),
        }
    }

    #[rstest]
    #[tokio::test]
    async fn reads_status_headers_and_body_without_following_redirects() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("location", "/elsewhere")
                    .append_header("etag", "\"a\"")
                    .append_header("etag", "\"b\"")
                    .insert_header("retry-after", "3")
                    .set_body_string("moved"),
            )
            .mount(&server)
            .await;
        let fetch = HttpFetch::new(litellm_http::Client::for_test(), server.uri());
        let Reply::Response {
            status,
            etag,
            retry_after,
            body,
        } = fetch.fetch(&request(None)).await
        else {
            panic!("a response")
        };
        assert_eq!(status, 302);
        assert_eq!(etag.as_deref(), Some("\"a\", \"b\""));
        assert_eq!(retry_after.as_deref(), Some("3"));
        assert_eq!(body, b"moved");
    }

    #[rstest]
    #[tokio::test]
    async fn sends_if_none_match_when_asked() {
        let server = MockServer::start().await;
        Mock::given(header("if-none-match", "\"v1\""))
            .respond_with(ResponseTemplate::new(304))
            .mount(&server)
            .await;
        let fetch = HttpFetch::new(litellm_http::Client::for_test(), server.uri());
        let reply = fetch.fetch(&request(Some("\"v1\""))).await;
        assert!(matches!(reply, Reply::Response { status: 304, .. }));
    }

    #[rstest]
    #[case::refused_connection("http://127.0.0.1:1/map.json", false)]
    #[case::unparseable("http://[invalid/map.json", true)]
    #[tokio::test]
    async fn transport_failures_retry_and_builder_failures_are_final(
        #[case] url: &str,
        #[case] refused: bool,
    ) {
        let fetch = HttpFetch::new(litellm_http::Client::for_test(), url.to_owned());
        let reply = fetch.fetch(&request(None)).await;
        assert_eq!(matches!(reply, Reply::Refused { .. }), refused, "{reply:?}");
        assert_eq!(
            matches!(reply, Reply::Transport { .. }),
            !refused,
            "{reply:?}"
        );
    }
}
