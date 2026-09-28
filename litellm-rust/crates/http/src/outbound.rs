//! The request a route hands to the transport. The body is serialized once,
//! when the request is built, and a [`RequestSigner`] sees those exact bytes.
//!
//! Host hooks may rewrite the wire request (redaction, guardrails) and a
//! signature such as AWS SigV4 covers the body, so a route builds this after
//! its hooks ran and cannot change or re-serialize it afterwards.

use std::time::Duration;

use serde::Serialize;

use crate::{
    Error,
    request::{HeaderPolicy, has_header, with_headers},
};

#[derive(Clone, Copy, Debug)]
pub struct UnsignedRequest<'a> {
    pub url: &'a str,
    pub headers: &'a [(String, String)],
    pub body: &'a [u8],
}

/// Returns the headers to add to the request; it never sees a mutable request.
pub trait RequestSigner: Send + Sync {
    fn sign(&self, request: UnsignedRequest<'_>) -> Result<Vec<(String, String)>, Error>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutboundRequest {
    url: url::Url,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    timeout: Option<Duration>,
}

impl OutboundRequest {
    pub fn json(
        url: url::Url,
        headers: Vec<(String, String)>,
        body: &impl Serialize,
        timeout: Option<Duration>,
    ) -> Result<Self, Error> {
        Self::build(url, headers, body, timeout, None)
    }

    pub fn signed_json(
        url: url::Url,
        headers: Vec<(String, String)>,
        body: &impl Serialize,
        timeout: Option<Duration>,
        signer: &dyn RequestSigner,
    ) -> Result<Self, Error> {
        Self::build(url, headers, body, timeout, Some(signer))
    }

    fn build(
        url: url::Url,
        headers: Vec<(String, String)>,
        body: &impl Serialize,
        timeout: Option<Duration>,
        signer: Option<&dyn RequestSigner>,
    ) -> Result<Self, Error> {
        if !matches!(url.scheme(), "http" | "https") || url.host().is_none() {
            return Err(Error::UrlScheme(url.scheme().into()));
        }
        let mut url = url;
        url.set_fragment(None);
        let body =
            serde_json::to_vec(body).map_err(|error| Error::RequestBody(error.to_string()))?;
        let content_type = (!has_header(&headers, "content-type"))
            .then(|| ("content-type".to_string(), "application/json".to_string()));
        let unsigned: Vec<(String, String)> = headers.into_iter().chain(content_type).collect();
        let signature = signer
            .map(|signer| {
                signer.sign(UnsignedRequest {
                    url: url.as_str(),
                    headers: &unsigned,
                    body: &body,
                })
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            url,
            headers: unsigned.into_iter().chain(signature).collect(),
            body,
            timeout,
        })
    }

    pub fn url(&self) -> &str {
        self.url.as_str()
    }

    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    pub fn body(&self) -> &[u8] {
        &self.body
    }

    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    pub async fn send(self, client: &crate::Client) -> Result<reqwest::Response, reqwest::Error> {
        let builder = with_headers(
            client.post(self.url).body(self.body),
            &self.headers,
            HeaderPolicy::All,
        );
        match self.timeout {
            Some(timeout) => builder.timeout(timeout),
            None => builder,
        }
        .send()
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;

    #[derive(Default)]
    struct Recording(Mutex<Vec<u8>>);

    impl RequestSigner for Recording {
        fn sign(&self, request: UnsignedRequest<'_>) -> Result<Vec<(String, String)>, Error> {
            *self.0.lock().unwrap() = request.body.to_vec();
            Ok(vec![("authorization".into(), "signed".into())])
        }
    }

    #[rstest::rstest]
    fn the_signer_sees_exactly_the_bytes_that_are_sent() {
        let signer = Recording::default();
        let request = OutboundRequest::signed_json(
            url::Url::parse("https://provider.test/").unwrap(),
            vec![("x-caller".into(), "kept".into())],
            &json!({"b": 1, "a": [true, null]}),
            None,
            &signer,
        )
        .unwrap();

        assert_eq!(request.body(), signer.0.lock().unwrap().as_slice());
        assert_eq!(request.header("authorization"), Some("signed"));
        assert_eq!(request.header("x-caller"), Some("kept"));
    }

    #[test]
    fn the_content_type_is_part_of_what_the_signer_sees() {
        struct RequiresContentType;
        impl RequestSigner for RequiresContentType {
            fn sign(&self, request: UnsignedRequest<'_>) -> Result<Vec<(String, String)>, Error> {
                has_header(request.headers, "content-type")
                    .then(Vec::new)
                    .ok_or_else(|| Error::Signature("content-type was not signed".into()))
            }
        }

        let defaulted = OutboundRequest::signed_json(
            url::Url::parse("https://provider.test/").unwrap(),
            Vec::new(),
            &json!({}),
            None,
            &RequiresContentType,
        )
        .unwrap();
        assert_eq!(defaulted.header("content-type"), Some("application/json"));

        let provider = OutboundRequest::signed_json(
            url::Url::parse("https://provider.test/").unwrap(),
            vec![("Content-Type".into(), "application/x-amz-json-1.1".into())],
            &json!({}),
            None,
            &RequiresContentType,
        )
        .unwrap();
        assert_eq!(
            provider.header("content-type"),
            Some("application/x-amz-json-1.1")
        );
        assert_eq!(provider.headers().len(), 1);
    }

    #[test]
    fn a_signer_failure_produces_no_request() {
        struct Refuses;
        impl RequestSigner for Refuses {
            fn sign(&self, _request: UnsignedRequest<'_>) -> Result<Vec<(String, String)>, Error> {
                Err(Error::ComputedHeader("authorization".into()))
            }
        }

        assert_eq!(
            OutboundRequest::signed_json(
                url::Url::parse("https://provider.test/").unwrap(),
                Vec::new(),
                &json!({}),
                None,
                &Refuses
            ),
            Err(Error::ComputedHeader("authorization".into()))
        );
    }
    #[rstest::rstest]
    #[tokio::test]
    async fn signed_url_equals_the_sent_url() {
        struct RecordUrl(Mutex<Option<String>>);
        impl RequestSigner for RecordUrl {
            fn sign(&self, request: UnsignedRequest<'_>) -> Result<Vec<(String, String)>, Error> {
                *self.0.lock().unwrap() = Some(request.url.into());
                Ok(Vec::new())
            }
        }
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::any())
            .respond_with(wiremock::ResponseTemplate::new(200))
            .mount(&server)
            .await;
        let signer = RecordUrl(Mutex::new(None));
        let url = url::Url::parse(&format!(
            "{}/prefix/a%2Fb?tenant=a%20b#fragment",
            server.uri()
        ))
        .unwrap();
        OutboundRequest::signed_json(url, Vec::new(), &json!({}), None, &signer)
            .unwrap()
            .send(&crate::Client::plain_for_test())
            .await
            .unwrap();
        let requests = server.received_requests().await.unwrap();
        let sent = &requests[0];
        let sent_url = format!(
            "http://{}{}",
            sent.headers["host"].to_str().unwrap(),
            &sent.url[url::Position::BeforePath..]
        );
        assert_eq!(signer.0.into_inner().unwrap().unwrap(), sent_url);
    }
}
