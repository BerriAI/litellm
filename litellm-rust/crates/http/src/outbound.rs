//! The request a route hands to the transport. The body is serialized once,
//! when the request is built, and a [`RequestSigner`] sees those exact bytes.
//!
//! Host hooks may rewrite the wire request (redaction, guardrails) and a
//! signature such as AWS SigV4 covers the body, so a route builds this after
//! its hooks ran and cannot change or re-serialize it afterwards.

use litellm_core_utils::url_utils::{ApiUrl, Complete};
use reqwest::Method;
use std::time::Duration;

use serde::Serialize;

use crate::{
    Error,
    request::{HeaderPolicy, has_header, with_headers},
};

#[derive(Clone, Copy, Debug)]
pub struct UnsignedRequest<'a> {
    pub url: &'a str,
    pub method: &'a Method,
    pub headers: &'a [(String, String)],
    pub body: &'a [u8],
}

/// Returns the headers to add to the request; it never sees a mutable request.
pub trait RequestSigner: Send + Sync {
    fn sign(&self, request: UnsignedRequest<'_>) -> Result<Vec<(String, String)>, Error>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutboundRequest {
    url: ApiUrl<Complete>,
    method: Method,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    timeout: Option<Duration>,
}

impl OutboundRequest {
    pub fn json(
        url: String,
        headers: Vec<(String, String)>,
        body: &impl Serialize,
        timeout: Option<Duration>,
    ) -> Result<Self, Error> {
        Self::build(url, headers, body, timeout, None)
    }

    pub fn signed_json(
        url: String,
        headers: Vec<(String, String)>,
        body: &impl Serialize,
        timeout: Option<Duration>,
        signer: &dyn RequestSigner,
    ) -> Result<Self, Error> {
        Self::build(url, headers, body, timeout, Some(signer))
    }

    fn build(
        url: String,
        headers: Vec<(String, String)>,
        body: &impl Serialize,
        timeout: Option<Duration>,
        signer: Option<&dyn RequestSigner>,
    ) -> Result<Self, Error> {
        Self::endpoint_json(
            Method::POST,
            ApiUrl::parse_exact(&url)?,
            headers,
            body,
            timeout,
            signer,
        )
    }

    pub fn endpoint_json(
        method: Method,
        url: ApiUrl<Complete>,
        headers: Vec<(String, String)>,
        body: &impl Serialize,
        timeout: Option<Duration>,
        signer: Option<&dyn RequestSigner>,
    ) -> Result<Self, Error> {
        let body =
            serde_json::to_vec(body).map_err(|error| Error::RequestBody(error.to_string()))?;
        let content_type = (!has_header(&headers, "content-type"))
            .then(|| ("content-type".to_string(), "application/json".to_string()));
        let unsigned: Vec<(String, String)> = headers.into_iter().chain(content_type).collect();
        let signature = signer
            .map(|signer| {
                signer.sign(UnsignedRequest {
                    url: url.as_url().as_str(),
                    method: &method,
                    headers: &unsigned,
                    body: &body,
                })
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            url,
            method,
            headers: unsigned.into_iter().chain(signature).collect(),
            body,
            timeout,
        })
    }

    pub fn url(&self) -> &str {
        self.url.as_url().as_str()
    }

    pub fn destination(&self) -> &ApiUrl<Complete> {
        &self.url
    }

    pub fn method(&self) -> &Method {
        &self.method
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
            client
                .request(self.method, self.url.into_url())
                .body(self.body),
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

    #[rstest::rstest]
    #[case::post(Method::POST)]
    #[case::delete(Method::DELETE)]
    #[tokio::test]
    async fn transport_and_signer_use_declared_method_and_destination(#[case] method: Method) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        struct Checks {
            method: Method,
            url: String,
        }
        impl RequestSigner for Checks {
            fn sign(&self, request: UnsignedRequest<'_>) -> Result<Vec<(String, String)>, Error> {
                assert_eq!(request.method, &self.method);
                assert_eq!(request.url, self.url);
                Ok(Vec::new())
            }
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 4096];
            let mut bytes = Vec::new();
            loop {
                let count = socket.read(&mut buffer).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
                if bytes.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .await
                .unwrap();
            String::from_utf8(bytes)
                .unwrap()
                .lines()
                .next()
                .unwrap()
                .to_owned()
        });
        let url = format!("http://{address}/team%2Fname/route?sig=a%2fb&key=1&key=2");
        let signer = Checks {
            method: method.clone(),
            url: url.clone(),
        };
        let request = OutboundRequest::endpoint_json(
            method.clone(),
            ApiUrl::parse_exact(&url).unwrap(),
            Vec::new(),
            &json!({}),
            None,
            Some(&signer),
        )
        .unwrap();
        request
            .send(&crate::Client::plain_for_test())
            .await
            .unwrap();
        assert_eq!(
            server.await.unwrap(),
            format!("{method} /team%2Fname/route?sig=a%2fb&key=1&key=2 HTTP/1.1")
        );
    }

    #[derive(Default)]
    struct Recording(Mutex<Vec<u8>>);

    impl RequestSigner for Recording {
        fn sign(&self, request: UnsignedRequest<'_>) -> Result<Vec<(String, String)>, Error> {
            *self.0.lock().unwrap() = request.body.to_vec();
            Ok(vec![("authorization".into(), "signed".into())])
        }
    }

    #[test]
    fn the_signer_sees_exactly_the_bytes_that_are_sent() {
        let signer = Recording::default();
        let request = OutboundRequest::signed_json(
            "https://provider.test/".into(),
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
            "https://provider.test/".into(),
            Vec::new(),
            &json!({}),
            None,
            &RequiresContentType,
        )
        .unwrap();
        assert_eq!(defaulted.header("content-type"), Some("application/json"));

        let provider = OutboundRequest::signed_json(
            "https://provider.test/".into(),
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
                "https://provider.test/".into(),
                Vec::new(),
                &json!({}),
                None,
                &Refuses
            ),
            Err(Error::ComputedHeader("authorization".into()))
        );
    }
}
