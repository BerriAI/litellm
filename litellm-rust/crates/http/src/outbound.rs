//! Building the request a route hands to the transport. The body is serialized
//! once, into the [`reqwest::Request`] that is sent, and a [`RequestSigner`]
//! reads those exact bytes back out of it.
//!
//! Host hooks may rewrite the wire request (redaction, guardrails) and a
//! signature such as AWS SigV4 covers the body, so a route builds this after
//! its hooks ran and signs last.

use std::time::Duration;

use reqwest::{
    Request,
    header::{CONTENT_TYPE, HeaderMap},
};
use serde::Serialize;

use crate::Error;

/// Returns the headers to add to the request; it never sees a mutable request.
pub trait RequestSigner: Send + Sync {
    fn sign(&self, request: &Request) -> Result<HeaderMap, Error>;
}

pub fn json(
    url: &str,
    headers: &[(String, String)],
    body: &impl Serialize,
    timeout: Option<Duration>,
) -> Result<Request, Error> {
    let body = serde_json::to_vec(body).map_err(|error| Error::RequestBody(error.to_string()))?;
    let with_headers = headers
        .iter()
        .fold(http::Request::post(url), |builder, (name, value)| {
            builder.header(name, value)
        });
    let has_content_type = with_headers
        .headers_ref()
        .is_some_and(|headers| headers.contains_key(CONTENT_TYPE));
    let builder = if has_content_type {
        with_headers
    } else {
        with_headers.header(CONTENT_TYPE, "application/json")
    };
    let mut request = builder
        .body(body)
        .map_err(|error| Error::Request(error.to_string()))
        .and_then(|request| {
            Request::try_from(request)
                .map_err(|error| Error::Request(error.without_url().to_string()))
        })?;
    *request.timeout_mut() = timeout;
    Ok(request)
}

pub fn signed(mut request: Request, signer: &dyn RequestSigner) -> Result<Request, Error> {
    let signature = signer.sign(&request)?;
    request.headers_mut().extend(signature);
    Ok(request)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use reqwest::header::HeaderValue;
    use serde_json::json;

    use super::*;

    #[derive(Default)]
    struct Recording(Mutex<Vec<u8>>);

    impl RequestSigner for Recording {
        fn sign(&self, request: &Request) -> Result<HeaderMap, Error> {
            *self.0.lock().unwrap() = request.body().unwrap().as_bytes().unwrap().to_vec();
            Ok(HeaderMap::from_iter([(
                reqwest::header::AUTHORIZATION,
                HeaderValue::from_static("signed"),
            )]))
        }
    }

    #[test]
    fn the_signer_sees_exactly_the_bytes_that_are_sent() {
        let signer = Recording::default();
        let request = signed(
            json(
                "https://provider.test/",
                &[("x-caller".into(), "kept".into())],
                &json!({"b": 1, "a": [true, null]}),
                Some(Duration::from_secs(7)),
            )
            .unwrap(),
            &signer,
        )
        .unwrap();

        assert_eq!(
            request.body().unwrap().as_bytes().unwrap(),
            signer.0.lock().unwrap().as_slice()
        );
        assert_eq!(request.headers()["authorization"], "signed");
        assert_eq!(request.headers()["x-caller"], "kept");
        assert_eq!(request.timeout(), Some(&Duration::from_secs(7)));
    }

    #[test]
    fn the_content_type_is_part_of_what_the_signer_sees() {
        struct RequiresContentType;
        impl RequestSigner for RequiresContentType {
            fn sign(&self, request: &Request) -> Result<HeaderMap, Error> {
                request
                    .headers()
                    .contains_key(CONTENT_TYPE)
                    .then(HeaderMap::new)
                    .ok_or_else(|| Error::Signature("content-type was not signed".into()))
            }
        }

        let defaulted = signed(
            json("https://u.test/", &[], &json!({}), None).unwrap(),
            &RequiresContentType,
        )
        .unwrap();
        assert_eq!(defaulted.headers()[CONTENT_TYPE], "application/json");

        let provider = signed(
            json(
                "https://u.test/",
                &[("Content-Type".into(), "application/x-amz-json-1.1".into())],
                &json!({}),
                None,
            )
            .unwrap(),
            &RequiresContentType,
        )
        .unwrap();
        assert_eq!(
            provider.headers()[CONTENT_TYPE],
            "application/x-amz-json-1.1"
        );
        assert_eq!(provider.headers().len(), 1);
    }

    #[test]
    fn a_signer_failure_produces_no_request() {
        struct Refuses;
        impl RequestSigner for Refuses {
            fn sign(&self, _request: &Request) -> Result<HeaderMap, Error> {
                Err(Error::ComputedHeader("authorization".into()))
            }
        }

        let error = signed(
            json("https://u.test/", &[], &json!({}), None).unwrap(),
            &Refuses,
        )
        .unwrap_err();
        assert_eq!(error, Error::ComputedHeader("authorization".into()));
    }

    #[test]
    fn a_request_that_cannot_be_sent_fails_when_it_is_built() {
        assert!(matches!(
            json("not a url", &[], &json!({}), None),
            Err(Error::Request(_))
        ));
        assert!(matches!(
            json(
                "https://u.test/",
                &[("x-bad".into(), "\u{7f}".into())],
                &json!({}),
                None
            ),
            Err(Error::Request(_))
        ));
    }
}
