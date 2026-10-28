use std::time::SystemTime;

use aws_credential_types::Credentials;
use reqwest::{Request, header::HeaderMap};

use crate::{
    AwsAuthService, AwsCredentialSource, Error, SigningScope, aws_signature_headers,
    is_sigv4_computed_header, sign_headers,
};
use litellm_http::outbound::RequestSigner;

#[derive(Clone, Debug)]
pub struct SigV4Signer {
    region: String,
    service: &'static str,
    credentials: Credentials,
    clock: fn() -> SystemTime,
}

impl SigV4Signer {
    pub fn new(region: String, service: &'static str, credentials: Credentials) -> Self {
        Self {
            region,
            service,
            credentials,
            clock: SystemTime::now,
        }
    }

    pub fn with_clock(self, clock: fn() -> SystemTime) -> Self {
        Self { clock, ..self }
    }

    pub async fn resolve(
        auth: &AwsAuthService,
        region: String,
        service: &'static str,
        credentials: AwsCredentialSource,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Self, Error> {
        Ok(Self::new(
            region,
            service,
            credentials.resolve(auth, env_lookup).await?,
        ))
    }
}

impl RequestSigner for SigV4Signer {
    fn sign(&self, request: &Request) -> Result<HeaderMap, litellm_http::Error> {
        if let Some(name) = request
            .headers()
            .keys()
            .find(|name| is_sigv4_computed_header(name.as_str()))
        {
            return Err(litellm_http::Error::computed_header(name.as_str()));
        }
        let body = match request.body() {
            None => &[],
            Some(body) => body
                .as_bytes()
                .ok_or_else(litellm_http::Error::unsignable_body)?,
        };
        sign_headers(
            request.method().as_str(),
            request.url().as_str(),
            body,
            &aws_signature_headers(request.headers()),
            SigningScope {
                region: &self.region,
                service: self.service,
                credentials: &self.credentials,
                time: (self.clock)(),
            },
        )
        .map_err(|error| match error {
            Error::NonAsciiHeader { .. } => litellm_http::Error::unsignable_header(error),
            Error::Credentials { .. }
            | Error::MissingWebIdentityCredentials
            | Error::WebIdentityExpiration(_)
            | Error::Signing { .. } => litellm_http::Error::signature(error),
        })
    }
}

#[cfg(test)]
mod tests {
    use reqwest::{Body, Method, Url};
    use serde_json::{Value, json as json_value};
    use std::time::{Duration, UNIX_EPOCH};

    use litellm_http::outbound::{json, signed};

    use super::*;

    const URL: &str = "https://textract.us-east-1.amazonaws.com/";

    fn fixed_clock() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    fn signer(service: &'static str) -> SigV4Signer {
        SigV4Signer::new(
            "us-east-1".into(),
            service,
            Credentials::new("AKIDEXAMPLE", "secret", None, None, "test"),
        )
        .with_clock(fixed_clock)
    }

    fn target() -> (String, String) {
        ("X-Amz-Target".into(), "Textract.DetectDocumentText".into())
    }

    fn authorization(body: &Value, service: &'static str) -> String {
        signed(
            json(URL, &[target()], body, None).unwrap(),
            &signer(service),
        )
        .unwrap()
        .headers()["authorization"]
            .to_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn the_signature_verifies_against_the_bytes_that_are_sent() {
        let sent = signed(
            json(
                URL,
                &[target()],
                &json_value!({"Document": {"Bytes": "aGk="}}),
                None,
            )
            .unwrap(),
            &signer("textract"),
        )
        .unwrap();
        let unsigned: HeaderMap = sent
            .headers()
            .iter()
            .filter(|(name, _)| !is_sigv4_computed_header(name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        let recomputed = sign_headers(
            "POST",
            sent.url().as_str(),
            sent.body().unwrap().as_bytes().unwrap(),
            &aws_signature_headers(&unsigned),
            SigningScope {
                region: "us-east-1",
                service: "textract",
                credentials: &Credentials::new("AKIDEXAMPLE", "secret", None, None, "test"),
                time: fixed_clock(),
            },
        )
        .unwrap();

        assert_eq!(sent.headers()["authorization"], recomputed["authorization"]);
    }

    #[test]
    fn the_signature_depends_on_the_body_and_the_service() {
        let original = authorization(&json_value!({"text": "card 4111"}), "textract");

        assert_ne!(
            original,
            authorization(&json_value!({"text": "card [REDACTED]"}), "textract")
        );
        assert_ne!(
            original,
            authorization(&json_value!({"text": "card 4111"}), "bedrock")
        );
        assert!(original.contains("/us-east-1/textract/aws4_request"));
    }

    #[test]
    fn a_forwarded_computed_header_is_refused_instead_of_sent_twice() {
        let error = signed(
            json(
                URL,
                &[("authorization".into(), "Bearer caller".into())],
                &json_value!({}),
                None,
            )
            .unwrap(),
            &signer("textract"),
        )
        .unwrap_err();

        assert!(error.is_computed_header());
        assert!(error.to_string().contains("authorization"), "{error}");
    }

    #[test]
    fn a_forwarded_header_that_cannot_be_signed_is_the_callers_fault() {
        let error = signed(
            json(
                URL,
                &[("x-amz-meta-note".into(), "café".into())],
                &json_value!({}),
                None,
            )
            .unwrap(),
            &signer("textract"),
        )
        .unwrap_err();

        let report = litellm_error::Report::new(&error).to_string();
        assert!(report.contains("x-amz-meta-note"), "{report}");
        assert!(error.is_request());
        assert!(!error.dispatched());
    }

    #[test]
    fn a_streaming_body_cannot_be_signed() {
        let mut request = Request::new(Method::POST, Url::parse(URL).unwrap());
        *request.body_mut() = Some(Body::wrap_stream(futures_util::stream::once(async {
            Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"{}"))
        })));

        assert!(matches!(
            signed(request, &signer("textract")),
            Err(error) if !error.dispatched() && !error.is_request()
        ));
    }
}
