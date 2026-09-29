use std::{collections::BTreeMap, time::SystemTime};

use crate::{
    AwsAuthService, AwsCredentialSource, Error, aws_signature_headers, is_sigv4_computed_header,
    sign_request,
};
use aws_credential_types::Credentials;
use litellm_http::outbound::{RequestSigner, UnsignedRequest};

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
    fn sign(
        &self,
        request: UnsignedRequest<'_>,
    ) -> Result<Vec<(String, String)>, litellm_http::Error> {
        if let Some((name, _)) = request
            .headers
            .iter()
            .find(|(name, _)| is_sigv4_computed_header(name))
        {
            return Err(litellm_http::Error::ComputedHeader(name.clone()));
        }
        let headers: BTreeMap<String, String> = request.headers.iter().cloned().collect();
        let signature_headers = aws_signature_headers(&headers);
        let signable = aws_sigv4::http_request::SignableRequest::new(
            request.method.as_str(),
            request.url,
            signature_headers
                .iter()
                .map(|(name, value)| (name.as_str(), value.as_str())),
            aws_sigv4::http_request::SignableBody::Bytes(request.body),
        )
        .map_err(|error| litellm_http::Error::Signature(error.to_string()))?;
        sign_request(
            signable,
            &self.region,
            self.service,
            &self.credentials,
            (self.clock)(),
        )
        .map(|signature| signature.into_iter().collect())
        .map_err(|error| litellm_http::Error::Signature(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use crate::sign_post;
    use std::time::{Duration, UNIX_EPOCH};

    use litellm_http::outbound::OutboundRequest;
    use serde_json::{Value, json};

    use super::*;

    #[rstest::rstest]
    fn signatures_include_the_declared_method() {
        let signer = signer("textract");
        let url = litellm_core_utils::url_utils::ApiUrl::parse_exact(
            "https://signing.test/operation?tenant=a",
        )
        .unwrap();
        let build = |method| {
            OutboundRequest::endpoint_json(
                method,
                url.clone(),
                Vec::new(),
                &json!({}),
                None,
                Some(&signer),
            )
            .unwrap()
        };
        let post = build(reqwest::Method::POST);
        let delete = build(reqwest::Method::DELETE);
        assert_ne!(post.header("authorization"), delete.header("authorization"));
    }

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

    fn authorization(body: &Value, service: &'static str) -> String {
        OutboundRequest::signed_json(
            "https://textract.us-east-1.amazonaws.com/".into(),
            vec![("X-Amz-Target".into(), "Textract.DetectDocumentText".into())],
            body,
            None,
            &signer(service),
        )
        .unwrap()
        .header("Authorization")
        .unwrap()
        .to_string()
    }

    #[test]
    fn the_signature_verifies_against_the_bytes_that_are_sent() {
        let sent = OutboundRequest::signed_json(
            "https://textract.us-east-1.amazonaws.com/".into(),
            vec![("X-Amz-Target".into(), "Textract.DetectDocumentText".into())],
            &json!({"Document": {"Bytes": "aGk="}}),
            None,
            &signer("textract"),
        )
        .unwrap();
        let unsigned: BTreeMap<String, String> = sent
            .headers()
            .iter()
            .filter(|(name, _)| !is_sigv4_computed_header(name))
            .cloned()
            .collect();
        let recomputed = sign_post(
            sent.url(),
            sent.body(),
            &aws_signature_headers(&unsigned),
            "us-east-1",
            "textract",
            &Credentials::new("AKIDEXAMPLE", "secret", None, None, "test"),
            fixed_clock(),
        )
        .unwrap();

        assert_eq!(
            sent.header("Authorization"),
            Some(recomputed["Authorization"].as_str())
        );
    }

    #[test]
    fn the_signature_depends_on_the_body_and_the_service() {
        let original = authorization(&json!({"text": "card 4111"}), "textract");

        assert_ne!(
            original,
            authorization(&json!({"text": "card [REDACTED]"}), "textract")
        );
        assert_ne!(
            original,
            authorization(&json!({"text": "card 4111"}), "bedrock")
        );
        assert!(original.contains("/us-east-1/textract/aws4_request"));
    }

    #[test]
    fn a_forwarded_computed_header_is_refused_instead_of_sent_twice() {
        let error = OutboundRequest::signed_json(
            "https://textract.us-east-1.amazonaws.com/".into(),
            vec![("authorization".into(), "Bearer caller".into())],
            &json!({}),
            None,
            &signer("textract"),
        )
        .unwrap_err();

        assert_eq!(
            error,
            litellm_http::Error::ComputedHeader("authorization".into())
        );
    }
}
