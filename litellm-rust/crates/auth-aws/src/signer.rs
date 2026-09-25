use std::{collections::BTreeMap, time::SystemTime};

use aws_credential_types::Credentials;
use litellm_http::outbound::{RequestSigner, UnsignedRequest};
use serde_json::{Map, Value};

use crate::{
    Error, aws_auth_config, aws_signature_headers, host_supplied_credentials,
    is_sigv4_computed_header, resolve_credentials, sign_post,
};

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
        region: String,
        service: &'static str,
        optional_params: &Map<String, Value>,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Self, Error> {
        let credentials = match host_supplied_credentials(optional_params) {
            Some(credentials) => credentials,
            None => {
                resolve_credentials(aws_auth_config(optional_params, env_lookup), env_lookup)
                    .await?
            }
        };
        Ok(Self::new(region, service, credentials))
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
        sign_post(
            request.url,
            request.body,
            &aws_signature_headers(&headers),
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
    use std::time::{Duration, UNIX_EPOCH};

    use litellm_http::outbound::OutboundRequest;
    use serde_json::json;

    use super::*;

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
