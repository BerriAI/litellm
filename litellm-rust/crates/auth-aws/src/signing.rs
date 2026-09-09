use std::collections::BTreeMap;
use std::time::SystemTime;

use aws_sigv4::http_request::{
    SignableBody, SignableRequest, SigningParams, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;

use crate::{Credentials, Error};

pub struct SigV4Request<'a> {
    pub method: &'a str,
    pub uri: &'a str,
    pub body: &'a [u8],
    pub headers: &'a BTreeMap<String, String>,
    pub region: &'a str,
    pub service: &'a str,
    pub signing_time: SystemTime,
}

pub fn sign_v4(
    request: SigV4Request<'_>,
    credentials: &Credentials,
) -> Result<BTreeMap<String, String>, Error> {
    let identity: Identity = credentials.0.clone().into();
    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(request.region)
        .name(request.service)
        .time(request.signing_time)
        .settings(SigningSettings::default())
        .build()
        .map(SigningParams::from)
        .map_err(|error| Error::SigningParameters(error.to_string()))?;
    let header_refs = request
        .headers
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()));
    let signable = SignableRequest::new(
        request.method,
        request.uri,
        header_refs,
        SignableBody::Bytes(request.body),
    )
    .map_err(|error| Error::SignableRequest(error.to_string()))?;
    let (instructions, _) = sign(signable, &params)
        .map_err(|error| Error::RequestSigning(error.to_string()))?
        .into_parts();
    Ok(instructions
        .headers()
        .map(|(name, value)| {
            let normalized_name = match name {
                "authorization" => "Authorization",
                "x-amz-date" => "X-Amz-Date",
                "x-amz-security-token" => "X-Amz-Security-Token",
                _ => name,
            };
            (normalized_name.to_string(), value.to_string())
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::static_credentials;

    fn signing_request<'a>(
        uri: &'a str,
        body: &'a [u8],
        headers: &'a BTreeMap<String, String>,
        service: &'a str,
    ) -> SigV4Request<'a> {
        SigV4Request {
            method: "POST",
            uri,
            body,
            headers,
            region: "us-east-1",
            service,
            signing_time: SystemTime::UNIX_EPOCH + Duration::from_secs(1_704_164_645),
        }
    }

    #[test]
    fn signing_matches_the_bedrock_golden_vector() {
        let uri = "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke";
        let body = br#"{"input":"hello"}"#;
        let headers = BTreeMap::from([("Content-Type".into(), "application/json".into())]);
        let credentials = Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            Some("session-token".into()),
            None,
            "test",
        );
        let signed = sign_v4(
            signing_request(uri, body, &headers, "bedrock"),
            &credentials,
        )
        .expect("signature");
        assert_eq!(
            signed.get("X-Amz-Date").map(String::as_str),
            Some("20240102T030405Z")
        );
        assert_eq!(
            signed.get("Authorization").map(String::as_str),
            Some(
                "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20240102/us-east-1/bedrock/aws4_request, SignedHeaders=content-type;host;x-amz-date;x-amz-security-token, Signature=55c027ef47527d3ad63f1735f9d099efdbc99f296ff914bd94e727e24ec0e464"
            )
        );
    }

    #[test]
    fn signer_is_generic_over_method_and_service() {
        let uri = "https://sts.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15";
        let headers = BTreeMap::new();
        let credentials = static_credentials("AKIDEXAMPLE", "secret");
        let request = SigV4Request {
            method: "GET",
            uri,
            body: b"",
            headers: &headers,
            region: "us-east-1",
            service: "sts",
            signing_time: SystemTime::UNIX_EPOCH,
        };
        let signed = sign_v4(request, &credentials).expect("signature");
        assert!(signed["Authorization"].contains("/sts/aws4_request"));
    }
}
