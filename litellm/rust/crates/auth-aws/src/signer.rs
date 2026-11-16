use std::collections::BTreeMap;
use std::time::SystemTime;

use aws_credential_types::Credentials;
use aws_sigv4::sign::v4;
use chrono::{DateTime, Utc};
use reqwest::{
    Request,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use sha2::{Digest, Sha256};

use crate::{
    AwsAuthService, AwsCredentialSource, Error, SigningScope, SigningStage, aws_signature_headers,
    constants::AWS4_AUTHORIZATION_PREFIX, is_sigv4_computed_header, sign_headers,
};
use litellm_http::outbound::RequestSigner;

#[derive(Clone, Debug)]
pub struct SigV4Signer {
    region: String,
    service: &'static str,
    credentials: Credentials,
    clock: fn() -> SystemTime,
    caller_headers: CallerSigningHeaders,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CallerSigningHeaders {
    #[default]
    Dropped,
    Botocore,
}

impl SigV4Signer {
    pub fn new(region: String, service: &'static str, credentials: Credentials) -> Self {
        Self {
            region,
            service,
            credentials,
            clock: SystemTime::now,
            caller_headers: CallerSigningHeaders::Dropped,
        }
    }

    pub fn with_clock(self, clock: fn() -> SystemTime) -> Self {
        Self { clock, ..self }
    }

    pub fn with_caller_headers(self, caller_headers: CallerSigningHeaders) -> Self {
        Self {
            caller_headers,
            ..self
        }
    }

    fn keeps_caller_security_token(&self) -> bool {
        self.caller_headers == CallerSigningHeaders::Botocore
            && self.credentials.session_token().is_none()
    }

    pub async fn resolve(
        auth: &AwsAuthService,
        region: String,
        service: &'static str,
        credentials: AwsCredentialSource,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Self, Error> {
        let credentials = credentials
            .resolve(auth, &region, service, env_lookup)
            .await?;
        Ok(Self::new(region, service, credentials))
    }
}

const CALLER_HEADERS_THE_SIGNER_REPLACES: &[&str] = &["x-amz-date", "x-amz-security-token", "date"];

const CALLER_DATE_HEADERS: &[&str] = &["x-amz-date", "date"];

const SECURITY_TOKEN: &str = "x-amz-security-token";

const BOTOCORE_UNSIGNED_HEADERS: &[&str] = &["expect", "user-agent", "x-amzn-trace-id"];

fn signable(headers: &HeaderMap, keep_security_token: bool) -> HeaderMap {
    aws_signature_headers(headers)
        .iter()
        .filter(|(name, _)| {
            !is_sigv4_computed_header(name.as_str())
                || (keep_security_token && name.as_str() == SECURITY_TOKEN)
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn quote_path(path: &str) -> String {
    path.bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' | b'/' => {
                char::from(byte).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

fn canonical_uri(path: &str) -> String {
    if path.is_empty() {
        return "/".to_string();
    }
    let segments = path.split('/').fold(Vec::new(), |mut kept, segment| {
        match segment {
            "" | "." => {}
            ".." => {
                kept.pop();
            }
            other => kept.push(other),
        }
        kept
    });
    let leading = if path.starts_with('/') { "/" } else { "" };
    let trailing = if path.ends_with('/') && !segments.is_empty() {
        "/"
    } else {
        ""
    };
    quote_path(&format!("{leading}{}{trailing}", segments.join("/")))
}

fn canonical_query(query: Option<&str>) -> String {
    let mut pairs: Vec<(&str, &str)> = query
        .filter(|query| !query.is_empty())
        .map(|query| {
            query
                .split('&')
                .map(|pair| pair.split_once('=').unwrap_or((pair, "")))
                .collect()
        })
        .unwrap_or_default();
    pairs.sort_unstable();
    pairs
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("&")
}

fn canonical_host(url: &reqwest::Url) -> String {
    let host = url.host_str().unwrap_or_default();
    match url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_string(),
    }
}

fn header_pair(name: &str, value: &str) -> Result<(HeaderName, HeaderValue), Error> {
    Ok((
        HeaderName::from_bytes(name.as_bytes()).map_err(Error::signing(SigningStage::Output))?,
        HeaderValue::from_str(value).map_err(Error::signing(SigningStage::Output))?,
    ))
}

fn sign_with_date_header(
    request: &Request,
    body: &[u8],
    headers: &HeaderMap,
    scope: SigningScope<'_>,
) -> Result<HeaderMap, Error> {
    let time: DateTime<Utc> = scope.time.into();
    let amz_date = time.format("%Y%m%dT%H%M%SZ").to_string();
    let http_date = time.format("%a, %d %b %Y %H:%M:%S -0000").to_string();
    let session_token = scope.credentials.session_token();
    let mut canonical: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, value) in headers {
        if BOTOCORE_UNSIGNED_HEADERS.contains(&name.as_str()) {
            continue;
        }
        let value = value.to_str().map_err(|_| Error::NonAsciiHeader {
            name: name.to_string(),
        })?;
        canonical
            .entry(name.as_str().to_string())
            .or_default()
            .push(value.split_whitespace().collect::<Vec<_>>().join(" "));
    }
    canonical.insert("date".into(), vec![http_date.clone()]);
    if let Some(token) = session_token {
        canonical.insert(SECURITY_TOKEN.into(), vec![token.to_string()]);
    }
    canonical
        .entry("host".into())
        .or_insert_with(|| vec![canonical_host(request.url())]);
    let payload = headers
        .get("x-amz-content-sha256")
        .and_then(|value| value.to_str().ok())
        .map_or_else(|| hex_sha256(body), str::to_string);
    let canonical_headers: String = canonical
        .iter()
        .map(|(name, values)| format!("{name}:{}\n", values.join(",")))
        .collect();
    let signed_headers = canonical.keys().cloned().collect::<Vec<_>>().join(";");
    let canonical_request = format!(
        "{}\n{}\n{}\n{canonical_headers}\n{signed_headers}\n{payload}",
        request.method().as_str(),
        canonical_uri(request.url().path()),
        canonical_query(request.url().query()),
    );
    let credential_scope = format!(
        "{}/{}/{}/aws4_request",
        &amz_date[..8],
        scope.region,
        scope.service
    );
    let string_to_sign = format!(
        "{AWS4_AUTHORIZATION_PREFIX}\n{amz_date}\n{credential_scope}\n{}",
        hex_sha256(canonical_request.as_bytes())
    );
    let signing_key = v4::generate_signing_key(
        scope.credentials.secret_access_key(),
        scope.time,
        scope.region,
        scope.service,
    );
    let signature = v4::calculate_signature(signing_key, string_to_sign.as_bytes());
    let authorization = format!(
        "{AWS4_AUTHORIZATION_PREFIX} Credential={}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}",
        scope.credentials.access_key_id()
    );
    [
        Some(("date", http_date.as_str())),
        session_token.map(|token| (SECURITY_TOKEN, token)),
        Some(("authorization", authorization.as_str())),
    ]
    .into_iter()
    .flatten()
    .map(|(name, value)| header_pair(name, value))
    .collect()
}

fn overrides_the_signature(headers: &HeaderMap) -> bool {
    headers
        .get(reqwest::header::AUTHORIZATION)
        .is_some_and(|value| {
            !value
                .as_bytes()
                .starts_with(AWS4_AUTHORIZATION_PREFIX.as_bytes())
        })
}

impl RequestSigner for SigV4Signer {
    fn owned_headers(&self) -> &'static [&'static str] {
        if self.keeps_caller_security_token() {
            CALLER_DATE_HEADERS
        } else {
            CALLER_HEADERS_THE_SIGNER_REPLACES
        }
    }

    fn sign(&self, request: &Request) -> Result<HeaderMap, litellm_http::Error> {
        let body = match request.body() {
            None => &[],
            Some(body) => body
                .as_bytes()
                .ok_or_else(litellm_http::Error::unsignable_body)?,
        };
        let headers = signable(request.headers(), self.keeps_caller_security_token());
        let scope = SigningScope {
            region: &self.region,
            service: self.service,
            credentials: &self.credentials,
            time: (self.clock)(),
        };
        let signed = match self.caller_headers == CallerSigningHeaders::Botocore
            && request.headers().contains_key(reqwest::header::DATE)
        {
            true => sign_with_date_header(request, body, &headers, scope),
            false => sign_headers(
                request.method().as_str(),
                request.url().as_str(),
                body,
                &headers,
                scope,
            ),
        }
        .map_err(|error| match error {
            Error::NonAsciiHeader { .. } => litellm_http::Error::unsignable_header(error),
            Error::ExpiredCredentials
            | Error::Credentials { .. }
            | Error::WebIdentityTokenReference
            | Error::MissingWebIdentityToken
            | Error::WebIdentityToken(_)
            | Error::MissingWebIdentityCredentials
            | Error::WebIdentityExpiration(_)
            | Error::Signing { .. } => litellm_http::Error::signature(error),
        })?;
        Ok(match overrides_the_signature(request.headers()) {
            true => signed
                .iter()
                .filter(|(name, _)| *name != reqwest::header::AUTHORIZATION)
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
            false => signed,
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

    fn sent(headers: &[(&str, &str)], credentials: Credentials) -> Request {
        signed(
            json(
                URL,
                &headers
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect::<Vec<_>>(),
                &json_value!({}),
                None,
            )
            .unwrap(),
            &SigV4Signer::new("us-east-1".into(), "bedrock", credentials).with_clock(fixed_clock),
        )
        .unwrap()
    }

    fn header<'a>(request: &'a Request, name: &str) -> Vec<&'a str> {
        request
            .headers()
            .get_all(name)
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect()
    }

    #[rstest::rstest]
    #[case::caller_date("date", None)]
    #[case::caller_amz_date("x-amz-date", Some("20231114T221320Z"))]
    #[case::caller_security_token_without_a_session("x-amz-security-token", None)]
    fn a_header_the_signer_computes_is_dropped_from_the_callers_copy(
        #[case] name: &str,
        #[case] expected: Option<&str>,
    ) {
        let request = sent(
            &[(name, "caller-value")],
            Credentials::new("AKIDEXAMPLE", "secret", None, None, "test"),
        );

        assert_eq!(
            header(&request, name),
            expected.into_iter().collect::<Vec<_>>()
        );
        assert!(header(&request, "authorization")[0].starts_with("AWS4-HMAC-SHA256 "));
    }

    #[test]
    fn a_session_token_replaces_the_callers_security_token() {
        let request = sent(
            &[("X-Amz-Security-Token", "caller-value")],
            Credentials::new(
                "AKIDEXAMPLE",
                "secret",
                Some("session".into()),
                None,
                "test",
            ),
        );

        assert_eq!(header(&request, "x-amz-security-token"), ["session"]);
    }

    #[rstest::rstest]
    #[case::bearer_overrides_the_signature("Bearer gateway", "Bearer gateway")]
    #[case::a_stale_signature_is_replaced(
        "AWS4-HMAC-SHA256 Credential=stale",
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20231114/us-east-1/bedrock/aws4_request, SignedHeaders=content-type;host;x-amz-date"
    )]
    fn a_callers_authorization_overrides_the_signature_unless_it_is_sigv4(
        #[case] caller: &str,
        #[case] expected: &str,
    ) {
        let request = sent(
            &[("Authorization", caller)],
            Credentials::new("AKIDEXAMPLE", "secret", None, None, "test"),
        );
        let authorization = header(&request, "authorization");

        assert_eq!(authorization.len(), 1);
        assert_eq!(
            authorization[0].split(", Signature=").next(),
            Some(expected)
        );
        assert_eq!(header(&request, "x-amz-date"), ["20231114T221320Z"]);
    }

    const BEDROCK_URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com/model/us-west-2/anthropic.claude-3-5-sonnet-20240620-v1:0/invoke";
    const ARN_URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A1%3Ainference-profile%2Fx/invoke?b=2&a=1";
    const CALLER_SIGNING_HEADERS: &[(&str, &str)] = &[
        ("date", "Wed, 01 Jan 2020 00:00:00 GMT"),
        ("x-amz-date", "20200101T000000Z"),
        ("x-amz-security-token", "caller-token"),
    ];
    const CALLER_TOKEN: &[(&str, &str)] = &[("x-amz-security-token", "caller-token")];
    const CALLER_DATE: &[(&str, &str)] = &[("Date", "Wed, 01 Jan 2020 00:00:00 GMT")];
    const HTTP_DATE: &str = "Tue, 14 Nov 2023 22:13:20 -0000";
    const AMZ_DATE: &str = "20231114T221320Z";

    #[rstest::rstest]
    #[case::date_and_caller_token(URL, CALLER_SIGNING_HEADERS, None, (Some(HTTP_DATE), None, Some("caller-token")), "content-type;date;host;x-amz-security-token", "1b3652d523d5296b3904a80981e936a201167da9cb7bd200c2da8171dadb2c08")]
    #[case::caller_token_only(URL, CALLER_TOKEN, None, (None, Some(AMZ_DATE), Some("caller-token")), "content-type;host;x-amz-date;x-amz-security-token", "0fbd1a8e98efbfbcdb6fa9d665fe89ffffcdd8a7656865d63def21a5344d02dd")]
    #[case::date_only(URL, CALLER_DATE, None, (Some(HTTP_DATE), None, None), "content-type;date;host", "cccdac7b04f20ebfe37a156c4e806a14e6c0a4b0aad11ca67e25dca337237085")]
    #[case::session_replaces_the_caller_token_under_a_date(URL, CALLER_SIGNING_HEADERS, Some("session"), (Some(HTTP_DATE), None, Some("session")), "content-type;date;host;x-amz-security-token", "3406b8200cae95c8b66037dc5a9a838e1fe4da94b57b89fada3c05cde8a2c15c")]
    #[case::session_replaces_the_caller_token(URL, CALLER_TOKEN, Some("session"), (None, Some(AMZ_DATE), Some("session")), "content-type;host;x-amz-date;x-amz-security-token", "d75425952d9f5ef7465a6ef3acabf15908c0edfd8ce2529cc3f0f8d2e9232e76")]
    #[case::colon_in_the_model_path(BEDROCK_URL, CALLER_SIGNING_HEADERS, None, (Some(HTTP_DATE), None, Some("caller-token")), "content-type;date;host;x-amz-security-token", "47d75a0dcb00499a025fc0f9f010f6dce89c7661eb54cb97c3a293e9c585b153")]
    #[case::encoded_arn_and_query(ARN_URL, CALLER_SIGNING_HEADERS, None, (Some(HTTP_DATE), None, Some("caller-token")), "content-type;date;host;x-amz-security-token", "8fb7d97a0ad5df9d1c3bbfc55897266059a5a2f35a1aaed9687f370047e78363")]
    #[case::encoded_arn_and_query_without_a_date(ARN_URL, CALLER_TOKEN, None, (None, Some(AMZ_DATE), Some("caller-token")), "content-type;host;x-amz-date;x-amz-security-token", "f5eb993af90c7896203bf6b3b85c0087087f712a0634e44ac0a04c9d87d6fe12")]
    #[case::encoded_arn_and_query_with_a_session(ARN_URL, CALLER_DATE, Some("session"), (Some(HTTP_DATE), None, Some("session")), "content-type;date;host;x-amz-security-token", "15e8d5a9f64eb64366363f951db64de2a747959573a8cda9b5faa0fcbabc713c")]
    fn botocore_caller_headers_sign_like_botocore(
        #[case] url: &str,
        #[case] caller: &[(&str, &str)],
        #[case] session: Option<&str>,
        #[case] sent: (Option<&str>, Option<&str>, Option<&str>),
        #[case] signed_headers: &str,
        #[case] signature: &str,
    ) {
        let request = signed(
            json(
                url,
                &caller
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect::<Vec<_>>(),
                &json_value!({}),
                None,
            )
            .unwrap(),
            &SigV4Signer::new(
                "us-east-1".into(),
                "bedrock",
                Credentials::new(
                    "AKIDEXAMPLE",
                    "secret",
                    session.map(str::to_string),
                    None,
                    "test",
                ),
            )
            .with_clock(fixed_clock)
            .with_caller_headers(CallerSigningHeaders::Botocore),
        )
        .unwrap();

        assert_eq!(
            (
                header(&request, "date"),
                header(&request, "x-amz-date"),
                header(&request, "x-amz-security-token"),
                header(&request, "authorization"),
            ),
            (
                sent.0.into_iter().collect::<Vec<_>>(),
                sent.1.into_iter().collect::<Vec<_>>(),
                sent.2.into_iter().collect::<Vec<_>>(),
                vec![format!(
                    "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20231114/us-east-1/bedrock/aws4_request, SignedHeaders={signed_headers}, Signature={signature}"
                )
                .as_str()],
            )
        );
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
