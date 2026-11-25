//! LiteLLM's explicit AWS login for an external account: workload identity federation with AWS
//! as the identity provider, Python's `VertexAIAwsWifAuth.credentials_from_explicit_aws`. AWS
//! vouches for the caller through a GetCallerIdentity request signed with the AWS credentials
//! LiteLLM resolves from the file's own `aws_*` keys. That signed request, serialized as the
//! subject token, is what Google's Rust SDK exchanges at Google's STS for a Google access token.

use std::{sync::Arc, time::SystemTime};

use google_cloud_auth::credentials::{
    external_account::ProgrammaticBuilder,
    subject_token::{self, SubjectToken, SubjectTokenProvider},
};
use http::{HeaderMap, HeaderValue, header::AUTHORIZATION};
use litellm_auth_aws::{AwsAuthConfig, AwsAuthService, SigningScope, sign_headers};
use litellm_python_compat::urllib::quote;
use litellm_secret_value::SecretValue;
use serde_json::json;

use crate::{
    GcpTokenSource,
    google::{
        GoogleLoginFailure, PythonException, builders::exchanged, credentials::ExternalAccountInfo,
        environment::GoogleEnvironment,
    },
};

/// An external account in `vertex_credentials` with LiteLLM's own `aws_*` keys, which Python's
/// `_credentials_from_json` signs in with through `aws`, AWS acting as the identity provider.
pub(crate) fn explicit_aws_login(
    info: &ExternalAccountInfo,
    environment: &Arc<dyn GoogleEnvironment>,
    aws: &AwsAuthService,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    exchanged(
        info,
        explicit_aws_credentials(info, environment.clone(), aws.clone())?,
        environment,
        PythonException::Malformed,
    )
}

/// A key's text when it is set and not empty.
fn text(field: &Option<Option<String>>) -> Option<String> {
    field.clone().flatten().filter(|value| !value.is_empty())
}

const MISSING_REGION: &str = "aws_region_name is required in the WIF credential JSON when using explicit AWS authentication. Add \"aws_region_name\": \"<your-region>\" to your credential file.";
/// `aws._DEFAULT_AWS_REGIONAL_CREDENTIAL_VERIFICATION_URL`.
const CALLER_IDENTITY_URL: &str =
    "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15";
const TARGET_RESOURCE_HEADER: &str = "x-goog-cloud-target-resource";

/// Python's `_credentials_from_json` routes an `external_account` whose `environment_id` names
/// AWS and that carries any of LiteLLM's `aws_*` keys, even a null one, to
/// `VertexAIAwsWifAuth.credentials_from_explicit_aws`.
pub(crate) fn uses_explicit_aws(info: &ExternalAccountInfo) -> bool {
    info.credential_source
        .as_ref()
        .and_then(|source| source.environment_id.as_deref())
        .is_some_and(|environment_id| environment_id.contains("aws"))
        && info.aws.any_given()
}

/// The external account Google's Rust SDK exchanges the signed request for at STS, before any
/// impersonation. `aws` resolves the AWS credentials, and with them an `aws_web_identity_token`
/// reference such as `oidc/...`, as Bedrock's login does.
pub(crate) fn explicit_aws_credentials(
    info: &ExternalAccountInfo,
    environment: Arc<dyn GoogleEnvironment>,
    aws: AwsAuthService,
) -> Result<ProgrammaticBuilder, GoogleLoginFailure> {
    let subject = signed_caller_identity(info, environment, aws)?;
    let audience = subject.audience.clone();
    let builder = ProgrammaticBuilder::new(Arc::new(subject))
        .with_audience(audience)
        .with_subject_token_type(info.subject_token_type.clone().unwrap_or_default());
    let builder = info
        .token_url
        .iter()
        .fold(builder, |builder, url| builder.with_token_url(url));
    Ok(info
        .universe_domain
        .iter()
        .fold(builder, |builder, domain| {
            builder.with_universe_domain(domain)
        }))
}

/// The subject token provider for the file's `aws_*` keys, refused without `aws_region_name`.
fn signed_caller_identity(
    info: &ExternalAccountInfo,
    environment: Arc<dyn GoogleEnvironment>,
    aws: AwsAuthService,
) -> Result<SignedCallerIdentity, GoogleLoginFailure> {
    let keys = &info.aws;
    let region = text(&keys.aws_region_name)
        .ok_or_else(|| GoogleLoginFailure::new(PythonException::Value, MISSING_REGION))?;
    let secret = |field: &Option<Option<String>>| text(field).map(SecretValue::new);
    let config = AwsAuthConfig {
        access_key_id: secret(&keys.aws_access_key_id),
        secret_access_key: secret(&keys.aws_secret_access_key),
        session_token: secret(&keys.aws_session_token),
        region_name: Some(region.clone()),
        session_name: text(&keys.aws_session_name),
        profile_name: text(&keys.aws_profile_name),
        role_name: text(&keys.aws_role_name),
        web_identity_token: secret(&keys.aws_web_identity_token),
        sts_endpoint: text(&keys.aws_sts_endpoint),
        external_id: secret(&keys.aws_external_id),
        ..AwsAuthConfig::default()
    };
    Ok(SignedCallerIdentity {
        aws,
        config,
        region,
        audience: info.audience.clone().unwrap_or_default(),
        environment,
    })
}

/// The subject token of LiteLLM's explicit AWS login: a GetCallerIdentity request signed with
/// the credentials LiteLLM's AWS login resolves from the file's `aws_*` keys.
struct SignedCallerIdentity {
    aws: AwsAuthService,
    config: AwsAuthConfig,
    region: String,
    audience: String,
    environment: Arc<dyn GoogleEnvironment>,
}

impl std::fmt::Debug for SignedCallerIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignedCallerIdentity")
            .field("region", &self.region)
            .finish_non_exhaustive()
    }
}

fn refresh_error(error: impl std::fmt::Display) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::Refresh, error.to_string())
}

impl SubjectTokenProvider for SignedCallerIdentity {
    type Error = GoogleLoginFailure;

    async fn subject_token(&self) -> Result<SubjectToken, GoogleLoginFailure> {
        let environment = self.environment.clone();
        let credentials = self
            .aws
            .resolve_credentials(self.config.clone(), &move |name| environment.os_env(name))
            .await
            .map_err(refresh_error)?;
        caller_identity_subject_token(
            &credentials,
            &self.region,
            &CALLER_IDENTITY_URL.replace("{region}", &self.region),
            &self.audience,
            SystemTime::now(),
        )
        .map(|token| subject_token::Builder::new(token).build())
        .map_err(refresh_error)
    }
}

/// google-auth's `aws.Credentials.retrieve_subject_token`: the signed request's headers in the
/// order google-auth lists them, with the unsigned target resource last, serialized with sorted
/// keys and URL-quoted.
fn caller_identity_subject_token(
    credentials: &litellm_auth_aws::Credentials,
    region: &str,
    url: &str,
    audience: &str,
    time: SystemTime,
) -> Result<String, litellm_auth_aws::Error> {
    let host = url::Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .unwrap_or_default();
    let unsigned = HeaderMap::from_iter([(
        http::header::HOST,
        HeaderValue::from_str(&host).map_err(|_| litellm_auth_aws::Error::NonAsciiHeader {
            name: "host".into(),
        })?,
    )]);
    let signed = sign_headers(
        "POST",
        url,
        &[],
        &unsigned,
        SigningScope {
            region,
            service: "sts",
            credentials,
            time,
        },
    )?;
    let header = |name: &str| {
        signed
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(|value| json!({"key": if name == AUTHORIZATION { "Authorization" } else { name }, "value": value}))
    };
    let headers: Vec<serde_json::Value> = [
        header(AUTHORIZATION.as_str()),
        Some(json!({"key": "host", "value": host})),
        header("x-amz-date"),
        header("x-amz-security-token"),
        Some(json!({"key": TARGET_RESOURCE_HEADER, "value": audience})),
    ]
    .into_iter()
    .flatten()
    .collect();
    let request = json!({"headers": headers, "method": "POST", "url": url}).to_string();
    Ok(quote(&request, "/"))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::Value;
    use std::time::{Duration, UNIX_EPOCH};

    use super::*;

    fn aws_info(extra: Value) -> ExternalAccountInfo {
        let base = json!({
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/pool/providers/aws",
            "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
            "credential_source": {"environment_id": "aws1"}
        });
        let info: serde_json::Map<String, Value> = base
            .as_object()
            .unwrap()
            .clone()
            .into_iter()
            .chain(extra.as_object().unwrap().clone())
            .collect();
        serde_json::from_value(Value::Object(info)).unwrap()
    }

    #[rstest]
    #[case::static_keys(aws_info(json!({"aws_access_key_id": "key"})), true)]
    #[case::null_key(aws_info(json!({"aws_role_name": null})), true)]
    #[case::metadata_only(aws_info(json!({})), false)]
    #[case::not_aws(
        serde_json::from_value(json!({"aws_access_key_id": "key", "credential_source": {"file": "/t"}})).unwrap(),
        false
    )]
    fn explicit_aws_is_chosen_as_python_chooses_it(
        #[case] info: ExternalAccountInfo,
        #[case] expected: bool,
    ) {
        assert_eq!(uses_explicit_aws(&info), expected);
    }

    #[rstest]
    #[case::missing(json!({"aws_access_key_id": "key"}))]
    #[case::empty(json!({"aws_access_key_id": "key", "aws_region_name": ""}))]
    #[case::null(json!({"aws_region_name": null}))]
    fn a_missing_region_is_refused_before_any_aws_call(#[case] extra: Value) {
        let environment = Arc::new(crate::google::OnGce {
            project: "p",
            token: "t",
        });
        let Err(failure) =
            explicit_aws_credentials(&aws_info(extra), environment, AwsAuthService::default())
        else {
            panic!("a credential without aws_region_name should not build");
        };
        assert_eq!(
            failure,
            GoogleLoginFailure::new(PythonException::Value, MISSING_REGION)
        );
    }

    /// The web identity tokens LiteLLM's secrets would mint: an `oidc/` reference becomes a token
    /// and any other text comes back as it is.
    #[derive(Debug)]
    struct MintedTokens;

    impl litellm_auth_aws::WebIdentityTokenSource for MintedTokens {
        fn web_identity_token<'a>(
            &'a self,
            reference: &'a str,
        ) -> litellm_auth_aws::WebIdentityTokenFuture<'a> {
            Box::pin(async move {
                Ok(Some(SecretValue::new(
                    match reference.strip_prefix("oidc/") {
                        Some(_) => "minted-oidc-token",
                        None => reference,
                    },
                )))
            })
        }
    }

    /// An AWS STS endpoint that answers AssumeRoleWithWebIdentity and records each token sent.
    async fn sts() -> (String, Arc<std::sync::Mutex<Vec<String>>>) {
        let sent = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = sent.clone();
        let app = axum::Router::new().route(
            "/",
            axum::routing::post(move |body: String| async move {
                let form: std::collections::HashMap<String, String> =
                    url::form_urlencoded::parse(body.as_bytes()).into_owned().collect();
                assert_eq!(form["Action"], "AssumeRoleWithWebIdentity");
                recorded.lock().unwrap().push(form["WebIdentityToken"].clone());
                "<AssumeRoleWithWebIdentityResponse xmlns=\"https://sts.amazonaws.com/doc/2011-06-15/\"><AssumeRoleWithWebIdentityResult><Credentials><AccessKeyId>web-access</AccessKeyId><SecretAccessKey>web-secret</SecretAccessKey><SessionToken>web-session</SessionToken><Expiration>2100-01-01T00:00:00Z</Expiration></Credentials></AssumeRoleWithWebIdentityResult></AssumeRoleWithWebIdentityResponse>"
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, sent)
    }

    #[rstest]
    #[case::oidc_reference("oidc/google/https://example.com", "minted-oidc-token")]
    #[case::plain_token("plain-web-identity-token", "plain-web-identity-token")]
    #[tokio::test]
    async fn the_web_identity_token_is_resolved_before_it_reaches_sts(
        #[case] configured: &str,
        #[case] expected: &str,
    ) {
        let (endpoint, sent) = sts().await;
        let info = aws_info(json!({
            "aws_region_name": "us-east-1",
            "aws_role_name": "arn:aws:iam::123456789012:role/vertex",
            "aws_session_name": "vertex-session",
            "aws_web_identity_token": configured,
            "aws_sts_endpoint": endpoint,
        }));
        let environment = Arc::new(crate::google::OnGce {
            project: "p",
            token: "t",
        });
        let subject = signed_caller_identity(
            &info,
            environment,
            AwsAuthService::with_web_identity_tokens(Arc::new(MintedTokens)),
        )
        .unwrap();

        let token = subject.subject_token().await.unwrap();

        assert_eq!(*sent.lock().unwrap(), [expected]);
        // The request is signed with the role's credentials STS granted for that token.
        assert!(format!("{token:?}").contains("web-access"));
    }

    #[rstest]
    fn the_subject_token_is_google_auth_s_serialized_signed_request() {
        let credentials = litellm_auth_aws::Credentials::new(
            "AKIDEXAMPLE",
            "secret",
            Some("session".into()),
            None,
            "test",
        );
        let token = caller_identity_subject_token(
            &credentials,
            "us-east-1",
            &CALLER_IDENTITY_URL.replace("{region}", "us-east-1"),
            "//iam.googleapis.com/projects/1/pool",
            UNIX_EPOCH + Duration::from_secs(1_700_000_000),
        )
        .unwrap();
        let decoded = percent_encoding::percent_decode_str(&token)
            .decode_utf8()
            .unwrap();
        let request: Value = serde_json::from_str(&decoded).unwrap();
        let keys: Vec<&str> = request["headers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|header| header["key"].as_str().unwrap())
            .collect();
        assert_eq!(
            keys,
            [
                "Authorization",
                "host",
                "x-amz-date",
                "x-amz-security-token",
                TARGET_RESOURCE_HEADER
            ]
        );
        assert_eq!(request["method"], "POST");
        assert_eq!(
            request["url"],
            "https://sts.us-east-1.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
        );
        assert_eq!(request["headers"][2]["value"], "20231114T221320Z");
        assert!(
            request["headers"][0]["value"]
                .as_str()
                .unwrap()
                .starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20231114/us-east-1/sts/aws4_request, SignedHeaders=host;x-amz-date;x-amz-security-token, ")
        );
        assert!(decoded.starts_with(r#"{"headers":[{"key":"Authorization","value":"#));
        assert!(!token.contains(':') && !token.contains('"'));
    }
}
