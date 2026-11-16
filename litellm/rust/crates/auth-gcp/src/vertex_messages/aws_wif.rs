use std::{sync::Arc, time::SystemTime};

use google_cloud_auth::{
    credentials::{
        CacheableResource, Credentials,
        external_account::ProgrammaticBuilder,
        subject_token::{Builder as SubjectTokenBuilder, SubjectToken, SubjectTokenProvider},
    },
    errors::CredentialsError,
};
use http::{Extensions, HeaderMap, HeaderValue, header::AUTHORIZATION};
use litellm_auth_aws::{AwsAuthConfig, AwsAuthService, SigningScope, sign_headers};
use litellm_auth_types::{Error, ErrorDetail};
use litellm_secret_value::SecretValue;
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use serde_json::{Map, Value, json};

use super::{
    builders::{TokenSource, token_failure},
    failure::{GoogleLoginFailure, PythonException},
};
use crate::{
    constants::CLOUD_PLATFORM_SCOPE,
    identity::{GcpAuthFuture, GcpTokenSource},
};

pub(crate) type OsEnvironment = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

const AWS_CREDENTIAL_KEYS: [&str; 10] = [
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_session_name",
    "aws_profile_name",
    "aws_role_name",
    "aws_web_identity_token",
    "aws_sts_endpoint",
    "aws_external_id",
];
const MISSING_REGION: &str = "aws_region_name is required in the WIF credential JSON when using explicit AWS authentication. Add \"aws_region_name\": \"<your-region>\" to your credential file.";
const CALLER_IDENTITY_URL: &str = "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15";
const TARGET_RESOURCE_HEADER: &str = "x-goog-cloud-target-resource";
const IMPERSONATION_URL_PREFIX: &str = "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/";
const IMPERSONATION_URL_SUFFIX: &str = ":generateAccessToken";
const PYTHON_QUOTE_SAFE: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'_')
    .remove(b'.')
    .remove(b'-')
    .remove(b'~')
    .remove(b'/');

/// Python's `_credentials_from_json` routes an `external_account` whose `environment_id` names
/// AWS and that carries any of LiteLLM's `aws_*` keys, even a null one, to
/// `VertexAIAwsWifAuth.credentials_from_explicit_aws`.
pub(crate) fn uses_explicit_aws(info: &Map<String, Value>) -> bool {
    let names_aws = info
        .get("credential_source")
        .and_then(|source| source.get("environment_id"))
        .and_then(Value::as_str)
        .is_some_and(|environment_id| environment_id.contains("aws"));
    names_aws && AWS_CREDENTIAL_KEYS.iter().any(|key| info.contains_key(*key))
}

fn text(info: &Map<String, Value>, field: &str) -> Option<String> {
    info.get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Python's `VertexAIAwsWifAuth.credentials_from_explicit_aws`: the subject token is a
/// GetCallerIdentity request signed with the AWS credentials LiteLLM's own `aws_*` keys
/// resolve, never the EC2 metadata service.
pub(crate) fn explicit_aws_credentials(
    info: &Map<String, Value>,
    os_env: OsEnvironment,
) -> Result<TokenSource, GoogleLoginFailure> {
    let region = text(info, "aws_region_name")
        .ok_or_else(|| GoogleLoginFailure::new(PythonException::Value, MISSING_REGION))?;
    let secret = |field: &str| text(info, field).map(SecretValue::new);
    let config = AwsAuthConfig {
        access_key_id: secret("aws_access_key_id"),
        secret_access_key: secret("aws_secret_access_key"),
        session_token: secret("aws_session_token"),
        region_name: Some(region.clone()),
        session_name: text(info, "aws_session_name"),
        profile_name: text(info, "aws_profile_name"),
        role_name: text(info, "aws_role_name"),
        web_identity_token: secret("aws_web_identity_token"),
        sts_endpoint: text(info, "aws_sts_endpoint"),
        external_id: secret("aws_external_id"),
        ..AwsAuthConfig::default()
    };
    let audience = text(info, "audience").unwrap_or_default();
    let subject_tokens = SignedCallerIdentity {
        aws: AwsAuthService::default(),
        config,
        region,
        audience: audience.clone(),
        os_env,
    };
    let builder = ProgrammaticBuilder::new(Arc::new(subject_tokens))
        .with_scopes([CLOUD_PLATFORM_SCOPE])
        .with_audience(audience);
    let builder = match text(info, "subject_token_type") {
        Some(kind) => builder.with_subject_token_type(kind),
        None => builder,
    };
    let builder = match text(info, "token_url") {
        Some(url) => builder.with_token_url(url),
        None => builder,
    };
    let builder = match text(info, "universe_domain") {
        Some(domain) => builder.with_universe_domain(domain),
        None => builder,
    };
    let builder = match text(info, "service_account_impersonation_url") {
        Some(url) => builder.with_target_principal(impersonated_principal(&url)?),
        None => builder,
    };
    builder
        .build()
        .map(|credentials| Arc::new(HeaderTokenSource(credentials)) as TokenSource)
        .map_err(|error| GoogleLoginFailure::new(PythonException::InvalidValue, error.to_string()))
}

fn impersonated_principal(url: &str) -> Result<String, GoogleLoginFailure> {
    url.strip_prefix(IMPERSONATION_URL_PREFIX)
        .and_then(|rest| rest.strip_suffix(IMPERSONATION_URL_SUFFIX))
        .filter(|principal| !principal.is_empty() && !principal.contains('/'))
        .map(str::to_string)
        .ok_or_else(|| {
            GoogleLoginFailure::new(
                PythonException::InvalidValue,
                "Unable to determine target principal from service account impersonation URL.",
            )
        })
}

struct SignedCallerIdentity {
    aws: AwsAuthService,
    config: AwsAuthConfig,
    region: String,
    audience: String,
    os_env: OsEnvironment,
}

impl std::fmt::Debug for SignedCallerIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SignedCallerIdentity")
            .field("region", &self.region)
            .finish_non_exhaustive()
    }
}

impl SubjectTokenProvider for SignedCallerIdentity {
    type Error = CredentialsError;

    async fn subject_token(&self) -> Result<SubjectToken, CredentialsError> {
        let os_env = self.os_env.clone();
        let credentials = self
            .aws
            .resolve_credentials(self.config.clone(), &move |name| os_env(name))
            .await
            .map_err(|error| CredentialsError::from_source(false, error))?;
        caller_identity_subject_token(&credentials, &self.region, &self.audience, SystemTime::now())
            .map(|token| SubjectTokenBuilder::new(token).build())
            .map_err(|error| CredentialsError::from_source(false, error))
    }
}

/// google-auth's `aws.Credentials.retrieve_subject_token`: the signed request's headers in the
/// order google-auth lists them, with the unsigned target resource last, serialized with sorted
/// keys and URL-quoted.
fn caller_identity_subject_token(
    credentials: &litellm_auth_aws::Credentials,
    region: &str,
    audience: &str,
    time: SystemTime,
) -> Result<String, litellm_auth_aws::Error> {
    let url = CALLER_IDENTITY_URL.replace("{region}", region);
    let host = format!("sts.{region}.amazonaws.com");
    let unsigned = HeaderMap::from_iter([(
        http::header::HOST,
        HeaderValue::from_str(&host).map_err(|_| litellm_auth_aws::Error::NonAsciiHeader {
            name: "host".into(),
        })?,
    )]);
    let signed = sign_headers(
        "POST",
        &url,
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
    let headers: Vec<Value> = [
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
    Ok(utf8_percent_encode(&request, PYTHON_QUOTE_SAFE).to_string())
}

struct HeaderTokenSource(Credentials);

impl GcpTokenSource for HeaderTokenSource {
    fn project_id(&self) -> GcpAuthFuture<'_, String> {
        Box::pin(async {
            Err(Error::CredentialAcquisition(ErrorDetail::Unresolved {
                subject: "GCP resource project",
            }))
        })
    }

    fn token(&self) -> GcpAuthFuture<'_, SecretValue> {
        Box::pin(async move {
            let headers = match self.0.headers(Extensions::new()).await {
                Ok(CacheableResource::New { data, .. }) => data,
                Ok(CacheableResource::NotModified) => HeaderMap::new(),
                Err(error) => return Err(token_failure(&error).into()),
            };
            headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.strip_prefix("Bearer "))
                .map(SecretValue::new)
                .ok_or(Error::CredentialAcquisition(ErrorDetail::Unresolved {
                    subject: "Google access token",
                }))
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use std::time::{Duration, UNIX_EPOCH};

    use super::*;

    fn aws_info(extra: Value) -> Map<String, Value> {
        let base = json!({
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/pool/providers/aws",
            "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
            "credential_source": {"environment_id": "aws1"}
        });
        base.as_object()
            .unwrap()
            .clone()
            .into_iter()
            .chain(extra.as_object().unwrap().clone())
            .collect()
    }

    #[rstest]
    #[case::static_keys(aws_info(json!({"aws_access_key_id": "key"})), true)]
    #[case::null_key(aws_info(json!({"aws_role_name": null})), true)]
    #[case::metadata_only(aws_info(json!({})), false)]
    #[case::not_aws(
        json!({"type": "external_account", "aws_access_key_id": "key", "credential_source": {"file": "/t"}})
            .as_object().unwrap().clone(),
        false
    )]
    fn explicit_aws_is_chosen_as_python_chooses_it(#[case] info: Map<String, Value>, #[case] expected: bool) {
        assert_eq!(uses_explicit_aws(&info), expected);
    }

    #[rstest]
    #[case::missing(json!({"aws_access_key_id": "key"}))]
    #[case::empty(json!({"aws_access_key_id": "key", "aws_region_name": ""}))]
    #[case::null(json!({"aws_region_name": null}))]
    fn a_missing_region_is_refused_before_any_aws_call(#[case] extra: Value) {
        let Err(failure) = explicit_aws_credentials(&aws_info(extra), Arc::new(|_| None)) else {
            panic!("a credential without aws_region_name should not build");
        };
        assert_eq!(failure, GoogleLoginFailure::new(PythonException::Value, MISSING_REGION));
    }

    #[rstest]
    fn the_subject_token_is_google_auth_s_serialized_signed_request() {
        let credentials = litellm_auth_aws::Credentials::new("AKIDEXAMPLE", "secret", Some("session".into()), None, "test");
        let token = caller_identity_subject_token(
            &credentials,
            "us-east-1",
            "//iam.googleapis.com/projects/1/pool",
            UNIX_EPOCH + Duration::from_secs(1_700_000_000),
        )
        .unwrap();
        let decoded = percent_encoding::percent_decode_str(&token).decode_utf8().unwrap();
        let request: Value = serde_json::from_str(&decoded).unwrap();
        let keys: Vec<&str> = request["headers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|header| header["key"].as_str().unwrap())
            .collect();
        assert_eq!(
            keys,
            ["Authorization", "host", "x-amz-date", "x-amz-security-token", TARGET_RESOURCE_HEADER]
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

    #[rstest]
    #[case::canonical(
        "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken",
        true
    )]
    #[case::other_host("https://example.com/v1/projects/-/serviceAccounts/sa:generateAccessToken", false)]
    fn the_impersonated_principal_comes_from_the_canonical_url(#[case] url: &str, #[case] accepted: bool) {
        assert_eq!(impersonated_principal(url).is_ok(), accepted);
    }
}
