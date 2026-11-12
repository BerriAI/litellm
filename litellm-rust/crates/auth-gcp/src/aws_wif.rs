use std::{sync::Arc, time::SystemTime};

use google_cloud_auth::credentials::{
    CacheableResource, Credentials,
    external_account::ProgrammaticBuilder,
    subject_token::{Builder as SubjectTokenBuilder, SubjectToken, SubjectTokenProvider},
};
use google_cloud_auth::errors::CredentialsError;
use http::{
    Extensions, HeaderMap, HeaderValue,
    header::{AUTHORIZATION, HOST},
};
use litellm_auth_aws::{AwsAuthConfig, AwsAuthService, SigningScope, sign_headers};
use litellm_auth_types::{Error, ErrorDetail};
use litellm_secret_value::SecretValue;
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use serde_json::{Value, json};

use crate::{
    constants::CLOUD_PLATFORM_SCOPE,
    identity::{GcpAuthFuture, GcpTokenSource},
};

pub(crate) type Environment = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

const AWS_PARAM_NAMES: [&str; 10] = [
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
const CALLER_IDENTITY_URL: &str =
    "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15";
const TARGET_RESOURCE_HEADER: &str = "x-goog-cloud-target-resource";
const IMPERSONATION_URL_PREFIX: &str =
    "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/";
const IMPERSONATION_URL_SUFFIX: &str = ":generateAccessToken";
const PYTHON_QUOTE_SAFE: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'_')
    .remove(b'.')
    .remove(b'-')
    .remove(b'~')
    .remove(b'/');

/// Whether LiteLLM's explicit AWS keys in an AWS workload identity federation config replace
/// the metadata service, as Python's `VertexAIAwsWifAuth.extract_aws_params` decides.
pub(crate) fn uses_explicit_aws(configured: &Value) -> bool {
    configured
        .get("credential_source")
        .and_then(|source| source.get("environment_id"))
        .and_then(Value::as_str)
        .is_some_and(|environment_id| environment_id.contains("aws"))
        && AWS_PARAM_NAMES
            .iter()
            .any(|name| configured.get(*name).is_some())
}

pub(crate) fn explicit_aws_credentials(
    configured: &Value,
    environment: Environment,
) -> Result<Arc<dyn GcpTokenSource>, Error> {
    let text = |name: &str| {
        configured
            .get(name)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let secret = |name: &str| text(name).map(SecretValue::new);
    let region = text("aws_region_name").ok_or(Error::InvalidConfiguration(
        ErrorDetail::MissingSettings {
            credential: "explicit AWS workload identity federation",
            settings: "aws_region_name in the credential JSON",
        },
    ))?;
    let config = AwsAuthConfig {
        access_key_id: secret("aws_access_key_id"),
        secret_access_key: secret("aws_secret_access_key"),
        session_token: secret("aws_session_token"),
        region_name: Some(region.clone()),
        session_name: text("aws_session_name"),
        profile_name: text("aws_profile_name"),
        role_name: text("aws_role_name"),
        web_identity_token: secret("aws_web_identity_token"),
        sts_endpoint: text("aws_sts_endpoint"),
        external_id: secret("aws_external_id"),
    };
    let audience = text("audience").unwrap_or_default();
    let subject_tokens = AwsSubjectTokens {
        auth: AwsAuthService::default(),
        config,
        region,
        audience: audience.clone(),
        environment,
    };
    let mut builder = ProgrammaticBuilder::new(Arc::new(subject_tokens))
        .with_scopes([CLOUD_PLATFORM_SCOPE])
        .with_audience(audience);
    if let Some(subject_token_type) = text("subject_token_type") {
        builder = builder.with_subject_token_type(subject_token_type);
    }
    if let Some(token_url) = text("token_url") {
        builder = builder.with_token_url(token_url);
    }
    if let Some(universe_domain) = text("universe_domain") {
        builder = builder.with_universe_domain(universe_domain);
    }
    if let Some(url) = text("service_account_impersonation_url") {
        builder = builder.with_target_principal(impersonated_principal(&url)?);
    }
    let credentials = builder.build().map_err(|error| {
        Error::CredentialAcquisition(ErrorDetail::failed("Google credentials", error))
    })?;
    Ok(Arc::new(ExternalAccountTokenSource(credentials)))
}

fn impersonated_principal(url: &str) -> Result<&str, Error> {
    url.strip_prefix(IMPERSONATION_URL_PREFIX)
        .and_then(|rest| rest.strip_suffix(IMPERSONATION_URL_SUFFIX))
        .filter(|principal| !principal.is_empty() && !principal.contains('/'))
        .ok_or_else(|| {
            Error::InvalidConfiguration(ErrorDetail::InvalidType {
                field: "service_account_impersonation_url".into(),
                expected: "an iamcredentials.googleapis.com generateAccessToken URL",
            })
        })
}

struct AwsSubjectTokens {
    auth: AwsAuthService,
    config: AwsAuthConfig,
    region: String,
    audience: String,
    environment: Environment,
}

impl std::fmt::Debug for AwsSubjectTokens {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AwsSubjectTokens")
            .field("region", &self.region)
            .finish_non_exhaustive()
    }
}

impl SubjectTokenProvider for AwsSubjectTokens {
    type Error = CredentialsError;

    async fn subject_token(&self) -> Result<SubjectToken, CredentialsError> {
        let credentials = self
            .auth
            .resolve_credentials(self.config.clone(), &*self.environment)
            .await
            .map_err(|error| CredentialsError::from_source(false, error))?;
        caller_identity_subject_token(
            &credentials,
            &self.region,
            &self.audience,
            SystemTime::now(),
        )
        .map(|token| SubjectTokenBuilder::new(token).build())
    }
}

/// The serialized, signed AWS STS `GetCallerIdentity` request Google's STS accepts as the
/// subject token, in the shape google-auth's `aws.Credentials.retrieve_subject_token` sends.
fn caller_identity_subject_token(
    credentials: &litellm_auth_aws::Credentials,
    region: &str,
    audience: &str,
    time: SystemTime,
) -> Result<String, CredentialsError> {
    let invalid = |error| CredentialsError::from_source(false, error);
    let url = CALLER_IDENTITY_URL.replace("{region}", region);
    let mut headers = HeaderMap::new();
    headers.insert(
        HOST,
        HeaderValue::from_str(&format!("sts.{region}.amazonaws.com")).map_err(invalid)?,
    );
    headers.insert(
        TARGET_RESOURCE_HEADER,
        HeaderValue::from_str(audience).map_err(invalid)?,
    );
    let signed = sign_headers(
        "POST",
        &url,
        &[],
        &headers,
        SigningScope {
            region,
            service: "sts",
            credentials,
            time,
        },
    )
    .map_err(|error| CredentialsError::from_source(false, error))?;
    let headers = signed
        .iter()
        .chain(headers.iter())
        .map(|(name, value)| {
            let key = if name == AUTHORIZATION {
                "Authorization"
            } else {
                name.as_str()
            };
            value
                .to_str()
                .map(|value| json!({"key": key, "value": value}))
                .map_err(|error| CredentialsError::from_source(false, error))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let request = json!({"headers": headers, "method": "POST", "url": url}).to_string();
    Ok(utf8_percent_encode(&request, PYTHON_QUOTE_SAFE).to_string())
}

struct ExternalAccountTokenSource(Credentials);

impl GcpTokenSource for ExternalAccountTokenSource {
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
                Err(error) => {
                    return Err(Error::CredentialAcquisition(ErrorDetail::failed(
                        "Google credentials",
                        error,
                    )));
                }
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
    use std::sync::Mutex;

    use axum::{Form, Json, Router, routing::post};
    use rstest::rstest;

    use super::*;

    fn no_environment() -> Environment {
        Arc::new(|_| None)
    }

    fn aws_config(extra: Value) -> Value {
        let mut configured = json!({
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/pool/providers/aws",
            "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
            "credential_source": {
                "environment_id": "aws1",
                "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
            }
        });
        configured
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        configured
    }

    #[rstest]
    #[case::explicit_keys(aws_config(json!({"aws_access_key_id": "key"})), true)]
    #[case::null_key_is_still_explicit(aws_config(json!({"aws_role_name": null})), true)]
    #[case::metadata_only(aws_config(json!({})), false)]
    #[case::not_aws(json!({"type": "external_account", "aws_access_key_id": "key", "credential_source": {"file": "/token"}}), false)]
    fn explicit_aws_is_detected_like_python(#[case] configured: Value, #[case] expected: bool) {
        assert_eq!(uses_explicit_aws(&configured), expected);
    }

    #[rstest]
    #[case::canonical(
        "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken",
        Some("sa@p.iam.gserviceaccount.com")
    )]
    #[case::other_host(
        "https://iamcredentials.example.com/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken",
        None
    )]
    #[case::missing_principal(
        "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/:generateAccessToken",
        None
    )]
    fn impersonation_url_names_its_principal(#[case] url: &str, #[case] expected: Option<&str>) {
        assert_eq!(impersonated_principal(url).ok(), expected);
    }

    #[rstest]
    fn a_missing_region_is_refused_before_any_aws_call() {
        let Err(error) = explicit_aws_credentials(
            &aws_config(json!({"aws_access_key_id": "key", "aws_secret_access_key": "secret"})),
            no_environment(),
        ) else {
            panic!("a config without aws_region_name should not load");
        };
        assert!(
            litellm_error::Report::new(&error)
                .to_string()
                .contains("aws_region_name")
        );
    }

    #[rstest]
    #[case::static_keys(None)]
    #[case::session_token(Some("session"))]
    fn subject_token_is_a_signed_caller_identity_request(#[case] session_token: Option<&str>) {
        let credentials = litellm_auth_aws::Credentials::new(
            "AKIDEXAMPLE",
            "secret",
            session_token.map(str::to_string),
            None,
            "test",
        );
        let token = caller_identity_subject_token(
            &credentials,
            "eu-west-1",
            "//iam.googleapis.com/audience",
            SystemTime::UNIX_EPOCH,
        )
        .unwrap();
        let decoded = percent_encoding::percent_decode_str(&token)
            .decode_utf8()
            .unwrap();
        let request: Value = serde_json::from_str(&decoded).unwrap();
        let header = |key: &str| {
            request["headers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|header| header["key"] == key)
                .map(|header| header["value"].as_str().unwrap().to_string())
        };
        assert_eq!(
            request["url"],
            "https://sts.eu-west-1.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
        );
        assert_eq!(request["method"], "POST");
        assert_eq!(
            header("host").as_deref(),
            Some("sts.eu-west-1.amazonaws.com")
        );
        assert_eq!(
            header(TARGET_RESOURCE_HEADER).as_deref(),
            Some("//iam.googleapis.com/audience")
        );
        assert_eq!(header("x-amz-date").as_deref(), Some("19700101T000000Z"));
        assert_eq!(header("x-amz-security-token").as_deref(), session_token);
        let authorization = header("Authorization").unwrap();
        assert!(
            authorization.starts_with(
                "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/19700101/eu-west-1/sts/aws4_request"
            ),
            "{authorization}"
        );
        assert!(authorization.contains(TARGET_RESOURCE_HEADER));
    }

    #[rstest]
    #[tokio::test]
    async fn explicit_aws_keys_are_exchanged_without_the_metadata_service() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let exchanges = Arc::new(Mutex::new(Vec::<Vec<(String, String)>>::new()));
        let recorded = exchanges.clone();
        let app = Router::new().route(
            "/token",
            post(move |Form(form): Form<Vec<(String, String)>>| {
                let recorded = recorded.clone();
                async move {
                    recorded.lock().unwrap().push(form);
                    Json(json!({
                        "access_token": "federated-token",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                        "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let configured = aws_config(json!({
            "token_url": format!("http://{address}/token"),
            "aws_access_key_id": "AKIDEXAMPLE",
            "aws_secret_access_key": "secret",
            "aws_region_name": "us-west-2",
        }));
        let ambient: Environment =
            Arc::new(|name: &str| (name == "AWS_ACCESS_KEY_ID").then(|| "ambient-key".to_string()));

        let provider = explicit_aws_credentials(&configured, ambient).unwrap();

        assert_eq!(provider.token().await.unwrap().expose(), "federated-token");
        let exchanges = exchanges.lock().unwrap();
        let form = exchanges.first().unwrap();
        let field = |name: &str| {
            form.iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
                .unwrap()
        };
        assert_eq!(field("audience"), configured["audience"]);
        assert_eq!(field("scope"), CLOUD_PLATFORM_SCOPE);
        let subject = percent_encoding::percent_decode_str(&field("subject_token"))
            .decode_utf8()
            .unwrap()
            .into_owned();
        assert!(subject.contains("sts.us-west-2.amazonaws.com"), "{subject}");
        assert!(subject.contains("Credential=AKIDEXAMPLE/"), "{subject}");
        server.abort();
    }
}
