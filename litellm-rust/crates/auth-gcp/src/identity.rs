use crate::{
    aws_wif::{explicit_aws_credentials, uses_explicit_aws},
    config::{
        CredentialSource, explicit_credentials_project, is_base64_encoded_json,
        names_a_credentials_file, non_empty_env, well_known_credentials_file,
    },
    constants::{
        CLOUD_PLATFORM_SCOPE, GOOGLE_APPLICATION_CREDENTIALS_ENV, GOOGLE_OAUTH_TOKEN_ENDPOINT,
    },
};
use gcp_auth::{CustomServiceAccount, TokenProvider};
use litellm_auth_types::{Error, ErrorDetail};
use litellm_secret_value::SecretValue;
use moka::future::Cache;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};

#[derive(Clone)]
pub struct GcpAuth {
    providers: Cache<CredentialCacheKey, Arc<dyn GcpTokenSource>>,
    loader: Arc<dyn GcpProviderLoader>,
}

impl Default for GcpAuth {
    fn default() -> Self {
        Self::new(Arc::new(NativeProviderLoader))
    }
}

impl GcpAuth {
    pub fn new(loader: Arc<dyn GcpProviderLoader>) -> Self {
        Self {
            providers: Cache::builder().max_capacity(64).build(),
            loader,
        }
    }

    pub async fn access_token(&self, source: CredentialSource) -> Result<SecretValue, Error> {
        self.load_provider(source).await?.token().await
    }

    pub(crate) async fn load_provider(
        &self,
        source: CredentialSource,
    ) -> Result<Arc<dyn GcpTokenSource>, Error> {
        self.providers
            .try_get_with(source.cache_key(), self.loader.load(source))
            .await
            .map_err(|error| (*error).clone())
    }
}

pub trait GcpTokenSource: Send + Sync {
    fn project_id(&self) -> GcpAuthFuture<'_, String>;
    fn token(&self) -> GcpAuthFuture<'_, SecretValue>;
}

pub trait GcpProviderLoader: Send + Sync {
    fn load(&self, source: CredentialSource) -> GcpAuthFuture<'_, Arc<dyn GcpTokenSource>>;
}

pub type GcpAuthFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

struct ServiceAccountTokenSource(Arc<dyn TokenProvider>);

impl GcpTokenSource for ServiceAccountTokenSource {
    fn project_id(&self) -> GcpAuthFuture<'_, String> {
        Box::pin(async move {
            self.0
                .project_id()
                .await
                .map(|project| project.to_string())
                .map_err(auth_acquisition_error)
        })
    }

    fn token(&self) -> GcpAuthFuture<'_, SecretValue> {
        Box::pin(async move {
            self.0
                .token(&[CLOUD_PLATFORM_SCOPE])
                .await
                .map(|token| SecretValue::new(token.as_str()))
                .map_err(auth_acquisition_error)
        })
    }
}

struct SdkTokenSource {
    credentials: google_cloud_auth::credentials::AccessTokenCredentials,
    project_id: Option<String>,
}

impl GcpTokenSource for SdkTokenSource {
    fn project_id(&self) -> GcpAuthFuture<'_, String> {
        Box::pin(async move {
            self.project_id
                .clone()
                .ok_or(Error::CredentialAcquisition(ErrorDetail::Unresolved {
                    subject: "GCP resource project",
                }))
        })
    }

    fn token(&self) -> GcpAuthFuture<'_, SecretValue> {
        Box::pin(async move {
            self.credentials
                .access_token()
                .await
                .map(|token| SecretValue::new(token.token))
                .map_err(auth_acquisition_error)
        })
    }
}

async fn load_application_credentials(
    path: impl AsRef<Path>,
) -> Result<Arc<dyn GcpTokenSource>, Error> {
    let contents = tokio::fs::read_to_string(path)
        .await
        .map_err(auth_acquisition_error)?;
    load_credentials_json(&contents)
}

fn load_credentials_json(contents: &str) -> Result<Arc<dyn GcpTokenSource>, Error> {
    let configured: Value = serde_json::from_str(contents).map_err(auth_acquisition_error)?;
    let kind = configured.get("type").and_then(Value::as_str);
    let project_id = explicit_credentials_project(&configured);
    let credentials = match kind {
        Some("authorized_user") => {
            google_cloud_auth::credentials::user_account::Builder::new(configured)
                .with_scopes([CLOUD_PLATFORM_SCOPE])
                .build_access_token_credentials()
        }
        Some("impersonated_service_account") => {
            google_cloud_auth::credentials::impersonated::Builder::new(configured)
                .with_scopes([CLOUD_PLATFORM_SCOPE])
                .build_access_token_credentials()
        }
        Some("external_account") if uses_explicit_aws(&configured) => {
            return explicit_aws_credentials(
                &configured,
                Arc::new(|name| std::env::var(name).ok()),
            );
        }
        Some("external_account") => {
            google_cloud_auth::credentials::external_account::Builder::new(configured)
                .with_scopes([CLOUD_PLATFORM_SCOPE])
                .build_access_token_credentials()
        }
        Some("service_account") | None => {
            let provider =
                CustomServiceAccount::from_json(contents).map_err(auth_acquisition_error)?;
            return Ok(Arc::new(ServiceAccountTokenSource(Arc::new(provider))));
        }
        Some(_) => {
            return Err(Error::InvalidConfiguration(ErrorDetail::InvalidType {
                field: "Google application credentials type".into(),
                expected: "a supported Google credential type",
            }));
        }
    }
    .map_err(auth_acquisition_error)?;
    Ok(Arc::new(SdkTokenSource {
        credentials,
        project_id,
    }))
}

struct NativeProviderLoader;

impl GcpProviderLoader for NativeProviderLoader {
    fn load(&self, source: CredentialSource) -> GcpAuthFuture<'_, Arc<dyn GcpTokenSource>> {
        Box::pin(async move {
            let provider: Arc<dyn TokenProvider> = match source {
                CredentialSource::Inline(configured) => Arc::new(
                    CustomServiceAccount::from_json(validate_request_credentials(
                        configured.expose(),
                    )?)
                    .map_err(auth_acquisition_error)?,
                ),
                CredentialSource::Trusted(configured) => {
                    return load_trusted_credentials(configured.expose()).await;
                }
                CredentialSource::ApplicationCredentials(path) => {
                    return load_application_credentials(&path).await;
                }
                CredentialSource::Adc => {
                    return load_default_credentials(&|name| std::env::var(name).ok()).await;
                }
            };
            Ok(Arc::new(ServiceAccountTokenSource(provider)) as Arc<dyn GcpTokenSource>)
        })
    }
}

/// `google.auth.default`'s order: `GOOGLE_APPLICATION_CREDENTIALS`, gcloud's credentials file
/// of any type, then the metadata server.
async fn load_default_credentials(
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Arc<dyn GcpTokenSource>, Error> {
    match non_empty_env(env_lookup, GOOGLE_APPLICATION_CREDENTIALS_ENV)
        .map(PathBuf::from)
        .or_else(|| well_known_credentials_file(env_lookup))
    {
        Some(path) => load_application_credentials(&path).await,
        None => Ok(Arc::new(ServiceAccountTokenSource(
            gcp_auth::provider().await.map_err(auth_acquisition_error)?,
        ))),
    }
}

fn validate_request_credentials(configured: &str) -> Result<&str, Error> {
    let token_uri = serde_json::from_str::<Value>(configured)
        .ok()
        .and_then(|credentials| {
            credentials
                .get("token_uri")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    if token_uri.as_deref() != Some(GOOGLE_OAUTH_TOKEN_ENDPOINT) {
        return Err(Error::InvalidConfiguration(ErrorDetail::InvalidType {
            field: "request-controlled vertex_credentials token_uri".into(),
            expected: "the canonical Google OAuth token endpoint",
        }));
    }
    Ok(configured)
}

impl CredentialSource {
    fn cache_key(&self) -> CredentialCacheKey {
        match self {
            Self::Inline(configured) => {
                CredentialCacheKey::Inline(Sha256::digest(configured.expose()).into())
            }
            Self::Trusted(configured) => {
                CredentialCacheKey::Trusted(Sha256::digest(configured.expose()).into())
            }
            Self::ApplicationCredentials(path) => {
                CredentialCacheKey::ApplicationCredentials(path.clone())
            }
            Self::Adc => CredentialCacheKey::Adc,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum CredentialCacheKey {
    Inline([u8; 32]),
    Trusted([u8; 32]),
    ApplicationCredentials(String),
    Adc,
}

async fn load_trusted_credentials(configured: &str) -> Result<Arc<dyn GcpTokenSource>, Error> {
    if is_base64_encoded_json(configured) {
        return Err(Error::InvalidConfiguration(
            ErrorDetail::Base64EncodedJson {
                field: "vertex_credentials",
            },
        ));
    }
    if names_a_credentials_file(configured) {
        load_application_credentials(configured).await
    } else {
        load_credentials_json(configured)
    }
}

fn auth_acquisition_error(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::CredentialAcquisition(litellm_auth_types::ErrorDetail::failed(
        "Google credentials",
        error,
    ))
}

#[cfg(test)]
mod tests {
    use litellm_secret_value::SecretValue;

    use super::*;

    #[rstest::rstest]
    #[case::authorized_user("authorized_user", 2, Some("quota-project"))]
    #[case::external_account("external_account", 2, None)]
    #[case::impersonated_service_account("impersonated_service_account", 4, None)]
    #[tokio::test]
    async fn explicit_application_credentials_refresh_expired_tokens(
        #[case] kind: &str,
        #[case] expected_requests: usize,
        #[case] expected_project: Option<&str>,
        #[values(false, true)] inline: bool,
    ) {
        use axum::{Json, Router, routing::post};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let _ = rustls::crypto::ring::default_provider().install_default();
        let requests = Arc::new(AtomicUsize::new(0));
        let recorded = requests.clone();
        let impersonations = requests.clone();
        let app = Router::new().route("/token", post(move || {
            let recorded = recorded.clone();
            async move {
                Json(serde_json::json!({
                    "access_token": format!("token-{}", recorded.fetch_add(1, Ordering::SeqCst)),
                    "expires_in": 3600,
                    "token_type": "Bearer",
                    "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                }))
            }
        }));
        let app = app.route("/v1/projects/-/serviceAccounts/test-principal:generateAccessToken", post(move || {
            let recorded = impersonations.clone();
            async move {
                let expiration = (time::OffsetDateTime::now_utc() + time::Duration::hours(1)).format(&time::format_description::well_known::Rfc3339).unwrap();
                Json(serde_json::json!({
                    "accessToken": format!("impersonated-{}", recorded.fetch_add(1, Ordering::SeqCst)),
                    "expireTime": expiration,
                }))
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let file = tempfile::NamedTempFile::new().unwrap();
        let subject_token = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(subject_token.path(), "subject-token").unwrap();
        let configured = if kind == "external_account" {
            serde_json::json!({
                "type": "external_account", "audience": "//iam.googleapis.com/projects/test/locations/global/workloadIdentityPools/test/providers/test",
                "subject_token_type": "urn:ietf:params:oauth:token-type:jwt", "token_url": format!("http://{address}/token"),
                "credential_source": { "file": subject_token.path().to_str().unwrap() }, "quota_project_id": "quota-project"
            })
        } else {
            let source = serde_json::json!({
                "type": "authorized_user", "client_id": "user-client", "client_secret": "user-secret",
                "refresh_token": "user-refresh", "quota_project_id": "quota-project", "token_uri": format!("http://{address}/token")
            });
            if kind == "impersonated_service_account" {
                serde_json::json!({
                    "type": kind, "source_credentials": source, "quota_project_id": "quota-project",
                    "service_account_impersonation_url": format!("http://{address}/v1/projects/-/serviceAccounts/test-principal:generateAccessToken")
                })
            } else {
                source
            }
        };
        std::fs::write(file.path(), configured.to_string()).unwrap();
        let source = if inline {
            CredentialSource::Trusted(SecretValue::new(configured.to_string()))
        } else {
            CredentialSource::ApplicationCredentials(file.path().to_str().unwrap().into())
        };
        let provider = NativeProviderLoader.load(source).await.unwrap();

        assert_eq!(
            provider.project_id().await.ok().as_deref(),
            expected_project
        );
        let original = provider.token().await.unwrap();
        tokio::time::pause();
        tokio::time::advance(std::time::Duration::from_secs(3601)).await;
        tokio::time::resume();
        let refreshed = provider.token().await.unwrap();
        assert_ne!(original, refreshed);
        assert_eq!(requests.load(Ordering::SeqCst), expected_requests);
        server.abort();
    }

    #[rstest::rstest]
    #[case::canonical_endpoint(r#"{"token_uri":"https://oauth2.googleapis.com/token"}"#, true)]
    #[case::noncanonical_endpoint(r#"{"token_uri":"http://127.0.0.1/token"}"#, false)]
    #[case::missing_endpoint("{}", false)]
    fn request_credentials_require_canonical_token_endpoint(
        #[case] credentials: &str,
        #[case] accepted: bool,
    ) {
        assert_eq!(validate_request_credentials(credentials).is_ok(), accepted);
    }

    #[rstest::rstest]
    #[case::missing_file("/nonexistent/vertex-key.json", "Google credentials")]
    #[case::malformed_json(" {\"type\": \"service_account\",", "Google credentials")]
    #[case::quoted_json("'{\"type\": \"service_account\"}'", "Google credentials")]
    #[tokio::test]
    async fn trusted_credentials_are_read_as_a_file_unless_they_are_json(
        #[case] configured: &str,
        #[case] expected: &str,
    ) {
        let Err(error) = NativeProviderLoader
            .load(CredentialSource::Trusted(SecretValue::new(configured)))
            .await
        else {
            panic!("{configured:?} should not load");
        };
        let report = litellm_error::Report::new(&error).to_string();
        assert!(report.contains(expected), "{report}");
    }

    #[rstest::rstest]
    #[case::padded("eyJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCJ9")]
    #[case::quoted_prefixed_unpadded("\"base64:eyJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCJ9\"")]
    #[case::url_safe_wrapped("eyJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsICJr\nZXkiOiAiPz8_In0=")]
    #[tokio::test]
    async fn base64_encoded_credentials_are_named_as_base64(#[case] configured: &str) {
        assert!(
            trusted_load_error(configured)
                .await
                .contains("is base64-encoded JSON, not the JSON itself")
        );
    }

    async fn trusted_load_error(configured: &str) -> String {
        match NativeProviderLoader
            .load(CredentialSource::Trusted(SecretValue::new(configured)))
            .await
        {
            Ok(_) => panic!("{configured:?} should not load"),
            Err(error) => litellm_error::Report::new(&error).to_string(),
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn a_relative_key_file_named_like_json_is_read_as_a_file() {
        let key = tempfile::Builder::new()
            .prefix("{vertex")
            .suffix(".json")
            .tempfile_in(".")
            .unwrap();
        std::fs::write(key.path(), r#"{"project_id":"file-project"}"#).unwrap();
        let relative_name = key.path().file_name().unwrap().to_str().unwrap();
        let absolute_path = std::fs::canonicalize(key.path()).unwrap();
        assert_eq!(
            trusted_load_error(relative_name).await,
            trusted_load_error(absolute_path.to_str().unwrap()).await
        );
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn unknown_application_credential_types_are_rejected_without_service_account_fallback() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            r#"{"type":"unsupported","private_key":"credential-value"}"#,
        )
        .unwrap();
        let result = GcpAuth::default()
            .access_token(CredentialSource::ApplicationCredentials(
                file.path().to_str().unwrap().into(),
            ))
            .await;
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidConfiguration(_)));
        assert!(
            !litellm_error::Report::new(&error)
                .to_string()
                .contains("credential-value")
        );
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn gcloud_credentials_of_any_type_are_loaded_before_the_metadata_server() {
        let config_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            config_dir
                .path()
                .join("application_default_credentials.json"),
            r#"{"type":"unsupported"}"#,
        )
        .unwrap();
        let directory = config_dir.path().to_str().unwrap().to_string();
        let Err(error) = load_default_credentials(&|name| {
            (name == "CLOUDSDK_CONFIG").then(|| directory.clone())
        })
        .await
        else {
            panic!("an unsupported gcloud credential type should not load");
        };
        assert!(matches!(error, Error::InvalidConfiguration(_)));
    }

    #[rstest::rstest]
    fn explicit_aws_federation_keys_bypass_the_aws_metadata_service() {
        let configured = serde_json::json!({
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/aws",
            "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
            "credential_source": {"environment_id": "aws1"},
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret"
        });
        let Err(error) = load_credentials_json(&configured.to_string()) else {
            panic!("explicit AWS keys without aws_region_name should not load");
        };
        assert!(
            litellm_error::Report::new(&error)
                .to_string()
                .contains("aws_region_name")
        );
    }

    #[rstest::rstest]
    fn inline_and_trusted_credentials_never_share_a_cache_entry() {
        assert_ne!(
            CredentialSource::Inline(SecretValue::new("same-value")).cache_key(),
            CredentialSource::Trusted(SecretValue::new("same-value")).cache_key()
        );
    }
}
