use crate::{
    config::{CredentialSource, is_base64_encoded_json, names_a_credentials_file},
    constants::{CLOUD_PLATFORM_SCOPE, GOOGLE_OAUTH_TOKEN_ENDPOINT},
};
use gcp_auth::{CustomServiceAccount, TokenProvider};
use litellm_auth_types::{Error, ErrorDetail};
use litellm_secret_value::SecretValue;
use moka::future::Cache;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    time::{Duration, UNIX_EPOCH},
};

use google_cloud_auth::{
    build_errors::Error as BuildError,
    credentials::{
        AccessTokenCredentials, CacheableResource, Credentials, external_account, impersonated,
        mds, service_account, user_account,
    },
    errors::CredentialsError,
};
use http::Extensions;
use serde_json::Map;

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
        self.load_keyed(source.cache_key(), self.loader.load(source))
            .await
    }

    /// The token source cached under `key`, loaded by `load` when there is none. A caller that
    /// parses its own credentials into [`SdkCredentials`] loads them through this.
    pub(crate) async fn load_keyed(
        &self,
        key: CredentialCacheKey,
        load: impl Future<Output = Result<Arc<dyn GcpTokenSource>, Error>>,
    ) -> Result<Arc<dyn GcpTokenSource>, Error> {
        self.providers
            .try_get_with(key, load)
            .await
            .map_err(|error| (*error).clone())
    }

    /// `source`'s token. Google's Rust SDK stops refreshing credentials for good after a failure
    /// that is not transient, so such a failure drops `key` and the next call loads them anew.
    pub(crate) async fn token_evicting(
        &self,
        key: &CredentialCacheKey,
        source: &dyn GcpTokenSource,
    ) -> Result<SecretValue, Error> {
        let token = source.token().await;
        if token.as_ref().is_err_and(|error| {
            !litellm_error::find::<CredentialsError>(error)
                .is_some_and(CredentialsError::is_transient)
        }) {
            self.providers.invalidate(key).await;
        }
        token
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

pub(crate) struct ServiceAccountTokenSource(pub(crate) Arc<dyn TokenProvider>);

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

pub(crate) struct SdkTokenSource {
    pub(crate) credentials: google_cloud_auth::credentials::AccessTokenCredentials,
    pub(crate) project_id: Option<String>,
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

/// Credentials read by Google's Rust SDK from the headers it would send, for builders that only
/// build [`Credentials`].
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
        Box::pin(async {
            let CacheableResource::New { data, .. } = self
                .0
                .headers(Extensions::new())
                .await
                .map_err(auth_acquisition_error)?
            else {
                return Err(unresolved_token());
            };
            data.get(http::header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.strip_prefix("Bearer "))
                .map(SecretValue::new)
                .ok_or_else(unresolved_token)
        })
    }
}

fn unresolved_token() -> Error {
    Error::CredentialAcquisition(ErrorDetail::Unresolved {
        subject: "Google access token",
    })
}

/// A credentials JSON object kept whole for the builders of Google's Rust SDK, which read it
/// themselves.
#[derive(Clone, Default, PartialEq, veil::Redact)]
pub(crate) struct SdkJson(#[redact(with = "[REDACTED]")] pub(crate) Map<String, Value>);

impl SdkJson {
    /// The object without `quota_project_id`, which google-auth keeps whatever its JSON type and
    /// a login that only takes the token never needs.
    fn without_quota_project(self) -> Value {
        Value::Object(
            self.0
                .into_iter()
                .filter(|(field, _)| field != "quota_project_id")
                .collect(),
        )
    }

    /// The object with an empty string for each of `fields` that is absent or null.
    fn with_empty(self, fields: &[&str]) -> Value {
        let missing = fields
            .iter()
            .filter(|field| self.0.get(**field).is_none_or(Value::is_null))
            .map(|field| ((*field).to_owned(), Value::String(String::new())));
        Value::Object(self.0.clone().into_iter().chain(missing).collect())
    }
}

/// Google credentials a caller has already parsed, which the shared loader builds with Google's
/// Rust SDK scoped to `cloud-platform`.
pub(crate) enum SdkCredentials {
    /// A service account key, which signs its own JWT as the token. The SDK requires a
    /// `private_key_id` and a `project_id`, which google-auth does not, so either may be empty.
    ServiceAccount(SdkJson),
    /// An authorized user, refreshed at `token_uri` whatever endpoint its file names.
    AuthorizedUser {
        json: SdkJson,
        token_uri: String,
    },
    ExternalAccount(SdkJson),
    Impersonated(SdkJson),
    /// An external account whose subject token the caller supplies.
    Programmatic(external_account::ProgrammaticBuilder),
    /// Credentials of the caller's own impersonating a service account.
    ImpersonatedFrom(impersonated::Builder),
    /// The GCE metadata server.
    Metadata,
}

impl SdkCredentials {
    pub(crate) fn build(self) -> Result<Arc<dyn GcpTokenSource>, BuildError> {
        let access = |credentials: AccessTokenCredentials| {
            Arc::new(SdkTokenSource {
                credentials,
                project_id: None,
            }) as Arc<dyn GcpTokenSource>
        };
        let scopes = [CLOUD_PLATFORM_SCOPE];
        match self {
            Self::ServiceAccount(json) => {
                service_account::Builder::new(json.with_empty(&["private_key_id", "project_id"]))
                    .build_access_token_credentials()
                    .map(access)
            }
            Self::AuthorizedUser { json, token_uri } => {
                user_account::Builder::new(json.without_quota_project())
                    .with_token_uri(token_uri)
                    .with_scopes(scopes)
                    .build_access_token_credentials()
                    .map(access)
            }
            Self::ExternalAccount(json) => {
                external_account::Builder::new(json.without_quota_project())
                    .with_scopes(scopes)
                    .build_access_token_credentials()
                    .map(access)
            }
            Self::Impersonated(json) => impersonated::Builder::new(json.without_quota_project())
                .with_scopes(scopes)
                .build_access_token_credentials()
                .map(access),
            Self::Programmatic(builder) => builder
                .with_scopes(scopes)
                .build()
                .map(|credentials| Arc::new(HeaderTokenSource(credentials)) as _),
            Self::ImpersonatedFrom(builder) => builder
                .with_scopes(scopes)
                .build_access_token_credentials()
                .map(access),
            Self::Metadata => mds::Builder::default()
                .with_scopes(scopes)
                .build_access_token_credentials()
                .map(access),
        }
    }
}

/// A key file as it is on disk: `os.stat`'s inode (unix only), modification time and size, which
/// change when the file is rewritten or replaced.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct FileVersion {
    inode: Option<u64>,
    modified: Duration,
    size: u64,
}

impl FileVersion {
    fn of(path: &str) -> Option<Self> {
        let metadata = std::fs::metadata(path).ok()?;
        Some(Self {
            inode: inode(&metadata),
            modified: metadata
                .modified()
                .ok()
                .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
                .unwrap_or_default(),
            size: metadata.len(),
        })
    }
}

#[cfg(unix)]
fn inode(metadata: &std::fs::Metadata) -> Option<u64> {
    Some(std::os::unix::fs::MetadataExt::ino(metadata))
}

#[cfg(not(unix))]
fn inode(_: &std::fs::Metadata) -> Option<u64> {
    None
}

async fn load_application_credentials(path: &str) -> Result<Arc<dyn GcpTokenSource>, Error> {
    let contents = tokio::fs::read_to_string(path)
        .await
        .map_err(auth_acquisition_error)?;
    load_credentials_json(&contents)
}

fn load_credentials_json(contents: &str) -> Result<Arc<dyn GcpTokenSource>, Error> {
    let configured: Value = serde_json::from_str(contents).map_err(auth_acquisition_error)?;
    let kind = configured.get("type").and_then(Value::as_str);
    let project_id = configured
        .get("project_id")
        .and_then(Value::as_str)
        .map(str::to_string);
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
                    gcp_auth::provider().await.map_err(auth_acquisition_error)?
                }
            };
            Ok(Arc::new(ServiceAccountTokenSource(provider)) as Arc<dyn GcpTokenSource>)
        })
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
pub(crate) enum CredentialCacheKey {
    Inline([u8; 32]),
    Trusted([u8; 32]),
    ApplicationCredentials(String),
    Adc,
    /// A key file known by its version on disk, so a rotated file loads anew.
    KeyFile {
        path: String,
        version: FileVersion,
    },
}

impl CredentialCacheKey {
    /// Trusted credentials text, keyed by the version of the file it names when it names one.
    pub(crate) fn versioned(configured: &str) -> Self {
        match FileVersion::of(configured) {
            Some(version) => Self::KeyFile {
                path: configured.to_owned(),
                version,
            },
            None => Self::Trusted(Sha256::digest(configured).into()),
        }
    }

    /// The GCE metadata server's credentials.
    pub(crate) fn metadata() -> Self {
        Self::Adc
    }
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
    #[case::authorized_user("authorized_user", 2)]
    #[case::external_account("external_account", 2)]
    #[case::impersonated_service_account("impersonated_service_account", 4)]
    #[tokio::test]
    async fn explicit_application_credentials_refresh_expired_tokens(
        #[case] kind: &str,
        #[case] expected_requests: usize,
        #[values(false, true)] inline: bool,
    ) {
        use axum::{Json, Router, routing::post};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
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

        assert!(matches!(
            provider.project_id().await,
            Err(Error::CredentialAcquisition(ErrorDetail::Unresolved {
                subject: "GCP resource project"
            }))
        ));
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
    fn a_key_file_s_cache_key_changes_when_the_file_is_rewritten() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "{}").unwrap();
        let path = file.path().to_str().unwrap();
        let before = CredentialCacheKey::versioned(path);
        std::fs::write(file.path(), r#"{"rotated": true}"#).unwrap();
        assert!(matches!(before, CredentialCacheKey::KeyFile { .. }));
        assert_ne!(before, CredentialCacheKey::versioned(path));
    }

    #[rstest::rstest]
    fn a_key_file_s_cache_key_changes_when_only_its_modification_time_does() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "{}").unwrap();
        let path = file.path().to_str().unwrap();
        file.as_file()
            .set_modified(UNIX_EPOCH + Duration::from_secs(1_700_000_000))
            .unwrap();
        let before = CredentialCacheKey::versioned(path);
        file.as_file()
            .set_modified(UNIX_EPOCH + Duration::from_secs(1_700_000_060))
            .unwrap();
        assert!(matches!(before, CredentialCacheKey::KeyFile { .. }));
        assert_ne!(before, CredentialCacheKey::versioned(path));
    }

    #[rstest::rstest]
    fn credentials_text_that_names_no_file_is_keyed_by_its_text() {
        assert_eq!(
            CredentialCacheKey::versioned(r#"{"type":"service_account"}"#),
            CredentialCacheKey::Trusted(Sha256::digest(r#"{"type":"service_account"}"#).into())
        );
    }

    struct Refusing(bool);

    impl GcpTokenSource for Refusing {
        fn project_id(&self) -> GcpAuthFuture<'_, String> {
            Box::pin(async { Err(unresolved_token()) })
        }

        fn token(&self) -> GcpAuthFuture<'_, SecretValue> {
            Box::pin(async {
                Err(auth_acquisition_error(CredentialsError::from_msg(
                    self.0, "refused",
                )))
            })
        }
    }

    #[rstest::rstest]
    #[case::permanent(false, 2)]
    #[case::transient(true, 1)]
    #[tokio::test]
    async fn only_a_permanent_token_failure_drops_the_cached_source(
        #[case] transient: bool,
        #[case] loads: usize,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let auth = GcpAuth::default();
        let loaded = AtomicUsize::new(0);
        let load = || async {
            loaded.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(Refusing(transient)) as Arc<dyn GcpTokenSource>)
        };
        let key = CredentialCacheKey::metadata();
        let source = auth.load_keyed(key.clone(), load()).await.unwrap();
        assert!(auth.token_evicting(&key, &*source).await.is_err());
        auth.load_keyed(key, load()).await.unwrap();
        assert_eq!(loaded.load(Ordering::SeqCst), loads);
    }

    #[rstest::rstest]
    fn inline_and_trusted_credentials_never_share_a_cache_entry() {
        assert_ne!(
            CredentialSource::Inline(SecretValue::new("same-value")).cache_key(),
            CredentialSource::Trusted(SecretValue::new("same-value")).cache_key()
        );
    }
}
