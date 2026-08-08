//! User-directed exception: this base provider owns Google auth I/O for parity
//! with Python's `VertexBase`; the broader core purity guidance is reconciled
//! separately.

use std::env;
use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use google_cloud_auth::credentials::AccessTokenCredentials;
use google_cloud_auth::credentials::external_account;
use google_cloud_auth::credentials::service_account;
use google_cloud_auth::credentials::user_account;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::caching::in_memory_cache::InMemoryCache;
use crate::error::{CoreError, CoreResult};

use super::constants::{
    CLOUD_PLATFORM_SCOPE, GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_CLOUD_PROJECT,
    GOOGLE_CLOUD_QUOTA_PROJECT, VERTEXAI_CREDENTIALS, VERTEXAI_LOCATION, VERTEXAI_PROJECT,
};

static VERTEX_CREDENTIALS_CACHE: OnceLock<Mutex<InMemoryCache<AccessTokenCredentials>>> =
    OnceLock::new();

// google-cloud-auth owns async token caching and refresh; Python's single-flight machinery is intentionally not ported.

#[derive(Clone, Debug, PartialEq)]
pub enum VertexCredentialsInput {
    Json(Value),
    String(String),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VertexAuthConfig {
    pub credentials: Option<VertexCredentialsInput>,
    pub location: Option<String>,
    pub project_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VertexAuthSource {
    AwsWorkloadIdentity,
    ExecutableWorkloadIdentity,
    IdentityPoolWorkloadIdentity,
    AuthorizedUser,
    ServiceAccount,
    DefaultAdc,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexToken {
    pub access_token: String,
    pub project_id: String,
}

impl VertexAuthConfig {
    pub fn from_environment() -> Self {
        Self {
            credentials: non_empty_env(VERTEXAI_CREDENTIALS)
                .or_else(|| non_empty_env(GOOGLE_APPLICATION_CREDENTIALS))
                .map(VertexCredentialsInput::String),
            location: non_empty_env(VERTEXAI_LOCATION),
            project_id: non_empty_env(VERTEXAI_PROJECT)
                .or_else(|| non_empty_env(GOOGLE_CLOUD_PROJECT)),
        }
    }
}

pub fn resolve_credentials_input(
    credentials: Option<VertexCredentialsInput>,
) -> CoreResult<Option<Value>> {
    let Some(credentials) = credentials else {
        return Ok(None);
    };

    let value = match credentials {
        VertexCredentialsInput::Json(value) => value,
        VertexCredentialsInput::String(value) => {
            if value.trim().is_empty() {
                return Ok(None);
            }
            let path = Path::new(&value);
            let json = if path.is_file() {
                fs::read_to_string(path).map_err(|error| {
                    CoreError::Auth(format!("unable to read Vertex credentials file: {error}"))
                })?
            } else {
                value
            };
            serde_json::from_str(&json).map_err(|error| {
                CoreError::Auth(format!("invalid Vertex credentials JSON: {error}"))
            })?
        }
    };

    if value.is_object() {
        Ok(Some(value))
    } else {
        Err(CoreError::InvalidType {
            expected: "object",
            actual: crate::error::json_type_name(&value),
        })
    }
}

pub fn classify_auth(credentials: Option<&Value>) -> VertexAuthSource {
    let Some(credentials) = credentials else {
        return VertexAuthSource::DefaultAdc;
    };

    let credential_type = credentials.get("type").and_then(Value::as_str);
    if credential_type == Some("external_account") {
        let source = credentials
            .get("credential_source")
            .and_then(Value::as_object);
        let environment_id = source
            .and_then(|source| source.get("environment_id"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if environment_id.contains("aws") {
            return VertexAuthSource::AwsWorkloadIdentity;
        }
        if source.is_some_and(|source| source.contains_key("executable")) {
            return VertexAuthSource::ExecutableWorkloadIdentity;
        }
        return VertexAuthSource::IdentityPoolWorkloadIdentity;
    }
    if credential_type == Some("authorized_user") {
        return VertexAuthSource::AuthorizedUser;
    }
    VertexAuthSource::ServiceAccount
}

pub async fn resolve_token(config: VertexAuthConfig) -> CoreResult<VertexToken> {
    let credentials = resolve_credentials_input(config.credentials)?;
    let source = classify_auth(credentials.as_ref());
    let project_id = project_id(config.project_id, credentials.as_ref())?;
    let cache_key = cache_key(credentials.as_ref(), &project_id);
    let provider = cached_credentials(&cache_key)?;
    let provider = match provider {
        Some(provider) => provider,
        None => {
            let provider = build_credentials(source, credentials)?;
            let access_token = provider.access_token().await.map_err(|error| {
                CoreError::Auth(format!("Google token resolution failed: {error}"))
            })?;
            store_credentials(cache_key, provider.clone())?;
            return Ok(VertexToken {
                access_token: access_token.token,
                project_id,
            });
        }
    };
    let access_token = provider
        .access_token()
        .await
        .map_err(|error| CoreError::Auth(format!("Google token resolution failed: {error}")))?;
    Ok(VertexToken {
        access_token: access_token.token,
        project_id,
    })
}

fn build_credentials(
    source: VertexAuthSource,
    credentials: Option<Value>,
) -> CoreResult<AccessTokenCredentials> {
    let scopes = [CLOUD_PLATFORM_SCOPE];
    match source {
        VertexAuthSource::DefaultAdc => google_cloud_auth::credentials::Builder::default()
            .with_scopes(scopes)
            .build_access_token_credentials()
            .map_err(|error| CoreError::Auth(format!("Google ADC resolution failed: {error}"))),
        VertexAuthSource::AwsWorkloadIdentity
        | VertexAuthSource::ExecutableWorkloadIdentity
        | VertexAuthSource::IdentityPoolWorkloadIdentity => {
            let value = credentials.ok_or(CoreError::MissingField("credentials"))?;
            external_account::Builder::new(value)
                .with_scopes(scopes)
                .build_access_token_credentials()
                .map_err(|error| {
                    CoreError::Auth(format!(
                        "Google external account resolution failed: {error}"
                    ))
                })
        }
        VertexAuthSource::AuthorizedUser => {
            let value = credentials.ok_or(CoreError::MissingField("credentials"))?;
            user_account::Builder::new(value)
                .with_scopes(scopes)
                .build_access_token_credentials()
                .map_err(|error| {
                    CoreError::Auth(format!("Google authorized-user resolution failed: {error}"))
                })
        }
        VertexAuthSource::ServiceAccount => {
            let value = credentials.ok_or(CoreError::MissingField("credentials"))?;
            service_account::Builder::new(value)
                .with_access_specifier(service_account::AccessSpecifier::from_scopes(scopes))
                .build_access_token_credentials()
                .map_err(|error| {
                    CoreError::Auth(format!("Google service-account resolution failed: {error}"))
                })
        }
    }
}

fn project_id(explicit: Option<String>, credentials: Option<&Value>) -> CoreResult<String> {
    let project_id = explicit
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            credentials.and_then(|value| {
                value
                    .get("project_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
        })
        .or_else(|| {
            credentials.and_then(|value| {
                value
                    .get("quota_project_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
        })
        .or_else(|| non_empty_env(GOOGLE_CLOUD_PROJECT))
        .or_else(|| non_empty_env(GOOGLE_CLOUD_QUOTA_PROJECT));
    project_id.ok_or(CoreError::MissingField("project_id"))
}

fn cache_key(credentials: Option<&Value>, project_id: &str) -> String {
    let mut hasher = Sha256::new();
    if let Some(credentials) = credentials {
        hasher.update(credentials.to_string());
    }
    hasher.update(project_id);
    format!("{:x}", hasher.finalize())
}

fn cached_credentials(key: &str) -> CoreResult<Option<AccessTokenCredentials>> {
    let cache = VERTEX_CREDENTIALS_CACHE.get_or_init(|| Mutex::new(InMemoryCache::default()));
    let mut cache = cache
        .lock()
        .map_err(|_| CoreError::Auth("Vertex credential cache lock poisoned".to_string()))?;
    Ok(cache.get_cache(key))
}

fn store_credentials(key: String, credentials: AccessTokenCredentials) -> CoreResult<()> {
    let cache = VERTEX_CREDENTIALS_CACHE.get_or_init(|| Mutex::new(InMemoryCache::default()));
    let mut cache = cache
        .lock()
        .map_err(|_| CoreError::Auth("Vertex credential cache lock poisoned".to_string()))?;
    cache.set_cache(key, credentials, None);
    Ok(())
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classifier_preserves_external_account_dispatch_order() {
        assert_eq!(
            classify_auth(Some(&json!({
                "type": "external_account",
                "credential_source": {"environment_id": "aws1", "executable": {}}
            }))),
            VertexAuthSource::AwsWorkloadIdentity
        );
        assert_eq!(
            classify_auth(Some(&json!({
                "type": "external_account",
                "credential_source": {"executable": {"command": "token"}}
            }))),
            VertexAuthSource::ExecutableWorkloadIdentity
        );
        assert_eq!(
            classify_auth(Some(
                &json!({"type": "external_account", "credential_source": {}})
            )),
            VertexAuthSource::IdentityPoolWorkloadIdentity
        );
    }

    #[test]
    fn classifier_covers_authorized_service_and_default_sources() {
        assert_eq!(
            classify_auth(Some(&json!({"type": "authorized_user"}))),
            VertexAuthSource::AuthorizedUser
        );
        assert_eq!(
            classify_auth(Some(&json!({"type": "service_account"}))),
            VertexAuthSource::ServiceAccount
        );
        assert_eq!(
            classify_auth(Some(&json!({"client_email": "test@example.com"}))),
            VertexAuthSource::ServiceAccount
        );
        assert_eq!(classify_auth(None), VertexAuthSource::DefaultAdc);
    }

    #[test]
    fn credential_input_reads_files_inline_json_and_objects() {
        let path = std::env::temp_dir().join(format!("vertex-auth-{}.json", std::process::id()));
        std::fs::write(&path, r#"{"type":"authorized_user"}"#).unwrap();
        assert_eq!(
            resolve_credentials_input(Some(VertexCredentialsInput::String(
                path.to_string_lossy().into_owned()
            )))
            .unwrap(),
            Some(json!({"type": "authorized_user"}))
        );
        std::fs::remove_file(path).unwrap();
        assert_eq!(
            resolve_credentials_input(Some(VertexCredentialsInput::String(
                r#"{"type":"service_account"}"#.to_string()
            )))
            .unwrap(),
            Some(json!({"type": "service_account"}))
        );
        assert_eq!(
            resolve_credentials_input(Some(VertexCredentialsInput::Json(json!({
                "type": "external_account"
            }))))
            .unwrap(),
            Some(json!({"type": "external_account"}))
        );
    }

    #[test]
    fn credential_input_treats_empty_as_absent_and_rejects_invalid_json() {
        assert_eq!(
            resolve_credentials_input(Some(VertexCredentialsInput::String(" \n".to_string())))
                .unwrap(),
            None
        );
        assert!(matches!(
            resolve_credentials_input(Some(VertexCredentialsInput::String("{".to_string()))),
            Err(CoreError::Auth(_))
        ));
    }

    #[test]
    fn project_id_explicit_value_wins_over_credential_and_environment() {
        let credentials = json!({"project_id": "credential-project"});
        assert_eq!(
            project_id(Some("explicit-project".to_string()), Some(&credentials)).unwrap(),
            "explicit-project"
        );
        assert_eq!(
            project_id(None, Some(&credentials)).unwrap(),
            "credential-project"
        );
        assert_eq!(
            project_id(None, Some(&json!({"quota_project_id": "quota-project"}))).unwrap(),
            "quota-project"
        );
    }
}
