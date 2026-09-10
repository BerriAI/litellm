use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use gcp_auth::{CustomServiceAccount, TokenProvider};
use moka::future::Cache;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::auth::error::AuthConfigurationError;
use crate::auth::http::apply_credential;
use crate::auth::{AuthError, CredentialPlacement, SecretValue};

const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const GOOGLE_APPLICATION_CREDENTIALS_ENV: &str = "GOOGLE_APPLICATION_CREDENTIALS";
const VERTEX_AI_API_KEY_ENV: &str = "VERTEX_AI_API_KEY";
const VERTEXAI_API_KEY_ENV: &str = "VERTEXAI_API_KEY";
const VERTEXAI_CREDENTIALS_ENV: &str = "VERTEXAI_CREDENTIALS";
const VERTEXAI_PROJECT_ENV: &str = "VERTEXAI_PROJECT";
const VERTEXAI_LOCATION_ENV: &str = "VERTEXAI_LOCATION";
const VERTEX_LOCATION_ENV: &str = "VERTEX_LOCATION";

#[derive(Clone, Debug, Default)]
pub(crate) struct VertexConfig {
    credentials: Option<SecretValue>,
    project_id: Option<String>,
    location: Option<String>,
}

impl VertexConfig {
    pub(crate) fn from_optional_params(params: &Map<String, Value>) -> Result<Self, AuthError> {
        Ok(Self {
            credentials: optional_credentials(
                params,
                &["vertex_credentials", "vertex_ai_credentials"],
            )?,
            project_id: optional_string(params, &["vertex_project", "vertex_ai_project"])?,
            location: optional_string(params, &["vertex_location", "vertex_ai_location"])?,
        })
    }

    pub(crate) fn project_id(&self) -> Option<&str> {
        self.project_id.as_deref()
    }

    pub(crate) fn location(&self) -> Option<&str> {
        self.location.as_deref()
    }
}

pub(crate) struct VertexEnvironment {
    pub headers: Vec<(String, String)>,
    pub project_id: String,
}

struct VertexAccessToken {
    token: String,
    project_id: String,
}

pub(crate) fn get_vertex_ai_project(
    config: &VertexConfig,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    config
        .project_id()
        .map(str::to_string)
        .or_else(|| non_empty_env(env_lookup, VERTEXAI_PROJECT_ENV))
}

pub(crate) fn get_vertex_ai_location(
    config: &VertexConfig,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    config
        .location()
        .map(str::to_string)
        .or_else(|| non_empty_env(env_lookup, VERTEXAI_LOCATION_ENV))
        .or_else(|| non_empty_env(env_lookup, VERTEX_LOCATION_ENV))
}

#[derive(Clone)]
pub(crate) struct VertexAuth {
    providers: Cache<CredentialCacheKey, Arc<dyn VertexTokenSource>>,
    loader: Arc<dyn VertexProviderLoader>,
}

impl Default for VertexAuth {
    fn default() -> Self {
        Self::new(Arc::new(GcpProviderLoader))
    }
}

impl VertexAuth {
    fn new(loader: Arc<dyn VertexProviderLoader>) -> Self {
        Self {
            providers: Cache::builder().max_capacity(64).build(),
            loader,
        }
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    pub(crate) async fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        config: &VertexConfig,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<VertexEnvironment, AuthError> {
        let has_authorization = headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("Authorization"));
        let static_token = api_key
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .or_else(|| non_empty_env(env_lookup, VERTEX_AI_API_KEY_ENV))
            .or_else(|| non_empty_env(env_lookup, VERTEXAI_API_KEY_ENV));
        let project_id = get_vertex_ai_project(config, env_lookup);

        if !has_authorization && static_token.is_none() {
            let access = self.get_access_token(config, env_lookup).await?;
            return Ok(VertexEnvironment {
                headers: apply_credential(headers, &access.token, CredentialPlacement::Bearer)?,
                project_id: project_id.unwrap_or(access.project_id),
            });
        }

        let project_id = match project_id {
            Some(project_id) => project_id,
            None => {
                self.load_provider(config, env_lookup)
                    .await?
                    .project_id()
                    .await?
            }
        };
        let headers = if has_authorization {
            headers
        } else {
            apply_credential(
                headers,
                static_token.as_deref().expect("static token was checked"),
                CredentialPlacement::Bearer,
            )?
        };
        Ok(VertexEnvironment {
            headers,
            project_id,
        })
    }

    async fn get_access_token(
        &self,
        config: &VertexConfig,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<VertexAccessToken, AuthError> {
        let provider = self.load_provider(config, env_lookup).await?;
        let (token, project_id) = tokio::try_join!(provider.token(), provider.project_id())?;
        Ok(VertexAccessToken { token, project_id })
    }

    async fn load_provider(
        &self,
        config: &VertexConfig,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Arc<dyn VertexTokenSource>, AuthError> {
        let source = credential_source(config, env_lookup);
        let key = source.cache_key();
        self.providers
            .try_get_with(key, self.loader.load(source))
            .await
            .map_err(|error| (*error).clone())
    }
}

trait VertexTokenSource: Send + Sync {
    fn project_id(&self) -> VertexAuthFuture<'_, String>;
    fn token(&self) -> VertexAuthFuture<'_, String>;
}

trait VertexProviderLoader: Send + Sync {
    fn load(&self, source: CredentialSource) -> VertexAuthFuture<'_, Arc<dyn VertexTokenSource>>;
}

type VertexAuthFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, AuthError>> + Send + 'a>>;

struct GcpTokenSource(Arc<dyn TokenProvider>);

impl VertexTokenSource for GcpTokenSource {
    fn project_id(&self) -> VertexAuthFuture<'_, String> {
        Box::pin(async move {
            self.0
                .project_id()
                .await
                .map(|project| project.to_string())
                .map_err(auth_acquisition_error)
        })
    }

    fn token(&self) -> VertexAuthFuture<'_, String> {
        Box::pin(async move {
            self.0
                .token(&[CLOUD_PLATFORM_SCOPE])
                .await
                .map(|token| token.as_str().to_string())
                .map_err(auth_acquisition_error)
        })
    }
}

struct GcpProviderLoader;

impl VertexProviderLoader for GcpProviderLoader {
    fn load(&self, source: CredentialSource) -> VertexAuthFuture<'_, Arc<dyn VertexTokenSource>> {
        Box::pin(async move {
            let provider: Arc<dyn TokenProvider> = match source {
                CredentialSource::Configured(configured) => {
                    let configured = configured.expose();
                    let service_account = if Path::new(configured).is_file() {
                        CustomServiceAccount::from_file(configured)
                    } else {
                        CustomServiceAccount::from_json(configured)
                    }
                    .map_err(auth_acquisition_error)?;
                    Arc::new(service_account)
                }
                CredentialSource::ApplicationCredentials(path) => {
                    Arc::new(CustomServiceAccount::from_file(path).map_err(auth_acquisition_error)?)
                }
                CredentialSource::Adc => {
                    gcp_auth::provider().await.map_err(auth_acquisition_error)?
                }
            };
            Ok(Arc::new(GcpTokenSource(provider)) as Arc<dyn VertexTokenSource>)
        })
    }
}

#[derive(Clone, Debug)]
enum CredentialSource {
    Configured(SecretValue),
    ApplicationCredentials(String),
    Adc,
}

impl CredentialSource {
    fn cache_key(&self) -> CredentialCacheKey {
        match self {
            Self::Configured(configured) => {
                CredentialCacheKey::Configured(Sha256::digest(configured.expose()).into())
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
    Configured([u8; 32]),
    ApplicationCredentials(String),
    Adc,
}

fn credential_source(
    config: &VertexConfig,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CredentialSource {
    let configured = config
        .credentials
        .clone()
        .or_else(|| non_empty_env(env_lookup, VERTEXAI_CREDENTIALS_ENV).map(SecretValue::new));
    if let Some(configured) = configured {
        return CredentialSource::Configured(configured);
    }
    non_empty_env(env_lookup, GOOGLE_APPLICATION_CREDENTIALS_ENV)
        .map(CredentialSource::ApplicationCredentials)
        .unwrap_or(CredentialSource::Adc)
}

fn optional_credentials(
    params: &Map<String, Value>,
    names: &[&str],
) -> Result<Option<SecretValue>, AuthError> {
    for name in names {
        match params.get(*name) {
            None | Some(Value::Null) => continue,
            Some(Value::String(value)) if value.trim().is_empty() => continue,
            Some(Value::String(value)) => return Ok(Some(SecretValue::new(value))),
            Some(Value::Object(value)) if value.is_empty() => continue,
            Some(Value::Object(value)) => {
                return serde_json::to_string(value)
                    .map(SecretValue::new)
                    .map(Some)
                    .map_err(|error| {
                        AuthError::Configuration(AuthConfigurationError::InvalidFieldType(format!(
                            "{}: {error}",
                            names[0]
                        )))
                    });
            }
            Some(_) => {
                return Err(AuthError::Configuration(
                    AuthConfigurationError::InvalidFieldType(names[0].to_string()),
                ));
            }
        }
    }
    Ok(None)
}

fn optional_string(
    params: &Map<String, Value>,
    names: &[&str],
) -> Result<Option<String>, AuthError> {
    for name in names {
        match params.get(*name) {
            None | Some(Value::Null) => continue,
            Some(Value::String(value)) if value.trim().is_empty() => continue,
            Some(Value::String(value)) => return Ok(Some(value.clone())),
            Some(_) => {
                return Err(AuthError::Configuration(
                    AuthConfigurationError::InvalidFieldType(names[0].to_string()),
                ));
            }
        }
    }
    Ok(None)
}

fn non_empty_env(env_lookup: &dyn Fn(&str) -> Option<String>, name: &str) -> Option<String> {
    env_lookup(name)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn auth_acquisition_error(error: gcp_auth::Error) -> AuthError {
    AuthError::VertexTokenAcquisition(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use super::*;

    struct FakeProvider {
        calls: Arc<AtomicUsize>,
    }

    impl VertexTokenSource for FakeProvider {
        fn project_id(&self) -> VertexAuthFuture<'_, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok("adc-project".into()) })
        }

        fn token(&self) -> VertexAuthFuture<'_, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok("adc-token".into()) })
        }
    }

    struct FakeLoader {
        loads: Arc<AtomicUsize>,
        provider: Arc<dyn VertexTokenSource>,
    }

    impl VertexProviderLoader for FakeLoader {
        fn load(
            &self,
            _source: CredentialSource,
        ) -> VertexAuthFuture<'_, Arc<dyn VertexTokenSource>> {
            let loads = self.loads.clone();
            let provider = self.provider.clone();
            Box::pin(async move {
                loads.fetch_add(1, Ordering::SeqCst);
                Ok(provider)
            })
        }
    }

    fn config(value: Value) -> VertexConfig {
        VertexConfig::from_optional_params(value.as_object().unwrap()).unwrap()
    }

    fn auth(calls: Arc<AtomicUsize>, loads: Arc<AtomicUsize>) -> VertexAuth {
        let provider: Arc<dyn VertexTokenSource> = Arc::new(FakeProvider { calls });
        VertexAuth::new(Arc::new(FakeLoader { loads, provider }))
    }

    #[test]
    fn config_is_typed_and_secrets_are_redacted() {
        let config = config(json!({
            "vertex_credentials":{"private_key":"secret-key"},
            "vertex_project":"project-1",
            "vertex_location":"europe-west4"
        }));
        assert_eq!(config.project_id(), Some("project-1"));
        assert_eq!(config.location(), Some("europe-west4"));
        assert!(!format!("{config:?}").contains("secret-key"));
        assert!(
            VertexConfig::from_optional_params(
                json!({"vertex_credentials":true}).as_object().unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn empty_primary_values_fall_back_to_python_aliases() {
        let config = config(json!({
            "vertex_credentials": null,
            "vertex_ai_credentials": "alias-credentials",
            "vertex_project": " ",
            "vertex_ai_project": "alias-project",
            "vertex_location": null,
            "vertex_ai_location": "alias-location"
        }));
        assert_eq!(
            config.credentials.as_ref().unwrap().expose(),
            "alias-credentials"
        );
        assert_eq!(config.project_id(), Some("alias-project"));
        assert_eq!(config.location(), Some("alias-location"));
    }

    #[test]
    fn project_and_location_prefer_input_then_environment() {
        let configured =
            config(json!({"vertex_project":"input-project","vertex_location":"input-location"}));
        let env = |name: &str| Some(format!("env-{name}"));
        assert_eq!(
            get_vertex_ai_project(&configured, &env).as_deref(),
            Some("input-project")
        );
        assert_eq!(
            get_vertex_ai_location(&configured, &env).as_deref(),
            Some("input-location")
        );
        let empty = VertexConfig::default();
        assert_eq!(
            get_vertex_ai_project(&empty, &|_| Some("env-project".into())).as_deref(),
            Some("env-project")
        );
        assert_eq!(
            get_vertex_ai_location(&empty, &|name| (name == VERTEX_LOCATION_ENV)
                .then(|| "fallback-location".into()))
            .as_deref(),
            Some("fallback-location")
        );
    }

    #[test]
    fn credential_discovery_prefers_input_then_environment_then_adc() {
        let configured = config(json!({"vertex_credentials":"input-json"}));
        assert!(
            matches!(credential_source(&configured, &|_| Some("environment-value".into())), CredentialSource::Configured(value) if value.expose() == "input-json")
        );
        let empty = VertexConfig::default();
        assert!(
            matches!(credential_source(&empty, &|name| (name == VERTEXAI_CREDENTIALS_ENV).then(|| "environment-json".into())), CredentialSource::Configured(value) if value.expose() == "environment-json")
        );
        assert!(
            matches!(credential_source(&empty, &|name| (name == GOOGLE_APPLICATION_CREDENTIALS_ENV).then(|| "adc.json".into())), CredentialSource::ApplicationCredentials(path) if path == "adc.json")
        );
        assert!(matches!(
            credential_source(&empty, &|_| None),
            CredentialSource::Adc
        ));
    }

    #[tokio::test]
    async fn explicit_token_and_header_do_not_acquire_adc() {
        let loads = Arc::new(AtomicUsize::new(0));
        let auth = auth(Arc::new(AtomicUsize::new(0)), loads.clone());
        let configured = config(json!({"vertex_project":"project-1"}));
        let explicit = auth
            .validate_environment(Vec::new(), Some("access-token"), &configured, &|_| None)
            .await
            .unwrap();
        assert_eq!(explicit.headers[0].1, "Bearer access-token");
        let existing = auth
            .validate_environment(
                vec![("authorization".into(), "Bearer existing".into())],
                None,
                &configured,
                &|_| None,
            )
            .await
            .unwrap();
        assert_eq!(existing.headers[0].1, "Bearer existing");
        assert_eq!(loads.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn provider_is_reused_across_authentication_calls() {
        let calls = Arc::new(AtomicUsize::new(0));
        let loads = Arc::new(AtomicUsize::new(0));
        let auth = auth(calls.clone(), loads.clone());
        for _ in 0..2 {
            let environment = auth
                .validate_environment(Vec::new(), None, &VertexConfig::default(), &|_| None)
                .await
                .unwrap();
            assert_eq!(environment.project_id, "adc-project");
            assert_eq!(environment.headers[0].1, "Bearer adc-token");
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }
}
