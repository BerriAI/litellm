use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use gcp_auth::{CustomServiceAccount, TokenProvider};
use serde_json::{Map, Value};

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
pub(crate) struct VertexAuthInputs {
    credentials: Option<SecretValue>,
    project_id: Option<String>,
    location: Option<String>,
}

impl VertexAuthInputs {
    pub(crate) fn from_optional_params(params: &Map<String, Value>) -> Result<Self, AuthError> {
        let credentials = params
            .get("vertex_credentials")
            .or_else(|| params.get("vertex_ai_credentials"));
        let credentials = match credentials {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) => Some(SecretValue::new(value)),
            Some(Value::Object(value)) => Some(SecretValue::new(
                serde_json::to_string(value).map_err(|error| {
                    AuthError::Configuration(AuthConfigurationError::InvalidFieldType(format!(
                        "vertex_credentials: {error}"
                    )))
                })?,
            )),
            Some(_) => {
                return Err(AuthError::Configuration(
                    AuthConfigurationError::InvalidFieldType("vertex_credentials".to_string()),
                ));
            }
        };
        Ok(Self {
            credentials,
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

pub(crate) struct VertexAuthentication {
    pub headers: Vec<(String, String)>,
    pub project_id: String,
}

pub(crate) fn resolve_project_id(
    inputs: &VertexAuthInputs,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    inputs
        .project_id()
        .map(str::to_string)
        .or_else(|| non_empty_env(env_lookup, VERTEXAI_PROJECT_ENV))
}

pub(crate) fn resolve_location(
    inputs: &VertexAuthInputs,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    inputs
        .location()
        .map(str::to_string)
        .or_else(|| non_empty_env(env_lookup, VERTEXAI_LOCATION_ENV))
        .or_else(|| non_empty_env(env_lookup, VERTEX_LOCATION_ENV))
}

pub(crate) async fn authenticate(
    headers: Vec<(String, String)>,
    api_key: Option<&str>,
    inputs: &VertexAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<VertexAuthentication, AuthError> {
    authenticate_with_provider(headers, api_key, inputs, env_lookup, None).await
}

trait VertexTokenSource: Send + Sync {
    fn project_id(&self) -> VertexAuthFuture<'_, String>;
    fn token(&self) -> VertexAuthFuture<'_, String>;
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

async fn authenticate_with_provider(
    headers: Vec<(String, String)>,
    api_key: Option<&str>,
    inputs: &VertexAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    injected_provider: Option<Arc<dyn VertexTokenSource>>,
) -> Result<VertexAuthentication, AuthError> {
    let has_authorization = headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("Authorization"));
    let static_token = api_key
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| non_empty_env(env_lookup, VERTEX_AI_API_KEY_ENV))
        .or_else(|| non_empty_env(env_lookup, VERTEXAI_API_KEY_ENV));
    let project_id = resolve_project_id(inputs, env_lookup);
    let needs_provider = (!has_authorization && static_token.is_none()) || project_id.is_none();
    let provider = if needs_provider {
        Some(match injected_provider {
            Some(provider) => provider,
            None => load_provider(inputs, env_lookup).await?,
        })
    } else {
        None
    };
    let project_id = match project_id {
        Some(project_id) => project_id,
        None => required_provider(provider.as_ref())?.project_id().await?,
    };
    let headers = if has_authorization {
        headers
    } else if let Some(token) = static_token {
        apply_credential(headers, &token, CredentialPlacement::Bearer)?
    } else {
        let token = required_provider(provider.as_ref())?.token().await?;
        apply_credential(headers, &token, CredentialPlacement::Bearer)?
    };
    Ok(VertexAuthentication {
        headers,
        project_id,
    })
}

fn required_provider(
    provider: Option<&Arc<dyn VertexTokenSource>>,
) -> Result<&Arc<dyn VertexTokenSource>, AuthError> {
    provider.ok_or(AuthError::MissingApiKey {
        provider: "Vertex AI",
    })
}

async fn load_provider(
    inputs: &VertexAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Arc<dyn VertexTokenSource>, AuthError> {
    let provider: Arc<dyn TokenProvider> = match credential_source(inputs, env_lookup) {
        CredentialSource::Configured(configured) => {
            let service_account = if Path::new(&configured).is_file() {
                CustomServiceAccount::from_file(&configured)
            } else {
                CustomServiceAccount::from_json(&configured)
            }
            .map_err(auth_acquisition_error)?;
            Arc::new(service_account)
        }
        CredentialSource::ApplicationCredentials(path) => {
            Arc::new(CustomServiceAccount::from_file(path).map_err(auth_acquisition_error)?)
        }
        CredentialSource::Adc => gcp_auth::provider().await.map_err(auth_acquisition_error)?,
    };
    Ok(Arc::new(GcpTokenSource(provider)))
}

#[derive(Debug, PartialEq, Eq)]
enum CredentialSource {
    Configured(String),
    ApplicationCredentials(String),
    Adc,
}

fn credential_source(
    inputs: &VertexAuthInputs,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CredentialSource {
    let configured = inputs
        .credentials
        .as_ref()
        .map(SecretValue::expose)
        .map(str::to_string)
        .or_else(|| non_empty_env(env_lookup, VERTEXAI_CREDENTIALS_ENV));
    if let Some(configured) = configured {
        return CredentialSource::Configured(configured);
    }
    non_empty_env(env_lookup, GOOGLE_APPLICATION_CREDENTIALS_ENV)
        .map(CredentialSource::ApplicationCredentials)
        .unwrap_or(CredentialSource::Adc)
}

fn optional_string(
    params: &Map<String, Value>,
    names: &[&str],
) -> Result<Option<String>, AuthError> {
    let value = names.iter().find_map(|name| params.get(*name));
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        Some(Value::String(_)) => Ok(None),
        Some(_) => Err(AuthError::Configuration(
            AuthConfigurationError::InvalidFieldType(names[0].to_string()),
        )),
    }
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

    fn inputs(value: Value) -> VertexAuthInputs {
        VertexAuthInputs::from_optional_params(value.as_object().unwrap()).unwrap()
    }

    #[test]
    fn inputs_are_typed_and_secrets_are_redacted() {
        let inputs = inputs(json!({
            "vertex_credentials":{"private_key":"secret-key"},
            "vertex_project":"project-1",
            "vertex_location":"europe-west4"
        }));
        assert_eq!(inputs.project_id(), Some("project-1"));
        assert_eq!(inputs.location(), Some("europe-west4"));
        assert!(!format!("{inputs:?}").contains("secret-key"));
        assert!(
            VertexAuthInputs::from_optional_params(
                json!({"vertex_credentials":true}).as_object().unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn project_and_location_prefer_input_then_environment() {
        let configured =
            inputs(json!({"vertex_project":"input-project","vertex_location":"input-location"}));
        let env = |name: &str| Some(format!("env-{name}"));
        assert_eq!(
            resolve_project_id(&configured, &env).as_deref(),
            Some("input-project")
        );
        assert_eq!(
            resolve_location(&configured, &env).as_deref(),
            Some("input-location")
        );
        let empty = VertexAuthInputs::default();
        assert_eq!(
            resolve_project_id(&empty, &|_| Some("env-project".into())).as_deref(),
            Some("env-project")
        );
        assert_eq!(
            resolve_location(&empty, &|name| (name == VERTEX_LOCATION_ENV)
                .then(|| "fallback-location".into()))
            .as_deref(),
            Some("fallback-location")
        );
    }

    #[test]
    fn credential_discovery_prefers_input_then_environment_then_adc() {
        let configured = inputs(json!({"vertex_credentials":"input-json"}));
        assert_eq!(
            credential_source(&configured, &|_| Some("environment-value".into())),
            CredentialSource::Configured("input-json".into())
        );
        let empty = VertexAuthInputs::default();
        assert_eq!(
            credential_source(&empty, &|name| {
                (name == VERTEXAI_CREDENTIALS_ENV).then(|| "environment-json".into())
            }),
            CredentialSource::Configured("environment-json".into())
        );
        assert_eq!(
            credential_source(&empty, &|name| {
                (name == GOOGLE_APPLICATION_CREDENTIALS_ENV).then(|| "adc.json".into())
            }),
            CredentialSource::ApplicationCredentials("adc.json".into())
        );
        assert_eq!(credential_source(&empty, &|_| None), CredentialSource::Adc);
    }

    #[tokio::test]
    async fn explicit_token_and_header_do_not_acquire_adc() {
        let configured = inputs(json!({"vertex_project":"project-1"}));
        let explicit = authenticate_with_provider(
            Vec::new(),
            Some("access-token"),
            &configured,
            &|_| None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(explicit.headers[0].1, "Bearer access-token");
        let existing = authenticate_with_provider(
            vec![("authorization".into(), "Bearer existing".into())],
            None,
            &configured,
            &|_| None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(existing.headers[0].1, "Bearer existing");
    }

    #[tokio::test]
    async fn injected_adc_supplies_missing_token_and_project() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider: Arc<dyn VertexTokenSource> = Arc::new(FakeProvider {
            calls: calls.clone(),
        });
        let authentication = authenticate_with_provider(
            Vec::new(),
            None,
            &VertexAuthInputs::default(),
            &|_| None,
            Some(provider),
        )
        .await
        .unwrap();
        assert_eq!(authentication.project_id, "adc-project");
        assert_eq!(authentication.headers[0].1, "Bearer adc-token");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
