use std::path::Path;
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
pub struct VertexAuthInputs {
    credentials: Option<SecretValue>,
    project_id: Option<String>,
    location: Option<String>,
}

impl VertexAuthInputs {
    pub fn from_optional_params(params: &Map<String, Value>) -> Result<Self, AuthError> {
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

    pub fn project_id(&self) -> Option<&str> {
        self.project_id.as_deref()
    }

    pub fn location(&self) -> Option<&str> {
        self.location.as_deref()
    }
}

pub struct VertexAuthentication {
    pub headers: Vec<(String, String)>,
    pub project_id: String,
}

pub fn resolve_project_id(
    auth_inputs: &VertexAuthInputs,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    auth_inputs
        .project_id()
        .map(str::to_string)
        .or_else(|| non_empty_env(env_lookup, VERTEXAI_PROJECT_ENV))
}

pub fn resolve_location(
    auth_inputs: &VertexAuthInputs,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    auth_inputs
        .location()
        .map(str::to_string)
        .or_else(|| non_empty_env(env_lookup, VERTEXAI_LOCATION_ENV))
        .or_else(|| non_empty_env(env_lookup, VERTEX_LOCATION_ENV))
}

pub async fn authenticate(
    headers: Vec<(String, String)>,
    api_key: Option<&str>,
    auth_inputs: &VertexAuthInputs,
    project_id: Option<String>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
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
    let project_id = project_id.or_else(|| resolve_project_id(auth_inputs, env_lookup));
    let needs_provider = !has_authorization && static_token.is_none() || project_id.is_none();
    let provider = if needs_provider {
        Some(resolve_provider(auth_inputs, env_lookup).await?)
    } else {
        None
    };
    let project_id = match project_id {
        Some(project_id) => project_id,
        None => required_provider(provider.as_ref())?
            .project_id()
            .await
            .map_err(auth_acquisition_error)?
            .to_string(),
    };
    let headers = if has_authorization {
        headers
    } else if let Some(token) = static_token {
        apply_credential(headers, &token, CredentialPlacement::Bearer)?
    } else {
        let token = required_provider(provider.as_ref())?
            .token(&[CLOUD_PLATFORM_SCOPE])
            .await
            .map_err(auth_acquisition_error)?;
        apply_credential(headers, token.as_str(), CredentialPlacement::Bearer)?
    };

    Ok(VertexAuthentication {
        headers,
        project_id,
    })
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

fn required_provider(
    provider: Option<&Arc<dyn TokenProvider>>,
) -> Result<&Arc<dyn TokenProvider>, AuthError> {
    provider.ok_or(AuthError::MissingApiKey {
        provider: "Vertex AI",
    })
}

async fn resolve_provider(
    auth_inputs: &VertexAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Arc<dyn TokenProvider>, AuthError> {
    let configured = auth_inputs
        .credentials
        .as_ref()
        .map(SecretValue::expose)
        .map(str::to_string)
        .or_else(|| non_empty_env(env_lookup, VERTEXAI_CREDENTIALS_ENV));
    if let Some(configured) = configured {
        let service_account = if Path::new(&configured).is_file() {
            CustomServiceAccount::from_file(&configured)
        } else {
            CustomServiceAccount::from_json(&configured)
        }
        .map_err(auth_acquisition_error)?;
        return Ok(Arc::new(service_account));
    }

    if let Some(path) = non_empty_env(env_lookup, GOOGLE_APPLICATION_CREDENTIALS_ENV) {
        return CustomServiceAccount::from_file(path)
            .map(|provider| Arc::new(provider) as Arc<dyn TokenProvider>)
            .map_err(auth_acquisition_error);
    }

    gcp_auth::provider().await.map_err(auth_acquisition_error)
}

fn non_empty_env(env_lookup: &dyn Fn(&str) -> Option<String>, name: &str) -> Option<String> {
    env_lookup(name).filter(|value| !value.trim().is_empty())
}

fn auth_acquisition_error(error: gcp_auth::Error) -> AuthError {
    AuthError::VertexTokenAcquisition(error.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{VertexAuthInputs, authenticate};

    #[test]
    fn credentials_accept_json_objects_without_exposing_secrets() {
        let inputs = VertexAuthInputs::from_optional_params(
            json!({"vertex_credentials": {"private_key": "secret-key"}})
                .as_object()
                .expect("object"),
        )
        .expect("credentials parse");

        assert!(!format!("{inputs:?}").contains("secret-key"));
    }

    #[test]
    fn credentials_reject_non_json_values() {
        let error = VertexAuthInputs::from_optional_params(
            json!({"vertex_credentials": true})
                .as_object()
                .expect("object"),
        )
        .expect_err("boolean credentials are rejected");

        assert!(error.to_string().contains("vertex_credentials"));
    }

    #[test]
    fn routing_inputs_are_typed_separately_from_model_params() {
        let inputs = VertexAuthInputs::from_optional_params(
            json!({
                "vertex_project": "project-1",
                "vertex_location": "europe-west4"
            })
            .as_object()
            .expect("object"),
        )
        .expect("inputs parse");

        assert_eq!(inputs.project_id(), Some("project-1"));
        assert_eq!(inputs.location(), Some("europe-west4"));
    }

    #[tokio::test]
    async fn explicit_access_token_does_not_require_adc() {
        let authentication = authenticate(
            Vec::new(),
            Some("access-token"),
            &VertexAuthInputs::from_optional_params(
                json!({"vertex_project": "project-1"})
                    .as_object()
                    .expect("object"),
            )
            .expect("inputs parse"),
            None,
            &|_| None,
        )
        .await
        .expect("static token authenticates");

        assert_eq!(authentication.project_id, "project-1");
        assert_eq!(
            authentication.headers,
            vec![(
                "Authorization".to_string(),
                "Bearer access-token".to_string()
            )]
        );
    }
}
