use std::{collections::BTreeMap, path::Path};

use litellm_auth_types::{Error, ErrorDetail, InputSource, SecretValue, Sourced};
use serde_json::{Map, Value};

use litellm_auth_types::VertexParams;

use crate::constants::GOOGLE_APPLICATION_CREDENTIALS_ENV;

#[derive(Clone, Debug, Default)]
pub struct VertexConfig {
    credentials: Option<Sourced<SecretValue>>,
    project_id: Option<String>,
    location: Option<String>,
}

impl VertexConfig {
    pub fn new(
        credentials: Option<Sourced<SecretValue>>,
        project_id: Option<String>,
        location: Option<String>,
    ) -> Self {
        Self {
            credentials: credentials.filter(|value| !value.value().expose().trim().is_empty()),
            project_id: project_id.filter(|value| !value.trim().is_empty()),
            location: location.filter(|value| !value.trim().is_empty()),
        }
    }

    pub fn from_sourced_optional_params(
        params: &Map<String, Value>,
        sources: &BTreeMap<String, InputSource>,
    ) -> Result<Self, Error> {
        Ok(Self::new(
            optional_credentials(params, sources, VertexParams::CREDENTIALS.wire)?,
            optional_string(params, VertexParams::PROJECT.wire)?,
            optional_string(params, VertexParams::LOCATION.wire)?,
        ))
    }

    /// The deployment's `litellm_params`, which a host already trusts the way it trusts its
    /// own configuration.
    pub fn from_params(params: &VertexParams) -> Self {
        Self::new(
            params
                .credentials()
                .map(|value| Sourced::new(SecretValue::new(value), InputSource::Deployment)),
            params.project(),
            params.location(),
        )
    }

    pub fn project_id(&self) -> Option<&str> {
        self.project_id.as_deref()
    }

    pub fn location(&self) -> Option<&str> {
        self.location.as_deref()
    }
}

pub fn get_vertex_ai_project(
    config: &VertexConfig,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    config
        .project_id()
        .map(str::to_string)
        .or_else(|| VertexParams::PROJECT.resolve(&|_| None, env_lookup))
}

/// The project a request is billed to when nothing names it: the `project_id` inside the
/// service account the call would authenticate with. Application default credentials carry
/// no project that can be read without a token exchange, so they resolve to `None`.
pub fn get_vertex_ai_project_from_credentials(
    config: &VertexConfig,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    let text = match credential_source(config, env_lookup) {
        CredentialSource::Inline(configured) => configured.expose().to_string(),
        CredentialSource::Trusted(configured) => {
            let configured = configured.expose();
            match std::fs::read_to_string(configured) {
                Ok(contents) if Path::new(configured).is_file() => contents,
                _ => configured.to_string(),
            }
        }
        CredentialSource::ApplicationCredentials(path) => std::fs::read_to_string(path).ok()?,
        CredentialSource::Adc => return None,
    };
    serde_json::from_str::<Value>(&text)
        .ok()?
        .get("project_id")
        .and_then(Value::as_str)
        .map(str::to_string)
}

pub fn get_vertex_ai_location(
    config: &VertexConfig,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    config
        .location()
        .map(str::to_string)
        .or_else(|| VertexParams::LOCATION.resolve(&|_| None, env_lookup))
}

#[derive(Clone, Debug)]
pub enum CredentialSource {
    Inline(SecretValue),
    Trusted(SecretValue),
    ApplicationCredentials(String),
    Adc,
}

pub(crate) fn credential_source(
    config: &VertexConfig,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CredentialSource {
    if let Some(configured) = config.credentials.clone() {
        return match configured.source() {
            InputSource::Request => CredentialSource::Inline(configured.into_value()),
            InputSource::Deployment | InputSource::Environment => {
                CredentialSource::Trusted(configured.into_value())
            }
        };
    }
    if let Some(configured) = VertexParams::CREDENTIALS.resolve(&|_| None, env_lookup) {
        return CredentialSource::Trusted(SecretValue::new(configured));
    }
    non_empty_env(env_lookup, GOOGLE_APPLICATION_CREDENTIALS_ENV)
        .map(CredentialSource::ApplicationCredentials)
        .unwrap_or(CredentialSource::Adc)
}

fn optional_credentials(
    params: &Map<String, Value>,
    sources: &BTreeMap<String, InputSource>,
    names: &[&str],
) -> Result<Option<Sourced<SecretValue>>, Error> {
    for name in names {
        let source = source_for(sources, name);
        match params.get(*name) {
            None | Some(Value::Null) => continue,
            Some(Value::String(value)) if value.trim().is_empty() => continue,
            Some(Value::String(value)) => {
                return Ok(Some(Sourced::new(SecretValue::new(value), source)));
            }
            Some(Value::Object(value)) if value.is_empty() => continue,
            Some(Value::Object(value)) => {
                return serde_json::to_string(value)
                    .map(SecretValue::new)
                    .map(|value| Sourced::new(value, source))
                    .map(Some)
                    .map_err(|error| {
                        Error::InvalidConfiguration(ErrorDetail::failed(
                            "credential serialization",
                            error,
                        ))
                    });
            }
            Some(_) => {
                return Err(Error::InvalidConfiguration(ErrorDetail::InvalidType {
                    field: names[0].into(),
                    expected: "a string or null",
                }));
            }
        }
    }
    Ok(None)
}

fn source_for(sources: &BTreeMap<String, InputSource>, name: &str) -> InputSource {
    sources.get(name).copied().unwrap_or_default()
}

fn optional_string(params: &Map<String, Value>, names: &[&str]) -> Result<Option<String>, Error> {
    for name in names {
        match params.get(*name) {
            None | Some(Value::Null) => continue,
            Some(Value::String(value)) if value.trim().is_empty() => continue,
            Some(Value::String(value)) => return Ok(Some(value.clone())),
            Some(_) => {
                return Err(Error::InvalidConfiguration(ErrorDetail::InvalidType {
                    field: names[0].into(),
                    expected: "a string or null",
                }));
            }
        }
    }
    Ok(None)
}

pub(crate) fn non_empty_env(
    env_lookup: &dyn Fn(&str) -> Option<String>,
    name: &str,
) -> Option<String> {
    env_lookup(name)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::constants::VERTEXAI_CREDENTIALS_ENV;

    fn config(value: Value) -> VertexConfig {
        VertexConfig::from_sourced_optional_params(value.as_object().unwrap(), &BTreeMap::new())
            .unwrap()
    }

    #[rstest::rstest]
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
            config.credentials.as_ref().unwrap().value().expose(),
            "alias-credentials"
        );
        assert_eq!(config.project_id(), Some("alias-project"));
        assert_eq!(config.location(), Some("alias-location"));
    }

    #[rstest::rstest]
    fn typed_config_preserves_source_and_empty_value_fallback() {
        let configured = VertexConfig::new(
            Some(Sourced::new(
                SecretValue::new("inline-json"),
                InputSource::Request,
            )),
            Some("project".into()),
            Some("location".into()),
        );
        assert!(matches!(
            credential_source(&configured, &|_| Some("environment-json".into())),
            CredentialSource::Inline(value) if value.expose() == "inline-json"
        ));
        let empty = VertexConfig::new(
            Some(Sourced::new(SecretValue::new(" "), InputSource::Request)),
            Some(" ".into()),
            Some(" ".into()),
        );
        assert!(matches!(
            credential_source(&empty, &|_| None),
            CredentialSource::Adc
        ));
        assert_eq!(
            get_vertex_ai_project(&empty, &|_| Some("env-project".into())).as_deref(),
            Some("env-project")
        );
        assert_eq!(
            get_vertex_ai_location(&empty, &|_| Some("env-location".into())).as_deref(),
            Some("env-location")
        );
    }

    #[rstest::rstest]
    fn credential_discovery_prefers_input_then_environment_then_adc() {
        let params = json!({"vertex_credentials":"input-json"});
        let sources = BTreeMap::from([("vertex_credentials".to_string(), InputSource::Request)]);
        let configured =
            VertexConfig::from_sourced_optional_params(params.as_object().unwrap(), &sources)
                .unwrap();
        assert!(
            matches!(credential_source(&configured, &|_| Some("environment-value".into())), CredentialSource::Inline(value) if value.expose() == "input-json")
        );
        let empty = VertexConfig::default();
        assert!(
            matches!(credential_source(&empty, &|name| (name == VERTEXAI_CREDENTIALS_ENV).then(|| "environment-json".into())), CredentialSource::Trusted(value) if value.expose() == "environment-json")
        );
        assert!(
            matches!(credential_source(&empty, &|name| (name == GOOGLE_APPLICATION_CREDENTIALS_ENV).then(|| "adc.json".into())), CredentialSource::ApplicationCredentials(path) if path == "adc.json")
        );
        assert!(matches!(
            credential_source(&empty, &|_| None),
            CredentialSource::Adc
        ));
    }

    #[rstest::rstest]
    fn params_become_a_deployment_sourced_config() {
        let params = VertexParams {
            vertex_credentials: Some("deployment-json".into()),
            vertex_project: Some("project-1".into()),
            vertex_ai_location: Some("europe-west4".into()),
            ..VertexParams::default()
        };
        let config = VertexConfig::from_params(&params);
        assert_eq!(config.project_id(), Some("project-1"));
        assert_eq!(config.location(), Some("europe-west4"));
        assert!(
            matches!(credential_source(&config, &|_| Some("environment-json".into())), CredentialSource::Trusted(value) if value.expose() == "deployment-json")
        );
        assert!(matches!(
            credential_source(
                &VertexConfig::from_params(&VertexParams::default()),
                &|_| None
            ),
            CredentialSource::Adc
        ));
    }
}
