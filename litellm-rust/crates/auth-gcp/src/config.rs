use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use base64::{
    Engine, alphabet,
    engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig},
};
use litellm_auth_types::{Error, ErrorDetail, InputSource, Sourced};
use litellm_secret_value::SecretValue;

use serde_json::{Map, Value};

use litellm_auth_types::VertexParams;

use crate::constants::{
    CLOUDSDK_ACTIVE_CONFIG_NAME_ENV, CLOUDSDK_CONFIG_ENV, CLOUDSDK_CORE_PROJECT_ENV,
    GCLOUD_PROJECT_ENV, GOOGLE_APPLICATION_CREDENTIALS_ENV, GOOGLE_CLOUD_PROJECT_ENV,
};

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

/// The project a request is billed to when nothing names it, read the way Python's
/// `VertexBase.load_auth` reads it: from explicit credentials (`quota_project_id` for an
/// authorized user, otherwise `project_id`), or as `google.auth.default` does for application
/// default credentials (`GOOGLE_CLOUD_PROJECT`, the key file, then the gcloud configuration
/// next to gcloud's credentials). The metadata server's project needs a network call, so
/// credentials found there resolve to `None`.
pub fn get_vertex_ai_project_from_credentials(
    config: &VertexConfig,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    let explicit = |text: &str| explicit_credentials_project(&serde_json::from_str(text).ok()?);
    match credential_source(config, env_lookup) {
        CredentialSource::Inline(configured) => explicit(configured.expose()),
        CredentialSource::Trusted(configured) => {
            let configured = configured.expose();
            if names_a_credentials_file(configured) {
                explicit(&std::fs::read_to_string(configured).ok()?)
            } else {
                explicit(configured)
            }
        }
        CredentialSource::ApplicationCredentials(path) => {
            default_project(env_lookup, Some(Path::new(&path)))
        }
        CredentialSource::Adc => default_project(
            env_lookup,
            well_known_credentials_file(env_lookup).as_deref(),
        ),
    }
}

pub(crate) fn explicit_credentials_project(credentials: &Value) -> Option<String> {
    let field = match credentials.get("type").and_then(Value::as_str) {
        Some("authorized_user") => "quota_project_id",
        _ => "project_id",
    };
    credentials
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn default_project(
    env_lookup: &dyn Fn(&str) -> Option<String>,
    path: Option<&Path>,
) -> Option<String> {
    non_empty_env(env_lookup, GOOGLE_CLOUD_PROJECT_ENV)
        .or_else(|| non_empty_env(env_lookup, GCLOUD_PROJECT_ENV))
        .or_else(|| {
            serde_json::from_str::<Value>(&std::fs::read_to_string(path?).ok()?)
                .ok()?
                .get("project_id")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .or_else(|| {
            (path.is_some() && well_known_credentials_file(env_lookup).as_deref() == path)
                .then(|| gcloud_project(env_lookup))
                .flatten()
        })
}

fn gcloud_config_dir(env_lookup: &dyn Fn(&str) -> Option<String>) -> Option<PathBuf> {
    non_empty_env(env_lookup, CLOUDSDK_CONFIG_ENV)
        .map(PathBuf::from)
        .or_else(default_gcloud_config_dir)
}

#[cfg(windows)]
fn default_gcloud_config_dir() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(|app_data| PathBuf::from(app_data).join("gcloud"))
}

#[cfg(not(windows))]
fn default_gcloud_config_dir() -> Option<PathBuf> {
    std::env::home_dir().map(|home| home.join(".config").join("gcloud"))
}

/// gcloud's `application_default_credentials.json`, which `google.auth.default` reads after
/// `GOOGLE_APPLICATION_CREDENTIALS` and before the metadata server.
pub(crate) fn well_known_credentials_file(
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<PathBuf> {
    gcloud_config_dir(env_lookup)
        .map(|dir| dir.join("application_default_credentials.json"))
        .filter(|path| path.is_file())
}

/// The `core/project` property of gcloud's active configuration, which
/// `google.auth._cloud_sdk.get_project_id` asks gcloud for.
fn gcloud_project(env_lookup: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    if let Some(project) = non_empty_env(env_lookup, CLOUDSDK_CORE_PROJECT_ENV) {
        return Some(project);
    }
    let dir = gcloud_config_dir(env_lookup)?;
    let active = non_empty_env(env_lookup, CLOUDSDK_ACTIVE_CONFIG_NAME_ENV)
        .or_else(|| {
            std::fs::read_to_string(dir.join("active_config"))
                .ok()
                .map(|name| name.trim().to_string())
                .filter(|name| !name.is_empty())
        })
        .unwrap_or_else(|| "default".to_string());
    let properties =
        std::fs::read_to_string(dir.join("configurations").join(format!("config_{active}")))
            .ok()?;
    let mut in_core = false;
    properties.lines().map(str::trim).find_map(|line| {
        if let Some(section) = line
            .strip_prefix('[')
            .and_then(|line| line.strip_suffix(']'))
        {
            in_core = section.trim() == "core";
            return None;
        }
        let (key, value) = line.split_once('=')?;
        (in_core && key.trim() == "project")
            .then(|| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
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

const INLINE_JSON_LEADING_NOISE: [char; 7] = [' ', '\t', '\r', '\n', '\u{feff}', '"', '\''];

fn is_inline_json(configured: &str) -> bool {
    configured
        .trim_start_matches(INLINE_JSON_LEADING_NOISE)
        .starts_with('{')
}

pub(crate) fn names_a_credentials_file(configured: &str) -> bool {
    !is_inline_json(configured) || Path::new(configured).is_file()
}

const BASE64_ANY_PADDING: GeneralPurpose = GeneralPurpose::new(
    &alphabet::STANDARD,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

pub(crate) fn is_base64_encoded_json(configured: &str) -> bool {
    let compact: String = configured.split_whitespace().collect();
    let quoted = compact.trim_matches(['"', '\'']);
    let encoded = quoted.strip_prefix("base64:").unwrap_or(quoted);
    BASE64_ANY_PADDING
        .decode(encoded.replace('-', "+").replace('_', "/"))
        .is_ok_and(|decoded| serde_json::from_slice::<Map<String, Value>>(&decoded).is_ok())
}

#[derive(Clone, Debug)]
pub enum CredentialSource {
    Inline(SecretValue),
    Trusted(SecretValue),
    ApplicationCredentials(String),
    Adc,
}

impl CredentialSource {
    pub fn from_vertex_environment(env_lookup: &dyn Fn(&str) -> Option<String>) -> Self {
        VertexParams::CREDENTIALS
            .resolve(&|_| None, env_lookup)
            .map(|configured| Self::Trusted(SecretValue::new(configured)))
            .unwrap_or_else(|| Self::application_default(env_lookup))
    }

    pub fn application_default(env_lookup: &dyn Fn(&str) -> Option<String>) -> Self {
        non_empty_env(env_lookup, GOOGLE_APPLICATION_CREDENTIALS_ENV)
            .map_or(Self::Adc, Self::ApplicationCredentials)
    }
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
    CredentialSource::from_vertex_environment(env_lookup)
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
    fn application_default_ignores_vertex_credentials() {
        let env = |name: &str| match name {
            VERTEXAI_CREDENTIALS_ENV => Some("environment-json".to_string()),
            GOOGLE_APPLICATION_CREDENTIALS_ENV => Some("adc.json".to_string()),
            _ => None,
        };
        assert!(
            matches!(CredentialSource::application_default(&env), CredentialSource::ApplicationCredentials(path) if path == "adc.json")
        );
        assert!(matches!(
            CredentialSource::application_default(
                &|name| (name == VERTEXAI_CREDENTIALS_ENV).then(|| "environment-json".into())
            ),
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
