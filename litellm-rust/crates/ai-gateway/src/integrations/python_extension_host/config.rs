use litellm_python_extension_protocol::{ExtensionKind, ExtensionSpec};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct PythonExtensionManifest {
    pub revision_id: String,
    pub extensions: Vec<ManifestExtension>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ManifestExtension {
    pub id: String,
    pub kind: ManifestExtensionKind,
    pub entrypoint: String,
    #[serde(default)]
    pub constructor: serde_json::Value,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ManifestExtensionKind {
    Callback,
    Guardrail,
}

impl PythonExtensionManifest {
    pub fn specs(&self) -> Result<Vec<ExtensionSpec>, serde_json::Error> {
        self.extensions
            .iter()
            .map(|extension| {
                Ok(ExtensionSpec {
                    id: extension.id.clone(),
                    kind: match extension.kind {
                        ManifestExtensionKind::Callback => ExtensionKind::Callback.into(),
                        ManifestExtensionKind::Guardrail => ExtensionKind::Guardrail.into(),
                    },
                    entrypoint: extension.entrypoint.clone(),
                    constructor_json: serde_json::to_vec(&extension.constructor)?,
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct PythonExtensionSettings {
    pub endpoint: String,
    pub token: String,
    pub connect_timeout: std::time::Duration,
    pub hook_timeout: std::time::Duration,
    pub callback_queue_size: usize,
    pub callback_batch_size: usize,
}

impl PythonExtensionSettings {
    pub fn from_env() -> Result<Option<Self>, String> {
        let Some(endpoint) = non_empty_env("LITELLM_PYTHON_EXTENSION_HOST_ENDPOINT") else {
            return Ok(None);
        };
        let token = non_empty_env("LITELLM_PYTHON_EXTENSION_HOST_TOKEN").ok_or_else(|| {
            "LITELLM_PYTHON_EXTENSION_HOST_TOKEN is required when the endpoint is configured"
                .to_string()
        })?;
        Ok(Some(Self {
            endpoint,
            token,
            connect_timeout: seconds_env("LITELLM_PYTHON_EXTENSION_CONNECT_TIMEOUT_SECONDS", 5.0)?,
            hook_timeout: seconds_env("LITELLM_PYTHON_EXTENSION_HOOK_TIMEOUT_SECONDS", 30.0)?,
            callback_queue_size: usize_env("LITELLM_PYTHON_EXTENSION_CALLBACK_QUEUE_SIZE", 1_000)?,
            callback_batch_size: usize_env("LITELLM_PYTHON_EXTENSION_CALLBACK_BATCH_SIZE", 50)?,
        }))
    }
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn seconds_env(name: &str, default: f64) -> Result<std::time::Duration, String> {
    let value = match non_empty_env(name) {
        Some(value) => value
            .parse::<f64>()
            .map_err(|error| format!("{name} must be a number: {error}"))?,
        None => default,
    };
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(std::time::Duration::from_secs_f64(value))
}

fn usize_env(name: &str, default: usize) -> Result<usize, String> {
    let value = match non_empty_env(name) {
        Some(value) => value
            .parse::<usize>()
            .map_err(|error| format!("{name} must be an integer: {error}"))?,
        None => default,
    };
    if value == 0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(value)
}
