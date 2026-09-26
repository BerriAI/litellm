mod error;

use std::path::Path;

use litellm_auth_types::SecretValue;
use serde::Deserialize;

pub use error::Error;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub model_list: Box<[Model]>,
    #[serde(default)]
    pub general_settings: GeneralSettings,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralSettings {
    pub master_key: Option<SecretValue>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub model_name: String,
    pub litellm_params: LiteLlmParams,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiteLlmParams {
    pub model: String,
    pub api_key: Option<SecretValue>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
}

impl Config {
    pub fn from_yaml(yaml: &str) -> Result<Self, Error> {
        Ok(serde_yaml_ng::from_str(yaml)?)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::from_yaml(&std::fs::read_to_string(path)?)
    }
}
