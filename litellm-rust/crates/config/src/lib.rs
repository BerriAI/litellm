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
    #[serde(default)]
    pub use_redis_transaction_buffer: bool,
    pub proxy_batch_write_at: Option<u64>,
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
    pub input_cost_per_token: Option<f64>,
    pub output_cost_per_token: Option<f64>,
    pub cache_read_input_token_cost: Option<f64>,
    pub cache_creation_input_token_cost: Option<f64>,
}

impl Config {
    pub fn from_yaml(yaml: &str) -> Result<Self, Error> {
        Ok(serde_yaml_ng::from_str(yaml)?)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::from_yaml(&std::fs::read_to_string(path)?)
    }
}
