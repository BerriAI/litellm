use std::collections::BTreeMap;

use litellm_config::Model;
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    #[default]
    Weighted,
    LeastBusy,
    Latency,
    Usage,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetryPolicy {
    pub retries: u32,
    #[serde(default)]
    pub fallbacks: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CooldownPolicy {
    pub allowed_failures: u32,
    pub duration: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterConfig {
    pub model_list: Vec<Model>,
    #[serde(default)]
    pub strategy: Strategy,
    #[serde(default)]
    pub retry: RetryPolicy,
    pub timeout: Option<f64>,
    pub cooldown: Option<CooldownPolicy>,
}
