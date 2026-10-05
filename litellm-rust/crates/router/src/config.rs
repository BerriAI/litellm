use std::collections::BTreeMap;

use serde::Deserialize;

use crate::Deployment;

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

#[derive(Clone, Debug, Default)]
pub struct DeploymentConfig {
    pub deployment_id: Option<String>,
    pub model_name: String,
    pub deployment: Deployment,
}

#[derive(Clone, Debug, Default)]
pub struct RouterConfig {
    pub deployments: Vec<DeploymentConfig>,
    pub strategy: Strategy,
    pub retry: RetryPolicy,
    pub timeout: Option<f64>,
    pub cooldown: Option<CooldownPolicy>,
}
