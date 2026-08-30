//! Native Rust YAML config loader for model_list.
//!
//! Parses the same YAML format as the Python proxy's config, without requiring
//! an embedded Python interpreter. Supports `os.environ/` references for API keys.

use litellm_core::router::{Deployment, LiteLLMParams, Router};
use serde::Deserialize;

/// Top-level config shape.
#[derive(Deserialize)]
struct Config {
    #[serde(default)]
    model_list: Vec<ModelListEntry>,
    #[serde(default)]
    general_settings: Option<GeneralSettings>,
    #[serde(default)]
    litellm_settings: Option<LiteLLMSettings>,
    #[serde(default)]
    router_settings: Option<RouterSettings>,
}

/// General settings for the gateway.
#[derive(Deserialize, Default)]
pub struct GeneralSettings {
    pub master_key: Option<String>,
    pub database_url: Option<String>,
    pub coordination_redis: Option<String>,
    pub max_parallel_requests: Option<u32>,
    pub global_max_parallel_requests: Option<u32>,
    pub max_request_size_mb: Option<u32>,
    pub max_response_size_mb: Option<u32>,
    pub alerting: Option<Vec<String>>,
    pub alert_webhook_url: Option<String>,
    pub allowed_routes: Option<Vec<String>>,
    pub pass_through_endpoints: Option<Vec<PassThroughEndpoint>>,
}

/// Pass-through endpoint configuration.
#[derive(Deserialize)]
pub struct PassThroughEndpoint {
    pub path: String,
    pub target: String,
    #[serde(default)]
    pub headers: Option<std::collections::HashMap<String, String>>,
}

/// LiteLLM-specific settings.
#[derive(Deserialize, Default)]
pub struct LiteLLMSettings {
    pub callbacks: Option<Vec<CallbackConfig>>,
    pub guardrails: Option<Vec<GuardrailConfig>>,
    pub cache: Option<bool>,
    pub cache_params: Option<CacheParams>,
    pub drop_params: Option<bool>,
    pub num_retries: Option<u32>,
    pub timeout: Option<u32>,
}

/// Callback configuration.
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum CallbackConfig {
    #[serde(rename = "langfuse")]
    Langfuse {
        public_key: String,
        secret_key: String,
        host: String,
    },
    #[serde(rename = "datadog")]
    Datadog {
        api_key: String,
        #[serde(default)]
        app_key: Option<String>,
        host: String,
    },
    #[serde(rename = "webhooks")]
    Webhooks {
        url: String,
        #[serde(default)]
        headers: Option<std::collections::HashMap<String, String>>,
        #[serde(default)]
        auth_token: Option<String>,
    },
    #[serde(rename = "slack")]
    Slack {
        webhook_url: String,
        #[serde(default)]
        channel: Option<String>,
        #[serde(default)]
        username: Option<String>,
        #[serde(default)]
        icon_emoji: Option<String>,
    },
}

/// Guardrail configuration.
#[derive(Deserialize)]
pub struct GuardrailConfig {
    pub guardrail_name: String,
    pub guardrail_type: String,
    #[serde(default)]
    pub enabled: Option<bool>,
}

/// Cache parameters.
#[derive(Deserialize)]
pub struct CacheParams {
    #[serde(rename = "type")]
    pub cache_type: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub password: Option<String>,
    pub ttl: Option<u32>,
}

/// Router settings.
#[derive(Deserialize, Default)]
pub struct RouterSettings {
    pub routing_strategy: Option<String>,
    pub num_retries: Option<u32>,
    pub timeout: Option<u32>,
    pub cooldown_seconds: Option<u32>,
    pub allowed_fails: Option<u32>,
}

impl RouterSettings {
    /// Convert the routing_strategy string to a RoutingStrategy enum.
    pub fn to_routing_strategy(&self) -> litellm_core::router::RoutingStrategy {
        match self.routing_strategy.as_deref() {
            Some("latency-based") | Some("latency_based") => {
                litellm_core::router::RoutingStrategy::LatencyBased
            }
            Some("load-based") | Some("load_based") => {
                litellm_core::router::RoutingStrategy::LoadBased
            }
            Some("cost-based") | Some("cost_based") => {
                litellm_core::router::RoutingStrategy::CostBased
            }
            Some("weighted") => litellm_core::router::RoutingStrategy::Weighted,
            _ => litellm_core::router::RoutingStrategy::SimpleShuffle,
        }
    }
}

#[derive(Deserialize)]
struct ModelListEntry {
    model_name: String,
    litellm_params: LiteLLMParamsConfig,
    #[serde(default)]
    model_info: Option<ModelInfo>,
    #[serde(default)]
    rpm: Option<u32>,
    #[serde(default)]
    tpm: Option<u32>,
    #[serde(default)]
    max_parallel_requests: Option<u32>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    healthy: Option<bool>,
    #[serde(default)]
    cooldown: Option<u32>,
    #[serde(default)]
    weight: Option<u32>,
}

/// Model info configuration.
#[derive(Deserialize)]
pub struct ModelInfo {
    pub input_cost_per_token: Option<f64>,
    pub output_cost_per_token: Option<f64>,
    pub mode: Option<String>,
}

#[derive(Deserialize)]
struct LiteLLMParamsConfig {
    model: String,
    api_key: Option<String>,
    api_base: Option<String>,
}

/// Loaded configuration with all settings.
pub struct LoadedConfig {
    pub router: Router,
    pub general_settings: GeneralSettings,
    pub litellm_settings: LiteLLMSettings,
    pub router_settings: RouterSettings,
}

/// Load a router and all settings from a YAML config file.
pub fn load_config_from_yaml(path: &str) -> Result<LoadedConfig, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read config file {path}: {e}"))?;
    let config: Config =
        serde_yaml::from_str(&content).map_err(|e| format!("failed to parse YAML: {e}"))?;

    let deployments: Vec<Deployment> = config
        .model_list
        .into_iter()
        .map(|entry| {
            let api_key = entry.litellm_params.api_key.map(resolve_env_ref);
            eprintln!(
                "  loaded deployment: model_name={}, model={}",
                entry.model_name, entry.litellm_params.model
            );
            Deployment {
                model_name: entry.model_name,
                litellm_params: LiteLLMParams {
                    model: entry.litellm_params.model,
                    api_key,
                    api_base: entry.litellm_params.api_base,
                },
                healthy: entry.healthy,
                weight: entry.weight,
                input_cost_per_token: entry
                    .model_info
                    .as_ref()
                    .and_then(|info| info.input_cost_per_token),
                output_cost_per_token: entry
                    .model_info
                    .as_ref()
                    .and_then(|info| info.output_cost_per_token),
            }
        })
        .collect();

    eprintln!("  total deployments loaded: {}", deployments.len());

    let general_settings = config.general_settings.unwrap_or_default();
    let litellm_settings = config.litellm_settings.unwrap_or_default();
    let router_settings = config.router_settings.unwrap_or_default();

    // Create router with the configured routing strategy
    let routing_strategy = router_settings.to_routing_strategy();
    let router = Router::with_strategy(deployments, routing_strategy);

    eprintln!("  routing strategy: {:?}", routing_strategy);

    Ok(LoadedConfig {
        router,
        general_settings,
        litellm_settings,
        router_settings,
    })
}

/// Load a router from a YAML config file (backward compatible).
pub fn load_router_from_yaml(path: &str) -> Result<Router, String> {
    load_config_from_yaml(path).map(|config| config.router)
}

/// Resolve `os.environ/VAR_NAME` references to actual env var values.
fn resolve_env_ref(value: String) -> String {
    if let Some(var_name) = value.strip_prefix("os.environ/") {
        std::env::var(var_name).unwrap_or_else(|_| {
            eprintln!("warning: env var {var_name} not set, API key will be empty");
            String::new()
        })
    } else {
        value
    }
}
