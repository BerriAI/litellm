//! Native Rust YAML config loader for model_list.
//!
//! Parses the same YAML format as the Python proxy's config, without requiring
//! an embedded Python interpreter. Supports `os.environ/` references for API keys.

use litellm_core::router::{Deployment, LiteLLMParams, Router};
use serde::Deserialize;

/// Top-level config shape.
#[derive(Deserialize)]
struct Config {
    model_list: Vec<ModelListEntry>,
}

#[derive(Deserialize)]
struct ModelListEntry {
    model_name: String,
    litellm_params: LiteLLMParamsConfig,
}

#[derive(Deserialize)]
struct LiteLLMParamsConfig {
    model: String,
    api_key: Option<String>,
    api_base: Option<String>,
}

/// Load a router from a YAML config file.
pub fn load_router_from_yaml(path: &str) -> Result<Router, String> {
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
            }
        })
        .collect();

    eprintln!("  total deployments loaded: {}", deployments.len());
    Ok(Router::new(deployments))
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
