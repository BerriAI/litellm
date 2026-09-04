#[cfg(feature = "python")]
use std::path::Path;

#[cfg(feature = "python")]
use litellm_core::router::Deployment;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use thiserror::Error;

pub mod gil;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("read_model_list failed: {0}")]
    PythonLoading(String),
    #[error("serializing model_list failed: {0}")]
    Serialization(String),
    #[error("parsing model_list failed: {0}")]
    ModelListParsing(#[source] serde_json::Error),
}

#[cfg(feature = "python")]
pub fn load_model_list(config_path: &Path) -> Result<Vec<Deployment>, ConfigError> {
    gil::record_acquisition();
    Python::attach(|python| {
        let model_list = python
            .import("litellm.proxy.read_model_list")
            .and_then(|module| module.getattr("read_model_list"))
            .and_then(|reader| reader.call1((config_path.to_string_lossy().as_ref(),)))
            .map_err(|error| ConfigError::PythonLoading(error.to_string()))?;

        let model_list_json = python
            .import("json")
            .and_then(|json| json.getattr("dumps"))
            .and_then(|dumps| dumps.call1((model_list,)))
            .and_then(|encoded| encoded.extract::<String>())
            .map_err(|error| ConfigError::Serialization(error.to_string()))?;

        parse_model_list(&model_list_json)
    })
}

#[cfg(feature = "python")]
fn parse_model_list(model_list_json: &str) -> Result<Vec<Deployment>, ConfigError> {
    serde_json::from_str(model_list_json).map_err(ConfigError::ModelListParsing)
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;

    #[test]
    fn parses_resolved_model_list() {
        let deployments = parse_model_list(
            r#"[
                {
                    "model_name": "realtime",
                    "litellm_params": {
                        "model": "openai/gpt-realtime",
                        "api_key": "resolved-secret",
                        "api_base": "https://api.example.test/v1"
                    }
                },
                {
                    "model_name": "without-optional-values",
                    "litellm_params": {"model": "openai/gpt-4.1"}
                }
            ]"#,
        )
        .expect("resolved model list should parse");

        assert_eq!(deployments.len(), 2);
        assert_eq!(deployments[0].model_name, "realtime");
        assert_eq!(
            deployments[0].litellm_params.api_key.as_deref(),
            Some("resolved-secret")
        );
        assert_eq!(
            deployments[0].litellm_params.api_base.as_deref(),
            Some("https://api.example.test/v1")
        );
        assert_eq!(deployments[1].litellm_params.api_key, None);
        assert_eq!(deployments[1].litellm_params.api_base, None);
    }

    #[test]
    fn malformed_model_list_returns_parsing_error() {
        let error = parse_model_list(r#"[{"model_name":"missing-params"}]"#)
            .expect_err("missing litellm_params should fail");

        assert!(matches!(error, ConfigError::ModelListParsing(_)));
    }
}
