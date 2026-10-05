use std::sync::{Mutex, MutexGuard};

use litellm_host_python::{from_py_argument, to_py};
use litellm_router::{
    Deployment, Error, Router,
    config::{CooldownPolicy, DeploymentConfig, RetryPolicy, RouterConfig, Strategy},
};
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
};
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouterInput {
    model_list: Vec<ModelInput>,
    #[serde(default)]
    strategy: Strategy,
    #[serde(default)]
    retry: RetryPolicy,
    timeout: Option<f64>,
    cooldown: Option<CooldownPolicy>,
}

#[derive(Deserialize)]
struct ModelInput {
    model_name: String,
    litellm_params: ProviderInput,
    #[serde(default)]
    model_info: Map<String, Value>,
}

#[derive(Deserialize)]
struct ProviderInput {
    model: String,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
}

impl From<RouterInput> for RouterConfig {
    fn from(input: RouterInput) -> Self {
        Self {
            deployments: input
                .model_list
                .into_iter()
                .map(|model| DeploymentConfig {
                    deployment_id: model
                        .model_info
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    model_name: model.model_name,
                    deployment: Deployment {
                        model: model.litellm_params.model,
                        api_key: model.litellm_params.api_key,
                        api_base: model.litellm_params.api_base,
                        custom_llm_provider: model.litellm_params.custom_llm_provider,
                        ..Deployment::default()
                    },
                })
                .collect(),
            strategy: input.strategy,
            retry: input.retry,
            timeout: input.timeout,
            cooldown: input.cooldown,
        }
    }
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub struct RouterHandle {
    router: Mutex<Router>,
}

impl RouterHandle {
    fn lock(&self) -> PyResult<MutexGuard<'_, Router>> {
        self.router
            .lock()
            .map_err(|_| PyRuntimeError::new_err("router state is unavailable"))
    }
}

fn public_error(error: Error) -> PyErr {
    match error {
        Error::Closed => PyRuntimeError::new_err(error.to_string()),
        Error::NotImplemented => PyNotImplementedError::new_err(error.to_string()),
        Error::DuplicateDeployment { .. }
        | Error::UnknownModel { .. }
        | Error::InvalidSelection { .. }
        | Error::Unauthorized => PyValueError::new_err(error.to_string()),
    }
}

#[pyfunction]
pub fn routing_create(config: &Bound<'_, PyAny>) -> PyResult<RouterHandle> {
    let input: RouterInput = from_py_argument(config)?;
    Ok(RouterHandle {
        router: Mutex::new(Router::new(input.into()).map_err(public_error)?),
    })
}

#[pyfunction]
pub fn routing_snapshot(py: Python<'_>, router: &RouterHandle) -> PyResult<Py<PyAny>> {
    let snapshot = router.lock()?.snapshot();
    to_py(py, &snapshot)
}

#[pyfunction]
pub fn routing_reconfigure(router: &RouterHandle, config: &Bound<'_, PyAny>) -> PyResult<()> {
    let input: RouterInput = from_py_argument(config)?;
    let config = input.into();
    router.lock()?.reconfigure(config).map_err(public_error)
}

#[pyfunction]
pub fn routing_close(router: &RouterHandle) -> PyResult<()> {
    router.lock()?.close();
    Ok(())
}

#[cfg(test)]
mod tests {
    use litellm_router::call::{Endpoint, RoutingOptions, RoutingRequest};
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::explicit_id(json!({"id": "configured", "unrelated": {"metadata": true}}), "configured")]
    #[case::non_string_id(json!({"id": 42}), "deployment-0")]
    #[case::missing_id(json!({}), "deployment-0")]
    fn input_projection_preserves_provider_parameters_and_policies(
        #[case] model_info: Value,
        #[case] expected_id: &str,
    ) {
        let input = serde_json::from_value::<RouterInput>(json!({
            "model_list": [{
                "model_name": "primary",
                "model_info": model_info,
                "blocked": false,
                "litellm_params": {
                    "model": "provider/model",
                    "api_key": "os.environ/ROUTER_TEST_KEY",
                    "api_base": "https://provider.example/v1",
                    "custom_llm_provider": "provider",
                    "provider_option": {"enabled": true}
                }
            }],
            "strategy": "least_busy",
            "retry": {"retries": 2, "fallbacks": {"primary": ["backup"]}},
            "timeout": 30.0,
            "cooldown": {"allowed_failures": 3, "duration": 5.0}
        }))
        .unwrap();
        let router = Router::new(input.into()).unwrap();
        let deployment = router.get("primary").unwrap();
        let call = router
            .start(RoutingRequest {
                model: "primary".into(),
                endpoint: Endpoint::ChatCompletions,
                options: RoutingOptions::default(),
            })
            .unwrap();

        assert_eq!(router.snapshot().deployments[0].deployment_id, expected_id);
        assert_eq!(deployment.model, "provider/model");
        assert_eq!(
            deployment.api_key.as_deref(),
            Some("os.environ/ROUTER_TEST_KEY")
        );
        assert_eq!(
            deployment.api_base.as_deref(),
            Some("https://provider.example/v1")
        );
        assert_eq!(deployment.custom_llm_provider.as_deref(), Some("provider"));
        assert!(matches!(call.config().strategy, Strategy::LeastBusy));
        assert_eq!(call.config().retry.retries, 2);
        assert_eq!(call.config().retry.fallbacks["primary"], ["backup"]);
        assert_eq!(call.config().timeout, Some(30.0));
        assert_eq!(call.config().cooldown.as_ref().unwrap().allowed_failures, 3);
        assert_eq!(call.config().cooldown.as_ref().unwrap().duration, 5.0);
    }
}
