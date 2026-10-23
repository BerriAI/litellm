use litellm_auth::SecretValue;
use litellm_host_python::from_py;
use litellm_inference_tokenize::TokenizeRequest;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use crate::{
    marshal::{optional_timeout, python_timeout_seconds},
    routes::RequestView,
};

pub(super) const MODULE: &str = "litellm.rust_bridge.host.routes.tokenize";

pub(super) fn project_request(
    py: Python<'_>,
    view: &mut RequestView,
    arguments: &Bound<'_, PyDict>,
) -> PyResult<TokenizeRequest> {
    view.resolve(py, arguments)?;
    let argument = |name: &str| view.argument(py, arguments, name);
    let string = |name: &str| -> PyResult<Option<String>> {
        argument(name)?.map(|value| value.extract()).transpose()
    };
    let required = |name: &str| -> PyResult<String> {
        string(name)?.ok_or_else(|| PyValueError::new_err(format!("{name} is required")))
    };
    let timeout = argument("timeout")?
        .or(argument("request_timeout")?)
        .map(|value| python_timeout_seconds(py, value.unbind()))
        .transpose()?
        .flatten();
    Ok(TokenizeRequest {
        model: required("model")?,
        prompt: required("prompt")?,
        custom_llm_provider: string("custom_llm_provider")?,
        api_key: string("api_key")?.map(SecretValue::new),
        api_base: string("api_base")?.or(string("base_url")?),
        extra_headers: argument("extra_headers")?
            .map(|value| from_py(&value))
            .transpose()?
            .unwrap_or_default(),
        optional_params: view.body_params(py, arguments, MODULE)?,
        deployment_params: argument("_tokenize_deployment_params")?
            .map(|value| from_py(&value))
            .transpose()?
            .unwrap_or_default(),
        timeout: optional_timeout(timeout),
    })
}
