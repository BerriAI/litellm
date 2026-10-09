use crate::execution::{run_async, run_sync};
use litellm_inference_transcription::{
    AudioTranscriptionRoute, Error, types::AudioTranscriptionRequest,
};
use pyo3::prelude::*;
use serde_json::{Map, Value};

use super::NativeCall;

use crate::{
    errors::route_error_to_pyerr,
    marshal::{RouteOptions, optional_object_field, required_field, value_route_options},
};

async fn execute(
    http: Result<litellm_http::Client, litellm_http::Error>,
    secrets: std::sync::Arc<dyn litellm_secrets::source::SecretSource>,
    audio: Value,
    optional_params: Map<String, Value>,
    options: RouteOptions,
) -> Result<Value, Error> {
    let RouteOptions {
        model,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout,
    } = options;
    AudioTranscriptionRoute::new(http?, crate::http::resources().auth.clone(), secrets)
        .execute(AudioTranscriptionRequest {
            model: &model,
            audio,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            optional_params,
            timeout,
        })
        .await
}

#[pyfunction]
pub(crate) fn transcription(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    let arguments = call.resolved()?;
    let audio: Value =
        litellm_host_python::from_py_argument(&required_field(&arguments, "audio")?)?;
    let options = value_route_options(&arguments)?;
    let optional_params = optional_object_field(&arguments, "optional_params")?.unwrap_or_default();
    let http = crate::http::provider_client(py, &call.kwargs, false)?;
    let secrets = crate::secrets::source(py)?;
    run_sync(
        py,
        execute(http, secrets, audio, optional_params, options),
        route_error_to_pyerr,
    )
}

#[pyfunction]
pub(crate) fn atranscription<'py>(
    py: Python<'py>,
    call: NativeCall<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let arguments = call.resolved()?;
    let audio: Value =
        litellm_host_python::from_py_argument(&required_field(&arguments, "audio")?)?;
    let options = value_route_options(&arguments)?;
    let optional_params = optional_object_field(&arguments, "optional_params")?.unwrap_or_default();
    let http = crate::http::provider_client(py, &call.kwargs, true)?;
    let secrets = crate::secrets::source(py)?;
    run_async(
        py,
        execute(http, secrets, audio, optional_params, options),
        route_error_to_pyerr,
    )
}
