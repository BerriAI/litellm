use crate::logger::{run_async, run_sync};
use litellm_core::audio_transcription::{
    Error, audio_transcription as run_audio_transcription, types::AudioTranscriptionRequest,
};
use litellm_host_python::from_py_argument;
use litellm_http::HttpClientConfig;
use pyo3::{prelude::*, types::PyDict};
use serde_json::{Map, Value};

use crate::{
    errors::route_error_to_pyerr,
    marshal::{RouteOptions, extra_headers_argument, optional_params_argument, optional_timeout},
};

async fn execute(
    config: HttpClientConfig,
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
    run_audio_transcription(
        crate::http::resources(),
        &config,
        AudioTranscriptionRequest {
            model: &model,
            audio,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            optional_params,
            timeout,
        },
    )
    .await
}

#[pyfunction]
#[pyo3(signature = (model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per Python keyword"
)]
pub(crate) fn transcription(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = from_py_argument)] audio: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = extra_headers_argument)] extra_headers: Option<Map<String, Value>>,
    #[pyo3(from_py_with = optional_params_argument)] optional_params: Option<Map<String, Value>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let options = RouteOptions {
        model,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout: optional_timeout(timeout_seconds),
    };
    let config = crate::http::call_config(py, &PyDict::new(py), false)?;
    run_sync(
        py,
        execute(config, audio, optional_params.unwrap_or_default(), options),
        route_error_to_pyerr,
    )
}

#[pyfunction]
#[pyo3(signature = (model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per Python keyword"
)]
pub(crate) fn atranscription<'py>(
    py: Python<'py>,
    model: String,
    #[pyo3(from_py_with = from_py_argument)] audio: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = extra_headers_argument)] extra_headers: Option<Map<String, Value>>,
    #[pyo3(from_py_with = optional_params_argument)] optional_params: Option<Map<String, Value>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let options = RouteOptions {
        model,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout: optional_timeout(timeout_seconds),
    };
    let config = crate::http::call_config(py, &PyDict::new(py), true)?;
    run_async(
        py,
        execute(config, audio, optional_params.unwrap_or_default(), options),
        route_error_to_pyerr,
    )
}
