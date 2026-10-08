use crate::execution::{run_async, run_sync};
use litellm_host_python::from_py_argument;
use litellm_inference::Connection;
use litellm_inference_transcription::{
    AudioTranscriptionRoute, Error, types::AudioTranscriptionRequest,
};
use pyo3::{prelude::*, types::PyDict};
use serde_json::Value;

use super::NativeCall;

use crate::{
    errors::route_error_to_pyerr,
    marshal::{optional_field, optional_object_field, optional_timeout, required_field},
};

fn request(bound: &Bound<'_, PyDict>) -> PyResult<AudioTranscriptionRequest> {
    Ok(AudioTranscriptionRequest {
        audio: from_py_argument(&required_field(bound, "audio")?)?,
        model: from_py_argument(&required_field(bound, "model")?)?,
        connection: Connection {
            api_key: optional_field(bound, "api_key")?,
            api_base: optional_field(bound, "api_base")?,
            extra_headers: optional_object_field(bound, "extra_headers")?,
            timeout: optional_timeout(optional_field(bound, "timeout_seconds")?),
        },
        custom_llm_provider: optional_field(bound, "custom_llm_provider")?,
        optional_params: optional_object_field(bound, "optional_params")?.unwrap_or_default(),
    })
}

async fn execute(
    http: Result<litellm_http::Client, litellm_http::Error>,
    secrets: std::sync::Arc<dyn litellm_secrets::source::SecretSource>,
    request: AudioTranscriptionRequest,
) -> Result<Value, Error> {
    AudioTranscriptionRoute::new(http?, crate::http::resources().auth.clone(), secrets)
        .execute(request)
        .await
}

#[pyfunction]
pub(crate) fn transcription(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    let request = request(&call.bound)?;
    let http = crate::http::provider_client(py, &call.kwargs, false)?;
    let secrets = crate::secrets::source(py)?;
    run_sync(py, execute(http, secrets, request), route_error_to_pyerr)
}

#[pyfunction]
pub(crate) fn atranscription<'py>(
    py: Python<'py>,
    call: NativeCall<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let request = request(&call.bound)?;
    let http = crate::http::provider_client(py, &call.kwargs, true)?;
    let secrets = crate::secrets::source(py)?;
    run_async(py, execute(http, secrets, request), route_error_to_pyerr)
}
