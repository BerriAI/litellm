use litellm_core::Error;
use std::future::Future;

use litellm_core::audio_transcription::{
    AudioInput, AudioTranscriptionRequest, DefaultAudioServices, audio_transcription_with_services,
};
use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::core_error_to_pyerr;
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty};

struct AudioTranscriptionInputs {
    model: String,
    audio: AudioInput,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Value>,
    optional_params: Option<Value>,
    timeout_seconds: Option<f64>,
}

fn prepare_transcription(
    inputs: AudioTranscriptionInputs,
) -> PyResult<impl Future<Output = Result<Value, Error>> + Send + 'static> {
    let audio = inputs.audio;
    let options = RouteOptions::from_python(RouteOptionsInputs {
        model: inputs.model,
        api_key: inputs.api_key,
        api_base: inputs.api_base,
        custom_llm_provider: inputs.custom_llm_provider,
        extra_headers: inputs.extra_headers,
        timeout_seconds: inputs.timeout_seconds,
    })?;
    let optional_params = object_or_empty("optional_params", inputs.optional_params)?;

    Ok(async move {
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = options;
        let services = DefaultAudioServices::with_authorization(
            Vec::new(),
            Vec::new(),
            crate::runtime::authorization_services().clone(),
        );
        audio_transcription_with_services(
            &services,
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
    })
}

#[pyfunction]
#[pyo3(signature = (model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn transcription(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] audio: AudioInput,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let future = prepare_transcription(AudioTranscriptionInputs {
        model,
        audio,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params,
        timeout_seconds,
    })?;
    litellm_python_interop::run_sync(py, future, core_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn atranscription(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] audio: AudioInput,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'_, PyAny>> {
    let future = prepare_transcription(AudioTranscriptionInputs {
        model,
        audio,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params,
        timeout_seconds,
    })?;
    litellm_python_interop::run_async(py, future, core_error_to_pyerr)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    super::definition::add_function(module, wrap_pyfunction!(transcription, module)?)?;
    super::definition::add_function(module, wrap_pyfunction!(atranscription, module)?)
}

#[cfg(feature = "trace-parity")]
mod trace {
    use litellm_core::audio_transcription::AudioInput;
    use pyo3::prelude::*;
    use serde_json::Value;

    use super::{AudioTranscriptionInputs, core_error_to_pyerr, prepare_transcription};

    #[pyfunction]
    #[pyo3(signature = (model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
    #[allow(clippy::too_many_arguments)]
    fn transcription(
        py: Python<'_>,
        model: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)] audio: AudioInput,
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let future = prepare_transcription(AudioTranscriptionInputs {
            model,
            audio,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            optional_params,
            timeout_seconds,
        })?;
        litellm_python_interop::run_sync(
            py,
            crate::trace_parity::capture(future),
            core_error_to_pyerr,
        )
    }

    #[pyfunction]
    #[pyo3(signature = (model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
    #[allow(clippy::too_many_arguments)]
    fn atranscription(
        py: Python<'_>,
        model: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)] audio: AudioInput,
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Bound<'_, PyAny>> {
        let future = prepare_transcription(AudioTranscriptionInputs {
            model,
            audio,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            optional_params,
            timeout_seconds,
        })?;
        litellm_python_interop::run_async(
            py,
            crate::trace_parity::capture(future),
            core_error_to_pyerr,
        )
    }

    pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        super::super::definition::add_function(module, wrap_pyfunction!(transcription, module)?)?;
        super::super::definition::add_function(module, wrap_pyfunction!(atranscription, module)?)
    }
}

#[cfg(feature = "trace-parity")]
pub(super) fn register_trace(module: &Bound<'_, PyModule>) -> PyResult<()> {
    trace::register(module)
}
