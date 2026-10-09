mod document;
mod errors;
mod host;
mod project;

use host::OcrPythonHost;
use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_core_utils::settings::ProcessEnvironment;
use litellm_host_python::to_py;
use litellm_inference_ocr::provider_config;
use litellm_llms::base_llm::ocr::settings::OcrSettings;
use pyo3::prelude::*;

use super::NativeCall;

use crate::{
    coercion::FieldSpec,
    http,
    python_settings::{PythonSettings, Snapshot},
};

const ENABLE_AZURE_AD_TOKEN_REFRESH: FieldSpec<bool> =
    FieldSpec::new("enable_azure_ad_token_refresh", |field| {
        Ok(field.exact_true())
    });

fn run_ocr(py: Python<'_>, call: NativeCall<'_>, asynchronous: bool) -> PyResult<Py<PyAny>> {
    let (arguments, hooks) =
        crate::routes::call_hooks(py, LoggingOperation::Ocr, &call, asynchronous)?;
    crate::routes::run_public_call(
        py,
        arguments,
        move |py, arguments, request| {
            let config = http::call_config(py, arguments, asynchronous)?;
            let client = litellm_llms::base_llm::ocr::handler::OcrClient::new(
                &http::resources().pool,
                &config,
                http::url_policy(py)?,
                http::resources().auth.clone(),
                ocr_settings(py)?,
                crate::secrets::source(py)?,
            )
            .map_err(http::client_error)?;
            let route = litellm_inference_ocr::OcrRoute::new(client);
            Ok(route.machine(request, None))
        },
        OcrPythonHost::new(call.resolved()?.unbind()),
        hooks,
        asynchronous,
    )
}

fn ocr_settings(py: Python<'_>) -> PyResult<OcrSettings> {
    project_provider_defaults(&PythonSettings::ProviderDefaults.read(py)?)
}

fn project_provider_defaults(snapshot: &Snapshot<'_>) -> PyResult<OcrSettings> {
    Ok(OcrSettings {
        enable_azure_ad_token_refresh: snapshot.read(&ENABLE_AZURE_AD_TOKEN_REFRESH)?,
        ..OcrSettings::from_environment(&ProcessEnvironment)
    })
}

#[pyfunction]
pub(crate) fn ocr(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_ocr(py, call, false)
}

#[pyfunction]
pub(crate) fn aocr(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_ocr(py, call, true)
}

#[pyfunction]
pub(crate) fn ocr_health_check_document(
    py: Python<'_>,
    model: &str,
    custom_llm_provider: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let document = provider_config::get_health_check_document(model, custom_llm_provider)
        .map_err(errors::to_pyerr)?;
    to_py(py, &document)
}

#[pyfunction]
pub(crate) fn ocr_passthrough_response(
    py: Python<'_>,
    model: &str,
    endpoint: &str,
    body: &[u8],
) -> PyResult<Option<Py<PyAny>>> {
    provider_config::passthrough_response(model, endpoint, body)
        .map_err(errors::to_pyerr)?
        .map(|response| to_py(py, &response.into_json()))
        .transpose()
}

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;

    use crate::python_settings::PythonSettings;

    #[rstest::rstest]
    fn provider_defaults_read_exact_true_only() {
        Python::initialize();
        Python::attach(|py| {
            let value = py
                .eval(
                    c"__import__('types').SimpleNamespace(enable_azure_ad_token_refresh=1)",
                    None,
                    None,
                )
                .unwrap();
            let snapshot = PythonSettings::ProviderDefaults.snapshot(value.clone());
            assert!(
                !super::project_provider_defaults(&snapshot)
                    .unwrap()
                    .enable_azure_ad_token_refresh
            );
            value
                .setattr("enable_azure_ad_token_refresh", true)
                .unwrap();
            assert!(
                super::project_provider_defaults(&snapshot)
                    .unwrap()
                    .enable_azure_ad_token_refresh
            );
        });
    }
}
