mod document;
mod errors;
mod host;
mod project;

use host::OcrPythonHost;
use litellm_callbacks_legacy_python::{LegacySurface, PublicCall, run_legacy_call};
use litellm_core::ocr::{provider_config, route::ocr_machine};
use litellm_core_utils::settings::ProcessEnvironment;
use litellm_host_python::to_py;
use litellm_llms::base_llm::ocr::settings::OcrSettings;
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

use crate::{
    coercion::FieldSpec,
    http,
    python_settings::{PythonSettings, Snapshot},
    secrets,
};

const VERTEX_PROJECT: FieldSpec<Option<String>> =
    FieldSpec::new("vertex_project", |field| field.falsy_optional_string());
const VERTEX_LOCATION: FieldSpec<Option<String>> =
    FieldSpec::new("vertex_location", |field| field.falsy_optional_string());
const ENABLE_AZURE_AD_TOKEN_REFRESH: FieldSpec<bool> =
    FieldSpec::new("enable_azure_ad_token_refresh", |field| {
        Ok(field.exact_true())
    });

const SURFACE: LegacySurface = LegacySurface {
    call_type: "ocr",
    input_description: "OCR document processing",
    stream: None,
};

const ASYNC_SURFACE: LegacySurface = LegacySurface {
    call_type: "aocr",
    ..SURFACE
};

fn run_ocr(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let secrets = secrets::source(py)?;
    let config = http::call_config(py, &kwargs, asynchronous)?;
    let client = http::resources()
        .ocr_client(&config, http::url_policy(py)?, ocr_settings(py)?, secrets)
        .map_err(http::client_error)?;
    run_legacy_call(
        py,
        if asynchronous { ASYNC_SURFACE } else { SURFACE },
        PublicCall::capture(&request, &args, &kwargs)?,
        crate::logger::LoggedMachine::new(ocr_machine(client)),
        OcrPythonHost::new(request.unbind()),
        crate::preflight::sdk_preflight,
        asynchronous,
    )
}

fn ocr_settings(py: Python<'_>) -> PyResult<OcrSettings> {
    project_provider_defaults(&PythonSettings::ProviderDefaults.read(py)?)
}

fn project_provider_defaults(snapshot: &Snapshot<'_>) -> PyResult<OcrSettings> {
    Ok(OcrSettings {
        vertex_project: snapshot.read(&VERTEX_PROJECT)?,
        vertex_location: snapshot.read(&VERTEX_LOCATION)?,
        enable_azure_ad_token_refresh: snapshot.read(&ENABLE_AZURE_AD_TOKEN_REFRESH)?,
        ..OcrSettings::from_environment(&ProcessEnvironment)
    })
}

#[pyfunction]
pub(crate) fn ocr(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_ocr(py, request, args, kwargs, false)
}

#[pyfunction]
pub(crate) fn aocr(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_ocr(py, request, args, kwargs, true)
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

    #[test]
    fn provider_defaults_distinguish_falsey_values_and_exact_true() {
        Python::initialize();
        Python::attach(|py| {
            let value = py.eval(c"__import__('types').SimpleNamespace(vertex_project=[], vertex_location=0, enable_azure_ad_token_refresh=1)", None, None).unwrap();
            let snapshot = PythonSettings::ProviderDefaults.snapshot(value.clone());
            let projected = super::project_provider_defaults(&snapshot).unwrap();
            assert_eq!(projected.vertex_project, None);
            assert_eq!(projected.vertex_location, None);
            assert!(!projected.enable_azure_ad_token_refresh);
            value.setattr("vertex_project", "project").unwrap();
            value.setattr("vertex_location", "region").unwrap();
            value
                .setattr("enable_azure_ad_token_refresh", true)
                .unwrap();
            let next = super::project_provider_defaults(&snapshot).unwrap();
            assert_eq!(next.vertex_project.as_deref(), Some("project"));
            assert_eq!(next.vertex_location.as_deref(), Some("region"));
            assert!(next.enable_azure_ad_token_refresh);
            value.setattr("vertex_project", 1).unwrap();
            let error = super::project_provider_defaults(&snapshot).err().unwrap();
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                error
                    .to_string()
                    .contains("provider_defaults.vertex_project")
            );
        });
    }
}
