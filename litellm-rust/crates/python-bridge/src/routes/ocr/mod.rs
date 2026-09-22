mod document;
mod errors;
mod host;
mod project;

use std::sync::{Arc, LazyLock};

use host::OcrRouteHost;
use litellm_auth_gcp::VertexAuth;
use litellm_callbacks_legacy_python::{LegacySurface, PublicCall, run_legacy_call};
use litellm_core::ocr::route::ocr_machine;
use litellm_core_utils::settings::ProcessEnvironment;
use litellm_llms::base_llm::ocr::{
    handler::OcrClient,
    settings::{OcrSettings, Secrets},
};
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

use crate::{coercion::Field, errors::RustBridgeDeclined, http, python_settings::PythonSettings};

const SURFACE: LegacySurface = LegacySurface {
    call_type: "ocr",
    input_description: "OCR document processing",
    stream: None,
};

const ASYNC_SURFACE: LegacySurface = LegacySurface {
    call_type: "aocr",
    ..SURFACE
};

static VERTEX_AUTH: LazyLock<VertexAuth> = LazyLock::new(VertexAuth::default);

fn run_ocr(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let secrets = process_environment_secrets(&PythonSettings::SecretManager.read(py)?)?;
    let config = http::call_config(py, &kwargs, asynchronous)?;
    let client = OcrClient::new(
        http::pool(),
        &config,
        http::url_policy(py)?,
        VERTEX_AUTH.clone(),
        ocr_settings(py)?,
        secrets,
    )
    .map_err(http::client_error)?;
    run_legacy_call(
        py,
        if asynchronous { ASYNC_SURFACE } else { SURFACE },
        PublicCall::capture(&request, &args, &kwargs)?,
        ocr_machine(client),
        OcrRouteHost::new(request.unbind()),
        asynchronous,
    )
}

fn process_environment_secrets(secret_manager: &Bound<'_, PyAny>) -> PyResult<Secrets> {
    if Field::read(secret_manager, "secret_manager.readable")?.schema_bool()? {
        return Err(RustBridgeDeclined::new_err(
            "a readable secret manager is configured and the Rust route only reads the process environment",
        ));
    }
    Ok(Arc::new(ProcessEnvironment))
}

fn ocr_settings(py: Python<'_>) -> PyResult<OcrSettings> {
    project_provider_defaults(&PythonSettings::ProviderDefaults.read(py)?)
}

fn project_provider_defaults(value: &Bound<'_, PyAny>) -> PyResult<OcrSettings> {
    Ok(OcrSettings {
        vertex_project: Field::read(value, "provider_defaults.vertex_project")?
            .falsy_optional_string()?
            .0,
        vertex_location: Field::read(value, "provider_defaults.vertex_location")?
            .falsy_optional_string()?
            .0,
        enable_azure_ad_token_refresh: Field::read(
            value,
            "provider_defaults.enable_azure_ad_token_refresh",
        )?
        .exact_true()
        .0,
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

#[cfg(test)]
mod tests {
    use pyo3::{prelude::*, types::PyDict};

    use super::process_environment_secrets;
    use crate::errors::RustBridgeDeclined;

    fn secret_manager<'py>(py: Python<'py>, readable: bool) -> Bound<'py, PyAny> {
        let locals = PyDict::new(py);
        locals.set_item("readable", readable).unwrap();
        py.run(
            c"import types\nmanager = types.SimpleNamespace(readable=readable)",
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        locals.get_item("manager").unwrap().unwrap()
    }

    #[test]
    fn provider_defaults_distinguish_falsey_values_and_exact_true() {
        Python::initialize();
        Python::attach(|py| {
            let value = py.eval(c"__import__('types').SimpleNamespace(vertex_project=[], vertex_location=0, enable_azure_ad_token_refresh=1)", None, None).unwrap();
            let projected = super::project_provider_defaults(&value).unwrap();
            assert_eq!(projected.vertex_project, None);
            assert_eq!(projected.vertex_location, None);
            assert!(!projected.enable_azure_ad_token_refresh);
            value.setattr("vertex_project", "project").unwrap();
            value.setattr("vertex_location", "region").unwrap();
            value
                .setattr("enable_azure_ad_token_refresh", true)
                .unwrap();
            let next = super::project_provider_defaults(&value).unwrap();
            assert_eq!(next.vertex_project.as_deref(), Some("project"));
            assert_eq!(next.vertex_location.as_deref(), Some("region"));
            assert!(next.enable_azure_ad_token_refresh);
            value.setattr("vertex_project", 1).unwrap();
            let error = super::project_provider_defaults(&value).err().unwrap();
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                error
                    .to_string()
                    .contains("provider_defaults.vertex_project")
            );
        });
    }

    #[test]
    fn a_readable_secret_manager_sends_the_call_back_to_python() {
        Python::initialize();
        Python::attach(|py| {
            let declined = process_environment_secrets(&secret_manager(py, true))
                .err()
                .expect("the Rust route declines");
            assert!(declined.is_instance_of::<RustBridgeDeclined>(py));
        });
    }

    #[test]
    fn without_a_readable_secret_manager_secrets_are_the_process_environment() {
        Python::initialize();
        Python::attach(|py| {
            let secrets = process_environment_secrets(&secret_manager(py, false)).unwrap();
            assert_eq!(
                secrets.get("LITELLM_RUST_BRIDGE_UNSET_VARIABLE_FOR_TEST"),
                None
            );
            assert_eq!(secrets.get("PATH"), std::env::var("PATH").ok());
        });
    }
}
