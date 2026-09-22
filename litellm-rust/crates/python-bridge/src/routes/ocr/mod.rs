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
use litellm_llms::base_llm::ocr::{handler::OcrClient, settings::OcrSettings};
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

use crate::{
    http,
    python_settings::{Adapter, PythonSettings, SettingSpec, Snapshot},
    secrets,
};

const fn provider_default(name: &'static str, adapter: Adapter) -> SettingSpec {
    SettingSpec::new(PythonSettings::ProviderDefaults, name, adapter)
}

pub(crate) const VERTEX_PROJECT: SettingSpec =
    provider_default("vertex_project", Adapter::FalsyOptionalString).sensitive();
pub(crate) const VERTEX_LOCATION: SettingSpec =
    provider_default("vertex_location", Adapter::FalsyOptionalString).sensitive();
pub(crate) const ENABLE_AZURE_AD_TOKEN_REFRESH: SettingSpec =
    provider_default("enable_azure_ad_token_refresh", Adapter::ExactTrue);

#[cfg(test)]
pub(crate) const PROVIDER_DEFAULT_SPECS: &[SettingSpec] = &[
    VERTEX_PROJECT,
    VERTEX_LOCATION,
    ENABLE_AZURE_AD_TOKEN_REFRESH,
];

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
    let secret_manager = secrets::binding::resolve(py)?;
    let config = http::call_config(py, &kwargs, asynchronous)?;
    let client = OcrClient::new(
        http::pool(),
        &config,
        http::url_policy(py)?,
        VERTEX_AUTH.clone(),
        ocr_settings(py)?,
        Arc::new(secrets::resolved::ResolvedSecrets::new(secret_manager)),
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

fn ocr_settings(py: Python<'_>) -> PyResult<OcrSettings> {
    project_provider_defaults(&PythonSettings::ProviderDefaults.read(py)?)
}

fn project_provider_defaults(snapshot: &Snapshot<'_>) -> PyResult<OcrSettings> {
    Ok(OcrSettings {
        vertex_project: snapshot.field(&VERTEX_PROJECT)?.falsy_optional_string()?.0,
        vertex_location: snapshot.field(&VERTEX_LOCATION)?.falsy_optional_string()?.0,
        enable_azure_ad_token_refresh: snapshot
            .field(&ENABLE_AZURE_AD_TOKEN_REFRESH)?
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
