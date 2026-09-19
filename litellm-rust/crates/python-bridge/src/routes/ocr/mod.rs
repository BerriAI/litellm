mod document;
mod errors;
mod host;
mod project;

use std::sync::{Arc, LazyLock};

use host::OcrRouteHost;
use litellm_auth_gcp::VertexAuth;
use litellm_callbacks_legacy::{LegacySurface, PublicCall, run_legacy_call};
use litellm_core::ocr::route::ocr_machine;
use litellm_core_utils::settings::ProcessEnvironment;
use litellm_llms::base_llm::ocr::{handler::OcrClient, settings::OcrSettings};
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

use crate::{
    errors::RustBridgeDeclined,
    http,
    python_settings::{PythonSecrets, PythonSettings},
};

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
    let config = http::call_config(py, &kwargs, asynchronous)?;
    let client = OcrClient::new(
        http::pool(),
        &config,
        http::url_policy(py)?,
        VERTEX_AUTH.clone(),
        ocr_settings(py)?,
        Arc::new(PythonSecrets),
    )
    .map_err(|error| RustBridgeDeclined::new_err(error.to_string()))?;
    run_legacy_call(
        py,
        if asynchronous { ASYNC_SURFACE } else { SURFACE },
        PublicCall::capture(&request, &args, &kwargs)?,
        ocr_machine(client),
        OcrRouteHost::new(request.unbind()),
        asynchronous,
    )
}

#[derive(FromPyObject)]
struct PythonProviderDefaults {
    vertex_project: Option<String>,
    vertex_location: Option<String>,
    enable_azure_ad_token_refresh: Option<bool>,
}

fn ocr_settings(py: Python<'_>) -> PyResult<OcrSettings> {
    let defaults: PythonProviderDefaults = PythonSettings::ProviderDefaults
        .read(py)?
        .extract()
        .map_err(|error: PyErr| {
            RustBridgeDeclined::new_err(format!(
                "litellm provider defaults cannot be used by the Rust route: {error}"
            ))
        })?;
    Ok(OcrSettings {
        vertex_project: defaults.vertex_project,
        vertex_location: defaults.vertex_location,
        enable_azure_ad_token_refresh: defaults.enable_azure_ad_token_refresh == Some(true),
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
