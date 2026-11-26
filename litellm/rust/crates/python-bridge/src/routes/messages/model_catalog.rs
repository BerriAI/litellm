use std::sync::{Arc, Mutex, PoisonError};

use litellm_host_python::json_from_py;
use litellm_llms::{
    anthropic::model_capabilities::{
        bedrock_model_capabilities, messages_route_capabilities, with_base_model_user_profile_id,
    },
    base_llm::messages::context::{BedrockCapabilities, MessagesModelCapabilities},
    vertex_ai::common_utils::supported_regions,
};
use litellm_model_catalog::Catalog;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyMapping},
};

use super::host::MODEL_CAPABILITIES_MODULE;

type Version = (usize, usize, u64);

static LIVE: Mutex<Option<(Version, Arc<Catalog>)>> = Mutex::new(None);

fn live_catalog(py: Python<'_>) -> PyResult<Arc<Catalog>> {
    let known = LIVE.lock().unwrap_or_else(PoisonError::into_inner).clone();
    let (version, document): (Version, Option<Bound<'_, PyAny>>) = py
        .import(MODEL_CAPABILITIES_MODULE)?
        .getattr("model_cost_document")?
        .call1((known.as_ref().map(|(version, _)| *version),))?
        .extract()?;
    let Some(document) = document else {
        return known
            .map(|(_, catalog)| catalog)
            .ok_or_else(|| PyValueError::new_err("the model map came back without a document"));
    };
    let body = serde_json::to_vec(&json_from_py(&document)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let catalog = Arc::new(
        py.detach(|| Catalog::parse(&body))
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .value,
    );
    *LIVE.lock().unwrap_or_else(PoisonError::into_inner) = Some((version, Arc::clone(&catalog)));
    Ok(catalog)
}

/// The live model map's `supported_regions` for a Claude on Vertex AI model.
pub(super) fn vertex_supported_regions(py: Python<'_>, model: &str) -> PyResult<Vec<String>> {
    Ok(supported_regions(&*live_catalog(py)?, model))
}

fn declared_flags<'py>(
    model_info: Option<Bound<'py, PyAny>>,
) -> impl Fn(&str) -> Option<bool> + use<'py> {
    let model_info = model_info.and_then(|info| info.cast_into::<PyMapping>().ok());
    move |name: &str| {
        model_info
            .as_ref()
            .and_then(|info| info.get_item(name).ok())
            .and_then(|value| value.cast_into::<PyBool>().ok())
            .map(|flag| flag.is_true())
    }
}

pub(super) fn bedrock_capabilities(
    py: Python<'_>,
    model: &str,
    base_model: Option<&str>,
    model_info: Option<Bound<'_, PyAny>>,
) -> PyResult<BedrockCapabilities> {
    Ok(bedrock_model_capabilities(
        &*live_catalog(py)?,
        model,
        base_model,
        &declared_flags(model_info),
    ))
}

/// The capabilities a route reads, all from the Rust catalog of the live model map.
pub(super) fn route_capabilities(
    py: Python<'_>,
    model: &str,
    custom_llm_provider: Option<&str>,
    base_model: Option<&str>,
    model_info: Option<Bound<'_, PyAny>>,
) -> PyResult<MessagesModelCapabilities> {
    let catalog = live_catalog(py)?;
    let declared = declared_flags(model_info);
    Ok(with_base_model_user_profile_id(
        messages_route_capabilities(&catalog, model, custom_llm_provider, None, &declared),
        &catalog,
        base_model,
        custom_llm_provider,
        &declared,
    ))
}
