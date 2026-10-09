//! Provider selection for Python-hosted calls. `litellm/rust_bridge/host/provider.py` owns the answer,
//! which reads Python's live model registries; this module only carries the question across.

use pyo3::prelude::*;

const MODULE: &str = "litellm.rust_bridge.host.provider";

pub(crate) struct ResolvedProvider {
    pub(crate) model: String,
    pub(crate) custom_llm_provider: String,
}

/// `None` is where Python raises `BadRequestError`: no provider claims the model.
pub(crate) fn resolve(
    py: Python<'_>,
    model: &str,
    custom_llm_provider: Option<&str>,
    api_base: Option<&str>,
) -> PyResult<Option<ResolvedProvider>> {
    let resolved: Option<(String, String)> = py
        .import(MODULE)?
        .getattr("resolve_provider")?
        .call1((model, custom_llm_provider, api_base))?
        .extract()?;
    Ok(
        resolved.map(|(model, custom_llm_provider)| ResolvedProvider {
            model,
            custom_llm_provider,
        }),
    )
}
