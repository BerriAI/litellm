use litellm_gateway_auth_core::RouteGroups;
use parking_lot::Mutex;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

const MODULE: &str = "litellm.rust_bridge.host.route_groups";

/// The groups last built, beside the `route_groups()` value they were built from.
static BUILT: Mutex<Option<(Py<PyAny>, RouteGroups)>> = Mutex::new(None);

/// `route_groups()` as the core reads it. `LiteLLMRoutes.openai_routes` grows while the proxy runs,
/// as each auth-enforced pass-through endpoint registers, so the groups are rebuilt whenever the
/// Python value differs from the one they were built from.
pub(crate) fn route_groups(py: Python<'_>) -> PyResult<RouteGroups> {
    let current = py.import(MODULE)?.call_method0("route_groups")?;
    let cached = BUILT
        .lock()
        .as_ref()
        .map(|(source, groups)| (source.clone_ref(py), groups.clone()));
    if let Some((source, groups)) = cached
        && source.bind(py).eq(&current)?
    {
        return Ok(groups);
    }
    let groups = RouteGroups::new(
        current
            .call_method0("items")?
            .try_iter()?
            .map(|item| item?.extract::<(String, Vec<String>)>())
            .collect::<PyResult<Vec<_>>>()?,
    )
    .map_err(|missing| PyRuntimeError::new_err(missing.to_string()))?;
    *BUILT.lock() = Some((current.unbind(), groups.clone()));
    Ok(groups)
}
