use litellm_gateway_auth_core::RouteGroups;
use pyo3::{exceptions::PyRuntimeError, prelude::*, sync::PyOnceLock};

const MODULE: &str = "litellm.rust_bridge.host.route_groups";

/// `route_groups()`, read once per process since `LiteLLMRoutes` never changes while the proxy runs.
pub(crate) fn route_groups(py: Python<'_>) -> PyResult<&'static RouteGroups> {
    static GROUPS: PyOnceLock<RouteGroups> = PyOnceLock::new();
    GROUPS.get_or_try_init(py, || {
        let groups = py
            .import(MODULE)?
            .call_method0("route_groups")?
            .call_method0("items")?
            .try_iter()?
            .map(|item| item?.extract::<(String, Vec<String>)>())
            .collect::<PyResult<Vec<_>>>()?;
        RouteGroups::new(groups).map_err(|missing| PyRuntimeError::new_err(missing.to_string()))
    })
}
