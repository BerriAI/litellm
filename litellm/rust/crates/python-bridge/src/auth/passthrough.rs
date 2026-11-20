use litellm_config::RouteList;
use litellm_gateway_auth_core::{
    matches_wildcard,
    routes::passthrough::{self, AllowedPassthrough, DeniedPassthrough, ForwardedRoutes},
};
use litellm_host_python::from_py;
use pyo3::prelude::*;

use super::{convert::parse_method, route_groups::route_groups};

#[pyfunction]
pub(crate) fn auth_route_matches_wildcard(route: &str, pattern: &str) -> bool {
    matches_wildcard(route, pattern)
}

#[pyfunction]
pub(crate) fn auth_route_matches_prefix(route: &str, allowed_route: &str) -> bool {
    passthrough::matches_prefix(route, allowed_route)
}

#[pyfunction]
pub(crate) fn auth_route_matches_denied(route: &str, denied_route: &str) -> bool {
    passthrough::matches_denied(route, denied_route)
}

/// `denied_sources` are the raw `denied_passthrough_routes` metadata values, key first.
#[pyfunction]
pub(crate) fn auth_first_denied_route(
    forwarded_routes: Vec<String>,
    denied_sources: Vec<Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let denied = denied_sources
        .iter()
        .map(route_list)
        .filter_map(Result::transpose)
        .collect::<PyResult<Vec<_>>>()?;
    Ok(DeniedPassthrough(&denied)
        .first_match(ForwardedRoutes(&forwarded_routes))
        .map(str::to_owned))
}

#[pyfunction]
pub(crate) fn auth_jwt_team_routes_grant(
    py: Python<'_>,
    route: &str,
    #[pyo3(from_py_with = route_list)] team_allowed_routes: Option<RouteList>,
) -> PyResult<bool> {
    let entries = team_allowed_routes
        .as_ref()
        .map(RouteList::items)
        .unwrap_or_default();
    Ok(route_groups(py)?
        .allowlist(entries)
        .grants_pass_through(route))
}

/// A raw `allowed_passthrough_routes` or `denied_passthrough_routes` metadata value. A falsy one
/// is absent, as `metadata.get(...) or ...` reads it. One that does not read as a route list is
/// `Malformed`, so it denies instead of failing the request.
pub(crate) fn route_list(value: &Bound<'_, PyAny>) -> PyResult<Option<RouteList>> {
    Ok(value
        .is_truthy()?
        .then(|| from_py(value).unwrap_or(RouteList::Malformed)))
}

/// `key` and `team` are the raw `allowed_passthrough_routes` metadata values.
#[pyfunction]
#[pyo3(signature = (route, *, key, team))]
pub(crate) fn auth_allowed_routes_grant(
    route: &str,
    #[pyo3(from_py_with = route_list)] key: Option<RouteList>,
    #[pyo3(from_py_with = route_list)] team: Option<RouteList>,
) -> bool {
    AllowedPassthrough {
        key: key.as_ref(),
        team: team.as_ref(),
    }
    .grants(route)
}

/// The kind of route a managed agent identity calls: `discovery`, `realtime`, `live`, `model`,
/// `tool` or `other`.
#[pyfunction]
pub(crate) fn auth_managed_route(py: Python<'_>, route: &str) -> PyResult<&'static str> {
    Ok(route_groups(py)?.managed_route(route).into())
}

#[pyfunction]
#[pyo3(signature = (route, method = None))]
pub(crate) fn auth_managed_agent_route_allowed(
    py: Python<'_>,
    route: &str,
    method: Option<&str>,
) -> PyResult<bool> {
    let method = method.map(parse_method).transpose()?;
    Ok(route_groups(py)?.managed_agent_route_allowed(route, method.as_ref()))
}
