use litellm_gateway_auth_core::{
    matches_wildcard,
    routes::passthrough::{self, AllowedPassthrough, DeniedPassthrough, ForwardedRoutes},
};
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

#[pyfunction]
pub(crate) fn auth_first_denied_route(
    forwarded_routes: Vec<String>,
    denied_routes: Vec<String>,
) -> Option<String> {
    DeniedPassthrough(&denied_routes)
        .first_match(ForwardedRoutes(&forwarded_routes))
        .map(str::to_owned)
}

#[pyfunction]
pub(crate) fn auth_jwt_team_routes_grant(
    py: Python<'_>,
    route: &str,
    team_allowed_routes: Vec<String>,
) -> PyResult<bool> {
    Ok(route_groups(py)?
        .allowlist(&team_allowed_routes)
        .grants_pass_through(route))
}

#[pyfunction]
#[pyo3(signature = (route, *, key, team))]
pub(crate) fn auth_allowed_routes_grant(
    route: &str,
    key: Option<Vec<String>>,
    team: Option<Vec<String>>,
) -> bool {
    AllowedPassthrough {
        key: key.as_deref(),
        team: team.as_deref(),
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
