use litellm_gateway_auth_core::{managed_route, passthrough};
use pyo3::prelude::*;

#[pyfunction]
pub(crate) fn auth_route_matches_wildcard(route: &str, pattern: &str) -> bool {
    passthrough::matches_wildcard(route, pattern)
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
pub(crate) fn auth_first_denied_route(forwarded_routes: Vec<String>, denied_routes: Vec<String>) -> Option<String> {
    passthrough::first_denied_route(&forwarded_routes, &denied_routes).map(str::to_owned)
}

#[pyfunction]
pub(crate) fn auth_jwt_team_routes_grant(route: &str, team_allowed_routes: Vec<String>) -> bool {
    passthrough::jwt_team_routes_grant(route, &team_allowed_routes)
}

#[pyfunction]
#[pyo3(signature = (route, *, key, team))]
pub(crate) fn auth_allowed_routes_grant(route: &str, key: Option<Vec<String>>, team: Option<Vec<String>>) -> bool {
    passthrough::allowed_routes_grant(route, key.as_deref(), team.as_deref())
}

/// The kind of route a managed agent identity calls: `discovery`, `realtime`, `live`, `model`,
/// `tool` or `other`.
#[pyfunction]
pub(crate) fn auth_managed_route(route: &str) -> &'static str {
    managed_route(route).into()
}

#[pyfunction]
#[pyo3(signature = (route, method = None))]
pub(crate) fn auth_managed_agent_route_allowed(route: &str, method: Option<&str>) -> bool {
    litellm_gateway_auth_core::managed_agent_route_allowed(route, method)
}
