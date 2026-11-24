//! `classify_billable_request`'s route table, shared with the gateway's request volume middleware
//! so the proxy and the gateway count the same requests under the same routes.

use litellm_core_utils::billable_route::{self, PASSTHROUGH_PREFIXES};
use pyo3::prelude::*;

/// The category (`"llm"`, `"mcp"` or `"a2a"`) and the route a request counts under, or `None`
/// when it does not count.
#[pyfunction]
pub(crate) fn classify_billable_route(
    path: &str,
    method: &str,
) -> Option<(&'static str, &'static str)> {
    billable_route::classify(path, method).map(|(category, route)| (category.into(), route))
}

/// The provider pass-through prefixes a request below counts under, for the proxy's check that
/// they match `LiteLLMRoutes.mapped_pass_through_routes`.
#[pyfunction]
pub(crate) fn billable_route_passthrough_prefixes() -> Vec<&'static str> {
    PASSTHROUGH_PREFIXES.to_vec()
}
