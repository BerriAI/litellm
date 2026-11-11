use litellm_gateway_auth_core::route_access::{
    self, OrgMembership, RouteAccess, RouteRequest, RouteToken, RouteUser,
};
use pyo3::prelude::*;

use super::call::DenialView;

/// `route_checks.RouteFacts`: what the route checks read from the request, the user row and the token.
#[derive(FromPyObject)]
pub(crate) struct RouteFacts {
    #[pyo3(attribute)]
    route: String,
    /// `RouteChecks.get_request_method`, upper-cased.
    #[pyo3(attribute)]
    method: Option<String>,
    #[pyo3(attribute)]
    route_template: Option<String>,
    #[pyo3(attribute)]
    query_user_id: Option<String>,
    #[pyo3(attribute)]
    body_keys: Vec<String>,
    #[pyo3(attribute)]
    organization_id: Option<String>,
    #[pyo3(attribute)]
    organizations: Option<Vec<String>>,
    #[pyo3(attribute)]
    team_id: Option<String>,
    #[pyo3(attribute)]
    is_llm_api_route: bool,
    #[pyo3(attribute)]
    auth_enforced_pass_through: bool,
    #[pyo3(attribute)]
    forwarded_routes: Vec<String>,
    #[pyo3(attribute)]
    admin_only_routes: Vec<String>,
    #[pyo3(attribute)]
    has_user: bool,
    #[pyo3(attribute)]
    user_id: Option<String>,
    #[pyo3(attribute)]
    user_role: Option<String>,
    /// `(organization_id, user_role)` per membership, unset when the user row has none.
    #[pyo3(attribute)]
    organization_memberships: Option<Vec<(String, Option<String>)>>,
    #[pyo3(attribute)]
    has_token: bool,
    #[pyo3(attribute)]
    token_user_id: Option<String>,
    #[pyo3(attribute)]
    can_read_spend: bool,
    #[pyo3(attribute)]
    is_team_service_account: bool,
    #[pyo3(attribute)]
    allowed_routes: Option<Vec<String>>,
    #[pyo3(attribute)]
    key_passthrough_routes: Option<Vec<String>>,
    #[pyo3(attribute)]
    team_passthrough_routes: Option<Vec<String>>,
    #[pyo3(attribute)]
    denied_passthrough_routes: Vec<String>,
    #[pyo3(attribute)]
    jwt_team_allowed_routes: Option<Vec<String>>,
}

impl RouteFacts {
    pub(crate) fn memberships(&self) -> Option<Vec<OrgMembership<'_>>> {
        self.organization_memberships.as_ref().map(|memberships| {
            memberships
                .iter()
                .map(|(organization_id, role)| OrgMembership {
                    organization_id,
                    role: role.as_deref(),
                })
                .collect()
        })
    }

    pub(crate) fn access<'a>(&'a self, memberships: Option<&'a [OrgMembership<'a>]>) -> RouteAccess<'a> {
        RouteAccess {
            request: RouteRequest {
                route: &self.route,
                method: self.method.as_deref(),
                route_template: self.route_template.as_deref(),
                query_user_id: self.query_user_id.as_deref(),
                body_keys: &self.body_keys,
                organization_id: self.organization_id.as_deref(),
                organizations: self.organizations.as_deref(),
                team_id: self.team_id.as_deref(),
                is_llm_api_route: self.is_llm_api_route,
                auth_enforced_pass_through: self.auth_enforced_pass_through,
                forwarded_routes: &self.forwarded_routes,
                admin_only_routes: &self.admin_only_routes,
            },
            user: self.has_user.then(|| RouteUser {
                user_id: self.user_id.as_deref(),
                user_role: self.user_role.as_deref(),
                organization_memberships: memberships,
            }),
            token: self.has_token.then(|| RouteToken {
                user_id: self.token_user_id.as_deref(),
                can_read_spend: self.can_read_spend,
                is_team_service_account: self.is_team_service_account,
                allowed_routes: self.allowed_routes.as_deref(),
                key_passthrough_routes: self.key_passthrough_routes.as_deref(),
                team_passthrough_routes: self.team_passthrough_routes.as_deref(),
                denied_passthrough_routes: &self.denied_passthrough_routes,
                jwt_team_allowed_routes: self.jwt_team_allowed_routes.as_deref(),
            }),
        }
    }
}

fn denial(result: Result<(), litellm_gateway_auth_core::RouteDenial>) -> Option<DenialView> {
    result.err().map(|denial| denial.public().into())
}

/// `organization_role_based_access_check`
#[pyfunction]
pub(crate) fn auth_organization_denial(facts: RouteFacts) -> Option<DenialView> {
    let memberships = facts.memberships();
    denial(route_access::organization_access(&facts.access(memberships.as_deref())))
}

/// `_is_api_route_allowed`, with the organization the request acts in.
#[pyfunction]
#[pyo3(signature = (facts, organization_id = None))]
pub(crate) fn auth_route_denial(facts: RouteFacts, organization_id: Option<String>) -> Option<DenialView> {
    let memberships = facts.memberships();
    denial(route_access::route_allowed(
        &facts.access(memberships.as_deref()),
        organization_id.as_deref(),
    ))
}

/// `_check_proxy_admin_viewer_access`
#[pyfunction]
pub(crate) fn auth_viewer_route_denial(facts: RouteFacts) -> Option<DenialView> {
    let memberships = facts.memberships();
    denial(route_access::viewer_access(&facts.access(memberships.as_deref()).request))
}

/// `user_is_org_admin`
#[pyfunction]
pub(crate) fn auth_user_is_org_admin(facts: RouteFacts) -> bool {
    let memberships = facts.memberships();
    let access = facts.access(memberships.as_deref());
    route_access::is_org_admin(access.user, access.request.organization_id, access.request.organizations)
}

/// The team whose organization a team route acts in, when the request names none itself.
#[pyfunction]
pub(crate) fn auth_org_context_team(facts: RouteFacts) -> Option<String> {
    let memberships = facts.memberships();
    route_access::org_context_team(&facts.access(memberships.as_deref()).request).map(str::to_owned)
}

/// `RouteChecks._mask_user_id`
#[pyfunction]
pub(crate) fn auth_mask_user_id(user_id: &str) -> String {
    route_access::masked_user_id(user_id)
}
