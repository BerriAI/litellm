use std::collections::HashSet;

use http::Method;

use super::TeamEligibility;
use crate::{
    KeyDenial,
    authn::{
        jwt::{JwtRequest, JwtSettings},
        key::KeySettings,
    },
    routes::passthrough::{ForwardedRoutes, PassthroughRequest},
};

/// What a request asks to call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target<'a> {
    pub(crate) route: &'a str,
    pub(crate) method: Option<&'a Method>,
    pub(crate) model: Option<&'a str>,
    pub(crate) forwarded_routes: ForwardedRoutes<'a>,
    /// `RouteChecks.is_auth_enforced_pass_through_route(route, method)`
    pub(crate) auth_enforced_pass_through: bool,
}

impl<'a> From<&JwtRequest<'a>> for Target<'a> {
    fn from(request: &JwtRequest<'a>) -> Self {
        Self {
            route: request.route,
            method: request.method,
            model: request.model,
            forwarded_routes: request.forwarded_routes,
            auth_enforced_pass_through: request.auth_enforced_pass_through,
        }
    }
}

/// The route rules: `master_key_only_routes`, the admin, team and role allowlists, the scope
/// mappings and the managed-agent routes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoutePolicy<'a> {
    pub(super) master_key_only_routes: &'a HashSet<String>,
    pub(super) jwt: &'a JwtSettings,
}

impl<'a> RoutePolicy<'a> {
    pub(crate) fn new(key: &'a KeySettings, jwt: &'a JwtSettings) -> Self {
        Self {
            master_key_only_routes: &key.master_key_only_routes,
            jwt,
        }
    }

    /// `master_key_only_routes`, which only the master key may call.
    pub(crate) fn key_credential(
        &self,
        presented_master_key: bool,
        route: &str,
    ) -> Result<(), KeyDenial> {
        if presented_master_key || !self.master_key_only_routes.contains(route) {
            return Ok(());
        }
        Err(KeyDenial::MasterKeyOnlyRoute {
            route: route.to_owned(),
        })
    }

    /// What a team has to allow for this target, which identity's team selection reads.
    pub(crate) fn team_eligibility(&self, target: &Target<'a>) -> TeamEligibility<'a> {
        let route_listed = self.jwt.team_routes().permits(target.route);
        TeamEligibility {
            route_listed,
            route_allowed: target.auth_enforced_pass_through || route_listed,
            passthrough: PassthroughRequest {
                route: target.route,
                auth_enforced: target.auth_enforced_pass_through,
                forwarded_routes: target.forwarded_routes,
                jwt_team_allowed_routes: Some(self.jwt.team_routes()),
            },
        }
    }
}
