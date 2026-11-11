use crate::{jwt::JwtRequest, key::KeyRequest, plan};

#[derive(Clone, Copy, Debug, Default)]
pub struct Request<'a> {
    pub route: &'a str,
    pub method: Option<&'a str>,
    pub token: Option<&'a str>,
    pub model: Option<&'a str>,
    /// The `x-litellm-team-id` header.
    pub team_header: Option<&'a str>,
    pub is_llm_api_route: bool,
    /// `RouteChecks.is_auth_enforced_pass_through_route(route, method)`, read from the host's
    /// registered pass-through endpoints.
    pub auth_enforced_pass_through: bool,
    /// `InitPassThroughEndpointHelpers.forwarded_routes(route)`
    pub forwarded_routes: &'a [String],
    pub is_info_route: bool,
    /// A configured pass-through endpoint on this route has `auth: false`.
    pub pass_through_open: bool,
    /// Custom auth handed back `token` in place of the request's own credential.
    pub from_custom_auth: bool,
    pub now: f64,
}

impl<'a> Request<'a> {
    pub(crate) fn plan(&self) -> plan::Request<'a> {
        plan::Request {
            route: self.route,
            token: self.token.unwrap_or_default(),
            is_llm_api_route: self.is_llm_api_route,
            is_info_route: self.is_info_route,
            pass_through_open: self.pass_through_open,
            from_custom_auth: self.from_custom_auth,
        }
    }

    /// The key path authenticates `key`, which a pass-through endpoint may have read from another
    /// header than the token the plan was made on.
    pub(crate) fn key(&self, key: Option<&'a str>) -> KeyRequest<'a> {
        KeyRequest {
            route: self.route,
            token: key,
            now: self.now,
            is_llm_api_route: self.is_llm_api_route,
        }
    }

    pub(crate) fn jwt(&self) -> JwtRequest<'a> {
        JwtRequest {
            route: self.route,
            method: self.method,
            model: self.model,
            team_header: self.team_header,
            is_llm_api_route: self.is_llm_api_route,
            auth_enforced_pass_through: self.auth_enforced_pass_through,
            forwarded_routes: self.forwarded_routes,
            now: self.now,
        }
    }
}
