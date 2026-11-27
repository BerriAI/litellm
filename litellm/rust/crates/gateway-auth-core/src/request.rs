use http::Method;
use jiff::Timestamp;

use crate::{
    Error, Token,
    authn::{jwt::JwtRequest, key::KeyRequest},
    plan,
    routes::passthrough::ForwardedRoutes,
};

/// The model a request's body names. A host that authenticates before it reads the body says
/// `Unread`, and a decision that depends on the model is then one that host does not serve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Model<'a> {
    #[default]
    Unread,
    Named(Option<&'a str>),
}

impl<'a> Model<'a> {
    pub(crate) fn read(self) -> Result<Option<&'a str>, Error> {
        match self {
            Self::Unread => Err(Error::HostDeclined),
            Self::Named(model) => Ok(model),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Request<'a> {
    pub route: &'a str,
    pub method: Option<&'a Method>,
    pub token: Option<&'a Token>,
    pub model: Model<'a>,
    /// The end-user id the request's body and headers name, by Python's `get_end_user_id_from_request_body`.
    pub end_user_id: Option<&'a str>,
    /// The `x-litellm-team-id` header.
    pub team_header: Option<&'a str>,
    pub is_llm_api_route: bool,
    /// `RouteChecks.runs_inference_checks(route)`: an LLM API route or the queued chat route.
    pub runs_inference_checks: bool,
    /// `RouteChecks.is_auth_enforced_pass_through_route(route, method)`, read from the host's
    /// registered pass-through endpoints.
    pub auth_enforced_pass_through: bool,
    pub forwarded_routes: ForwardedRoutes<'a>,
    pub is_info_route: bool,
    /// A configured pass-through endpoint on this route has `auth: false`.
    pub pass_through_open: bool,
    /// Custom auth handed back `token` in place of the request's own credential.
    pub from_custom_auth: bool,
    pub now: Timestamp,
}

impl<'a> Request<'a> {
    pub(crate) fn plan(&self) -> plan::Request<'a> {
        plan::Request {
            route: self.route,
            token: self.token,
            is_llm_api_route: self.is_llm_api_route,
            is_info_route: self.is_info_route,
            pass_through_open: self.pass_through_open,
            from_custom_auth: self.from_custom_auth,
        }
    }

    pub(crate) fn key(&self) -> KeyRequest<'a> {
        KeyRequest {
            route: self.route,
            end_user_id: self.end_user_id,
            now: self.now,
            runs_inference_checks: self.runs_inference_checks,
        }
    }

    /// The JWT path reads a missing token as an empty one, which it refuses as malformed.
    pub(crate) fn token_or_empty(&self) -> Token {
        self.token.cloned().unwrap_or_else(|| Token::new(""))
    }

    pub(crate) fn jwt(&self) -> JwtRequest<'a> {
        JwtRequest {
            route: self.route,
            method: self.method,
            model: self.model,
            team_header: self.team_header,
            runs_inference_checks: self.runs_inference_checks,
            auth_enforced_pass_through: self.auth_enforced_pass_through,
            forwarded_routes: self.forwarded_routes,
            now: self.now,
        }
    }
}
