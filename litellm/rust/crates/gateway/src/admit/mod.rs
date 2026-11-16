mod config;

use axum::{body::HttpBody, extract::Request, http::request::Parts};
use litellm_config::Config;
use litellm_error::Report;
use litellm_gateway_auth::{ApiKey, Auth, AuthenticatedRequest};
use litellm_gateway_auth_core::{
    PremiumRequired, RouteGroups, Token,
    plan::{self, Plan},
};

pub use config::{Unsupported, unsupported};

/// Why the gateway does not serve a request the way the Python proxy would.
#[derive(Debug, thiserror::Error, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum Refused {
    // TODO: no body is on the allowlist until the gateway runs `is_request_body_safe` and knows
    // which body fields mean something to Python alone
    #[error("the request has a body")]
    Body,
    #[error("the request carries no key")]
    NoKey,
    #[error("the proxy authenticates the request as {0:?}, and the gateway only as a key")]
    Plan(Plan),
    #[error("the proxy authenticates the request with a license the gateway can't check: {0:?}")]
    License(PremiumRequired),
    #[error("the gateway did not authenticate the caller: {}", Report::new(.0))]
    Unauthenticated(litellm_gateway_auth::Error),
}

/// Decides, request by request, whether the gateway serves it the way the Python proxy would.
#[derive(Clone)]
pub struct Admission {
    auth: Auth,
    plans: plan::Settings,
}

impl Admission {
    /// The gateway can't check a license, so every authentication path that needs one is refused,
    /// and it has no route groups, which list no inference route.
    pub fn new(config: &Config, auth: Auth) -> Self {
        Self {
            auth,
            plans: plan::Settings::new(
                &config.general_settings,
                false,
                false,
                &RouteGroups::default(),
            ),
        }
    }

    /// Lets in a request that has no body, that the proxy would authenticate as a key, and whose
    /// key the gateway authenticates, with the caller in its extensions. Every other request
    /// comes back as the client sent it, with why. This only authenticates. What the caller may
    /// do is still each handler's check.
    pub async fn admit(self, request: Request) -> Result<Request, (Box<Request>, Refused)> {
        let (mut parts, body) = request.into_parts();
        match self.caller(&parts, !body.is_end_stream()).await {
            Ok(caller) => {
                parts.extensions.insert(caller);
                Ok(Request::from_parts(parts, body))
            }
            Err(refused) => Err((Box::new(Request::from_parts(parts, body)), refused)),
        }
    }

    async fn caller(&self, parts: &Parts, has_body: bool) -> Result<AuthenticatedRequest, Refused> {
        if has_body {
            return Err(Refused::Body);
        }
        let key = ApiKey::from_headers(&parts.headers).map_err(|_| Refused::NoKey)?;
        let token = Token::new(key.0.clone());
        let request = plan::Request {
            route: parts.uri.path(),
            token: Some(&token),
            is_llm_api_route: true,
            ..plan::Request::default()
        };
        match plan::plan(&request, &self.plans).map_err(Refused::License)? {
            Plan::Key => {}
            other => return Err(Refused::Plan(other)),
        }
        self.auth
            .authenticate_request(&key.into(), parts)
            .await
            .map_err(Refused::Unauthenticated)
    }
}
