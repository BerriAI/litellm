use std::sync::Arc;

use axum::http::request::Parts;
use jiff::Timestamp;
use litellm_error::Source;
use litellm_gateway_auth_core::{
    AuthHost, Identity, Outcome, Request, Rows, Settings, authenticate_jwt, authenticate_key,
    identity::key::KeyIdentity, plan, plan::Plan,
};

use crate::{
    AuthFuture, Authenticator, Clock, Credential, Error, VerifiedIdentity,
    authentication::master_identity,
};

/// Authenticates with `gateway-auth-core`, asking `host` for what the core reads. It runs the key
/// and JWT paths and refuses every other plan as [`Error::Unsupported`], so it only authenticates
/// and never authorizes.
pub struct CoreAuthenticator<H> {
    host: H,
    settings: Settings,
    clock: Arc<dyn Clock>,
}

impl<H> CoreAuthenticator<H> {
    pub fn new(host: H, settings: Settings, clock: Arc<dyn Clock>) -> Self {
        Self {
            host,
            settings,
            clock,
        }
    }
}

impl<H: AuthHost> Authenticator for CoreAuthenticator<H> {
    fn validate(&self) -> AuthFuture<'_, ()> {
        let configured = self.settings.key.master_key.is_some();
        Box::pin(async move { configured.then_some(()).ok_or(Error::Unconfigured) })
    }

    fn verify<'a>(
        &'a self,
        credential: &'a Credential,
        parts: &'a Parts,
    ) -> AuthFuture<'a, VerifiedIdentity> {
        Box::pin(async move {
            let Credential::Token(token) = credential else {
                return Err(Error::InvalidToken);
            };
            let request = Request {
                route: parts.uri.path(),
                method: Some(&parts.method),
                token: Some(token),
                // Every route the gateway mounts is an LLM API route.
                is_llm_api_route: true,
                now: Timestamp::try_from(self.clock.now())
                    .map_err(|error| Error::Unavailable(Source::new(error)))?,
                ..Request::default()
            };
            let outcome = match plan(&request, &self.settings) {
                Ok(Plan::Key) => {
                    authenticate_key(&self.host, &request, request.token, &self.settings).await
                }
                Ok(Plan::Jwt) => authenticate_jwt(&self.host, &request, &self.settings).await,
                Ok(
                    Plan::Public
                    | Plan::OAuth2Introspection
                    | Plan::OAuth2Proxy
                    | Plan::JwtWithOperatorCode
                    | Plan::PassThroughOpen,
                )
                | Err(_) => return Err(Error::Unsupported),
            };
            match outcome {
                Ok(Outcome::Authenticated(identity)) => verified(identity),
                Ok(Outcome::Denied(denial, _)) => Err(Error::Denied(denial)),
                Err(failure) => Err(Error::Unavailable(Source::new(failure))),
            }
        })
    }
}

fn verified<R: Rows>(identity: Identity<R>) -> Result<VerifiedIdentity, Error> {
    match identity {
        Identity::Key(KeyIdentity::MasterKey { .. }) => Ok(master_identity()),
        Identity::Key(KeyIdentity::NoMasterKey { .. }) => Err(Error::Unconfigured),
        // TODO: the gateway has no caller for these until a host reads keys, users and JWT rows
        Identity::Key(
            KeyIdentity::UserAuthRoute
            | KeyIdentity::CachedAdmin { .. }
            | KeyIdentity::VirtualKey(_),
        )
        | Identity::MappedJwtKey(..)
        | Identity::AdminJwt(_)
        | Identity::FinishedJwt(_) => Err(Error::Unsupported),
    }
}
