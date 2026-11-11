use std::{sync::Arc, time::UNIX_EPOCH};

use axum::http::{StatusCode, request::Parts};
use litellm_error::Source;
use litellm_gateway_auth_core::{
    AuthHost, Identity, Outcome, PublicDenial, Request, Settings, authenticate_key,
    key::KeyIdentity,
};

use crate::{
    AuthFuture, Authenticator, Clock, Credential, Error, VerifiedIdentity,
    authentication::master_identity,
};

/// Authenticates with `gateway-auth-core`'s key path, asking `host` for what the core reads.
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
            let now = self
                .clock
                .now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            let request = Request {
                route: parts.uri.path(),
                method: Some(parts.method.as_str()),
                token: Some(token.expose()),
                now,
                ..Request::default()
            };
            match authenticate_key(&self.host, &request, request.token, &self.settings).await {
                Ok(Outcome::Authenticated(identity)) => verified(identity),
                Ok(Outcome::Denied(error, _)) => Err(rejected(error)),
                Err(error) => Err(Error::Unavailable(Source::new(error))),
            }
        })
    }
}

fn verified<R: litellm_gateway_auth_core::Rows>(
    identity: Identity<R>,
) -> Result<VerifiedIdentity, Error> {
    match identity {
        Identity::Key(KeyIdentity::MasterKey { .. }) => Ok(master_identity()),
        Identity::Key(KeyIdentity::NoMasterKey { .. }) => Err(Error::Unconfigured),
        // TODO: the gateway has no caller for these until its host reads keys, users and JWTs
        Identity::Key(
            KeyIdentity::UserAuthRoute
            | KeyIdentity::CachedAdmin { .. }
            | KeyIdentity::VirtualKey(_),
        )
        | Identity::MappedJwtKey(..)
        | Identity::AdminJwt(_)
        | Identity::FinishedJwt(_)
        | Identity::DbUnavailable(_) => Err(Error::Forbidden),
    }
}

fn rejected(error: litellm_gateway_auth_core::Error) -> Error {
    let Some(denial) = error.public() else {
        return Error::Unavailable(Source::new(error));
    };
    Error::Denied {
        status: StatusCode::from_u16(status(&denial)).unwrap_or(StatusCode::UNAUTHORIZED),
        message: error.to_string(),
    }
}

/// The status the proxy's auth exception handler answers each denial with.
fn status(denial: &PublicDenial) -> u16 {
    match denial {
        PublicDenial::Proxy { status, .. }
        | PublicDenial::Http { status, .. }
        | PublicDenial::InvalidVirtualKey { status, .. }
        | PublicDenial::ModelAccess { status, .. } => *status,
        PublicDenial::Plain { .. } => 401,
        // TODO: read `litellm_settings.budget_exceeded_status_code` once budgets run in the gateway
        PublicDenial::BudgetExceeded(_) => 422,
        PublicDenial::TeamNotFound { .. } => 404,
    }
}
