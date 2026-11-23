use std::{future::Future, pin::Pin, sync::Arc};

use axum::http::request::Parts;
use jiff::Timestamp;
use litellm_gateway_auth_core::{
    self as auth_core, AuthHost, Caller, Outcome, Rows, Settings, Token, authenticate_key,
};
use litellm_secret_value::SecretValue;

use crate::{Authenticated, Bearer, CredentialExtractor, Error};

pub type AuthFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

pub trait Clock: Send + Sync {
    fn now(&self) -> Timestamp;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Timestamp {
        Timestamp::now()
    }
}

pub enum Credential {
    Token(SecretValue),
    Transport,
}

/// Who a selected credential stands for. The only answer is `gateway-auth-core`'s [`Caller`], so
/// nothing this crate hands a route was decided here.
pub trait Authenticator<R: Rows>: Send + Sync {
    fn authenticate<'a>(
        &'a self,
        credential: &'a Credential,
        request: &'a Parts,
    ) -> AuthFuture<'a, Caller<R>>;
}

pub struct Auth<R: Rows> {
    extractor: Arc<dyn CredentialExtractor>,
    authenticator: Arc<dyn Authenticator<R>>,
}

impl<R: Rows> Clone for Auth<R> {
    fn clone(&self) -> Self {
        Self {
            extractor: self.extractor.clone(),
            authenticator: self.authenticator.clone(),
        }
    }
}

impl<R: Rows> Auth<R> {
    pub fn new(authenticator: Arc<dyn Authenticator<R>>) -> Self {
        Self {
            extractor: Arc::new(Bearer),
            authenticator,
        }
    }

    pub fn with_extractor(self, extractor: Arc<dyn CredentialExtractor>) -> Self {
        Self { extractor, ..self }
    }

    pub async fn authenticate_parts(&self, request: &Parts) -> Result<Authenticated<R>, Error> {
        let credential = self.extractor.extract(request)?;
        self.authenticate_credential(&credential, request)
            .await
            .map_err(|error| self.extractor.reject(request, error))
    }

    pub(crate) async fn authenticate_credential(
        &self,
        credential: &Credential,
        request: &Parts,
    ) -> Result<Authenticated<R>, Error> {
        let caller = self.authenticator.authenticate(credential, request).await?;
        Ok(Authenticated::from(caller))
    }
}

/// Hands a credential to `gateway-auth-core`'s key path over the host `H`, which verifies the
/// master key and reads the keys `H` holds. Every route this crate guards is an inference route,
/// which is what the request it hands the core says.
pub struct KeyAuthenticator<H> {
    host: H,
    settings: Settings,
    clock: Arc<dyn Clock>,
}

impl<H> KeyAuthenticator<H> {
    pub fn new(host: H, settings: Settings, clock: Arc<dyn Clock>) -> Self {
        Self {
            host,
            settings,
            clock,
        }
    }
}

impl<H: AuthHost> Authenticator<H::Rows> for KeyAuthenticator<H> {
    fn authenticate<'a>(
        &'a self,
        credential: &'a Credential,
        request: &'a Parts,
    ) -> AuthFuture<'a, Caller<H::Rows>> {
        Box::pin(async move {
            let token = match credential {
                Credential::Token(presented) => Some(Token::new(presented.clone())),
                Credential::Transport => None,
            };
            let request = auth_core::Request {
                route: request.uri.path(),
                method: Some(&request.method),
                token: token.as_ref(),
                is_llm_api_route: true,
                runs_inference_checks: true,
                now: self.clock.now(),
                ..auth_core::Request::default()
            };
            match authenticate_key(&self.host, &request, token.as_ref(), &self.settings).await {
                Ok(Outcome::Authenticated(identity)) => Ok(Caller::try_from(identity)?),
                Ok(Outcome::Denied(error, _)) | Err(error) => Err(error.into()),
            }
        })
    }
}
