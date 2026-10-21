use std::{future::Future, pin::Pin, sync::Arc, time::SystemTime};

use axum::http::{Request, request::Parts};
use litellm_auth_types::SecretValue;
use litellm_config::Config;
use litellm_secrets::source::SecretSource;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::{
    AuthenticatedCaller, AuthenticatedRequest, Authentication, AuthenticationMethod, Authorizer,
    Bearer, CredentialExtractor, Error, NoAdditionalPolicy, Permissions, Principal, PrincipalKind,
    ResolvedIdentity, VerifiedIdentity,
};

pub type AuthFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

pub trait Clock: Send + Sync {
    fn now(&self) -> SystemTime;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

pub enum Credential {
    Token(SecretValue),
    Transport,
}

pub trait Authenticator: Send + Sync {
    fn validate(&self) -> AuthFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn verify<'a>(
        &'a self,
        credential: &'a Credential,
        request: &'a Parts,
    ) -> AuthFuture<'a, VerifiedIdentity>;
}

pub trait IdentityResolver: Send + Sync {
    fn resolve<'a>(&'a self, identity: &'a VerifiedIdentity) -> AuthFuture<'a, ResolvedIdentity>;
}

pub struct LocalAdministrator;

impl IdentityResolver for LocalAdministrator {
    fn resolve<'a>(&'a self, identity: &'a VerifiedIdentity) -> AuthFuture<'a, ResolvedIdentity> {
        Box::pin(async move {
            let allowed = match identity.authentication.method {
                AuthenticationMethod::MasterKey => identity.principal == master_principal(),
                AuthenticationMethod::Session => {
                    identity.principal.authority() == "litellm:local-ui"
                        && identity.principal.kind() == PrincipalKind::Human
                }
                AuthenticationMethod::External(_) => false,
            };
            if !allowed {
                return Err(Error::Forbidden);
            }
            Ok(ResolvedIdentity {
                principal: identity.principal.clone(),
                permissions: Permissions::All,
            })
        })
    }
}

#[derive(Clone)]
pub struct Auth {
    extractor: Arc<dyn CredentialExtractor>,
    authenticator: Arc<dyn Authenticator>,
    identities: Arc<dyn IdentityResolver>,
    authorizer: Arc<dyn Authorizer>,
    clock: Arc<dyn Clock>,
}

impl Auth {
    pub fn new(
        authenticator: Arc<dyn Authenticator>,
        identities: Arc<dyn IdentityResolver>,
        authorizer: Arc<dyn Authorizer>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            extractor: Arc::new(Bearer),
            authenticator,
            identities,
            authorizer,
            clock,
        }
    }

    pub fn from_config(config: &Config, secrets: Arc<dyn SecretSource>) -> Self {
        let master = Arc::new(MasterKeyAuthenticator {
            key: config.general_settings.master_key.clone(),
            secrets,
        });
        Self {
            extractor: Arc::new(Bearer),
            authenticator: master,
            identities: Arc::new(LocalAdministrator),
            authorizer: Arc::new(NoAdditionalPolicy),
            clock: Arc::new(SystemClock),
        }
    }

    pub fn with_extractor(self, extractor: Arc<dyn CredentialExtractor>) -> Self {
        Self { extractor, ..self }
    }

    pub async fn authenticate_parts(&self, request: &Parts) -> Result<AuthenticatedRequest, Error> {
        let credential = self.extractor.extract(request)?;
        self.authenticate_request(&credential, request).await
    }

    pub async fn validate(&self) -> Result<(), Error> {
        self.authenticator.validate().await
    }

    pub async fn authenticate(
        &self,
        credential: &SecretValue,
    ) -> Result<AuthenticatedRequest, Error> {
        self.authenticate_request(
            &Credential::Token(credential.clone()),
            &Request::new(()).into_parts().0,
        )
        .await
    }

    pub async fn authenticate_request(
        &self,
        credential: &Credential,
        request: &Parts,
    ) -> Result<AuthenticatedRequest, Error> {
        let verified = self.authenticator.verify(credential, request).await?;
        resolve(
            verified,
            self.identities.as_ref(),
            self.authorizer.clone(),
            self.clock.clone(),
        )
        .await
    }
}

pub(crate) async fn resolve(
    verified: VerifiedIdentity,
    identities: &dyn IdentityResolver,
    authorizer: Arc<dyn Authorizer>,
    clock: Arc<dyn Clock>,
) -> Result<AuthenticatedRequest, Error> {
    if verified
        .authentication
        .expires_at
        .is_some_and(|expiry| expiry <= clock.now())
    {
        return Err(Error::Expired);
    }
    let resolved = identities.resolve(&verified).await?;
    Ok(AuthenticatedRequest::new(
        Arc::new(AuthenticatedCaller::new(verified, resolved)),
        authorizer,
        clock,
    ))
}

pub struct MasterKeyAuthenticator {
    key: Option<SecretValue>,
    secrets: Arc<dyn SecretSource>,
}

impl MasterKeyAuthenticator {
    pub fn new(key: Option<SecretValue>, secrets: Arc<dyn SecretSource>) -> Self {
        Self { key, secrets }
    }

    async fn key(&self) -> Result<SecretValue, Error> {
        let configured = self.key.as_ref().ok_or(Error::Unconfigured)?;
        let resolved = match configured.expose().strip_prefix("os.environ/") {
            Some(name) if !name.is_empty() => self
                .secrets
                .get_secret_str(name)
                .await?
                .ok_or(Error::Unconfigured)?,
            Some(_) => return Err(Error::Unconfigured),
            None => configured.clone(),
        };
        if resolved.expose().trim().is_empty() {
            return Err(Error::Unconfigured);
        }
        Ok(resolved)
    }
}

impl Authenticator for MasterKeyAuthenticator {
    fn validate(&self) -> AuthFuture<'_, ()> {
        Box::pin(async { self.key().await.map(|_| ()) })
    }

    fn verify<'a>(
        &'a self,
        credential: &'a Credential,
        _: &'a Parts,
    ) -> AuthFuture<'a, VerifiedIdentity> {
        Box::pin(async move {
            let expected = self.key().await?;
            let Credential::Token(credential) = credential else {
                return Err(Error::InvalidToken);
            };
            if !bool::from(
                Sha256::digest(credential.expose()).ct_eq(&Sha256::digest(expected.expose())),
            ) {
                return Err(Error::InvalidToken);
            }
            Ok(VerifiedIdentity {
                principal: master_principal(),
                authentication: Authentication {
                    method: AuthenticationMethod::MasterKey,
                    verifier: "litellm:master-key".into(),
                    credential_id: "master".into(),
                    expires_at: None,
                },
                restrictions: Permissions::All,
            })
        })
    }
}

fn master_principal() -> Principal {
    Principal::new(
        "litellm:master-key".into(),
        "master".into(),
        PrincipalKind::System,
    )
}
