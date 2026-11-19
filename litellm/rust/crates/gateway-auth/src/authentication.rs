use std::{future::Future, pin::Pin, sync::Arc, time::SystemTime};

use axum::http::{Request, request::Parts};
use litellm_config::Config;
use litellm_secret_value::SecretValue;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::{
    AuthenticatedCaller, AuthenticatedRequest, Authentication, AuthenticationMethod, Authorizer,
    Bearer, CredentialExtractor, Error, NoAdditionalPolicy, Permissions, Principal, PrincipalKind,
    ResolvedIdentity, VerifiedIdentity,
    keys::{KeyHash, KeyLookup, KeyStatus},
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
                AuthenticationMethod::VirtualKey | AuthenticationMethod::External(_) => false,
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

/// Grants the master key and a verified virtual key everything. A key with limits of its own is
/// never verified, since a [`KeyLookup`] does not report one.
pub struct KeyHolder;

impl IdentityResolver for KeyHolder {
    fn resolve<'a>(&'a self, identity: &'a VerifiedIdentity) -> AuthFuture<'a, ResolvedIdentity> {
        match identity.authentication.method {
            AuthenticationMethod::VirtualKey => Box::pin(async {
                Ok(ResolvedIdentity {
                    principal: identity.principal.clone(),
                    permissions: Permissions::All,
                })
            }),
            AuthenticationMethod::MasterKey | AuthenticationMethod::External(_) => {
                LocalAdministrator.resolve(identity)
            }
        }
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

    pub fn from_config(config: &Config) -> Self {
        let master = Arc::new(MasterKeyAuthenticator {
            key: config.general_settings.master_key.clone(),
        });
        Self {
            extractor: Arc::new(Bearer),
            authenticator: master,
            identities: Arc::new(LocalAdministrator),
            authorizer: Arc::new(NoAdditionalPolicy),
            clock: Arc::new(SystemClock),
        }
    }

    /// [`Self::from_config`], with every other `sk-` token verified as a virtual key of `keys`.
    pub fn with_keys(config: &Config, keys: Arc<dyn KeyLookup>) -> Self {
        Self {
            authenticator: Arc::new(KeyAuthenticator::new(
                config.general_settings.master_key.clone(),
                keys,
            )),
            identities: Arc::new(KeyHolder),
            ..Self::from_config(config)
        }
    }

    pub fn with_extractor(self, extractor: Arc<dyn CredentialExtractor>) -> Self {
        Self { extractor, ..self }
    }

    pub async fn authenticate_parts(&self, request: &Parts) -> Result<AuthenticatedRequest, Error> {
        let credential = self.extractor.extract(request)?;
        self.authenticate_request(&credential, request)
            .await
            .map_err(|error| self.extractor.reject(request, error))
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
}

impl MasterKeyAuthenticator {
    pub fn new(key: Option<SecretValue>) -> Self {
        Self { key }
    }

    fn key(&self) -> Result<&SecretValue, Error> {
        self.key
            .as_ref()
            .filter(|key| !key.expose().trim().is_empty())
            .ok_or(Error::Unconfigured)
    }

    /// Whether `credential` is the master key.
    fn holds(&self, credential: &Credential) -> Result<bool, Error> {
        let expected = self.key()?;
        Ok(match credential {
            Credential::Token(token) => {
                bool::from(Sha256::digest(token.expose()).ct_eq(&Sha256::digest(expected.expose())))
            }
            Credential::Transport => false,
        })
    }
}

impl Authenticator for MasterKeyAuthenticator {
    fn validate(&self) -> AuthFuture<'_, ()> {
        Box::pin(async { self.key().map(|_| ()) })
    }

    fn verify<'a>(
        &'a self,
        credential: &'a Credential,
        _: &'a Parts,
    ) -> AuthFuture<'a, VerifiedIdentity> {
        Box::pin(async move {
            match self.holds(credential)? {
                true => Ok(master_identity()),
                false => Err(Error::InvalidToken),
            }
        })
    }
}

const VIRTUAL_KEY: &str = "litellm:virtual-key";
const VIRTUAL_KEY_PREFIX: &str = "sk-";

/// Verifies the master key, and every other `sk-` token as the virtual key `keys` holds under its
/// hash.
pub struct KeyAuthenticator {
    master: MasterKeyAuthenticator,
    keys: Arc<dyn KeyLookup>,
}

impl KeyAuthenticator {
    pub fn new(master_key: Option<SecretValue>, keys: Arc<dyn KeyLookup>) -> Self {
        Self {
            master: MasterKeyAuthenticator::new(master_key),
            keys,
        }
    }
}

impl Authenticator for KeyAuthenticator {
    fn validate(&self) -> AuthFuture<'_, ()> {
        self.master.validate()
    }

    fn verify<'a>(
        &'a self,
        credential: &'a Credential,
        _: &'a Parts,
    ) -> AuthFuture<'a, VerifiedIdentity> {
        Box::pin(async move {
            if self.master.holds(credential)? {
                return Ok(master_identity());
            }
            let Credential::Token(token) = credential else {
                return Err(Error::InvalidToken);
            };
            if !token.expose().starts_with(VIRTUAL_KEY_PREFIX) {
                return Err(Error::InvalidToken);
            }
            let hash = KeyHash::from_token(token);
            match self.keys.lookup(&hash).await? {
                Some(KeyStatus::Active { expires_at }) => Ok(VerifiedIdentity {
                    principal: Principal::new(
                        VIRTUAL_KEY.into(),
                        hash.as_str().into(),
                        PrincipalKind::Service,
                    ),
                    authentication: Authentication {
                        method: AuthenticationMethod::VirtualKey,
                        verifier: VIRTUAL_KEY.into(),
                        credential_id: hash.as_str().into(),
                        expires_at,
                    },
                    restrictions: Permissions::All,
                }),
                Some(KeyStatus::Revoked) | None => Err(Error::InvalidToken),
            }
        })
    }
}

fn master_identity() -> VerifiedIdentity {
    VerifiedIdentity {
        principal: master_principal(),
        authentication: Authentication {
            method: AuthenticationMethod::MasterKey,
            verifier: "litellm:master-key".into(),
            credential_id: "master".into(),
            expires_at: None,
        },
        restrictions: Permissions::All,
    }
}

fn master_principal() -> Principal {
    Principal::new(
        "litellm:master-key".into(),
        "master".into(),
        PrincipalKind::System,
    )
}
