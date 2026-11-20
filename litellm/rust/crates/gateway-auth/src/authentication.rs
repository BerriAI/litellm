use std::{future::Future, pin::Pin, sync::Arc, time::SystemTime};

use axum::http::{Request, request::Parts};
use jiff::Timestamp;
use litellm_config::Config;
use litellm_error::Source;
use litellm_gateway_auth_core::{
    self as core, AuthHost, Identity, KeyDenial, KeyRow, Outcome, Settings, Token,
    authenticate_key, identity::key::KeyIdentity,
};
use litellm_secret_value::SecretValue;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::{
    AuthenticatedCaller, AuthenticatedRequest, Authentication, AuthenticationMethod, Authorizer,
    Bearer, CredentialExtractor, Error, NoAdditionalPolicy, Permissions, Principal, PrincipalKind,
    ResolvedIdentity, VerifiedIdentity,
    keys::{HeldKey, KeyHash, KeyLimits},
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

/// Grants the master key and a verified virtual key everything, within the model list a key's
/// credential carries. A key with other limits of its own is never verified, since its host
/// declines one.
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

    /// [`Self::from_config`], with every credential verified by `keys`, a virtual key's included.
    pub fn with_keys(config: &Config, keys: Arc<dyn Authenticator>) -> Self {
        Self {
            authenticator: keys,
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

/// Verifies the master key and the virtual keys `H` holds through `gateway-auth-core`'s key path.
/// Every route this crate guards is an inference route, which is what the request it hands the
/// core says.
pub struct KeyAuthenticator<H> {
    host: H,
    settings: Settings,
    clock: Arc<dyn Clock>,
}

impl<H> KeyAuthenticator<H> {
    pub fn new(host: H, settings: Settings) -> Self {
        Self {
            host,
            settings,
            clock: Arc::new(SystemClock),
        }
    }

    pub fn with_clock(self, clock: Arc<dyn Clock>) -> Self {
        Self { clock, ..self }
    }
}

impl<H> Authenticator for KeyAuthenticator<H>
where
    H: AuthHost,
    KeyRow<H>: HeldKey,
{
    fn validate(&self) -> AuthFuture<'_, ()> {
        Box::pin(async {
            match self.settings.key.master_key {
                Some(_) => Ok(()),
                None => Err(Error::Unconfigured),
            }
        })
    }

    fn verify<'a>(
        &'a self,
        credential: &'a Credential,
        request: &'a Parts,
    ) -> AuthFuture<'a, VerifiedIdentity> {
        Box::pin(async move {
            let Credential::Token(presented) = credential else {
                return Err(Error::InvalidToken);
            };
            let token = Token::new(presented.clone());
            let now = Timestamp::try_from(self.clock.now())
                .map_err(|error| Error::Unavailable(Source::new(error)))?;
            let request = core::Request {
                route: request.uri.path(),
                method: Some(&request.method),
                token: Some(&token),
                is_llm_api_route: true,
                runs_inference_checks: true,
                now,
                ..core::Request::default()
            };
            match authenticate_key(&self.host, &request, Some(&token), &self.settings).await {
                Ok(Outcome::Authenticated(Identity::Key(identity))) => {
                    identified::<H>(identity, presented)
                }
                Ok(Outcome::Authenticated(_)) => Err(Error::NotNative),
                Ok(Outcome::Denied(error, _)) | Err(error) => Err(rejected(error)),
            }
        })
    }
}

fn identified<H>(
    identity: KeyIdentity<H::Rows>,
    presented: &SecretValue,
) -> Result<VerifiedIdentity, Error>
where
    H: AuthHost,
    KeyRow<H>: HeldKey,
{
    match identity {
        KeyIdentity::MasterKey { .. } => Ok(master_identity()),
        KeyIdentity::VirtualKey(context) => Ok(VerifiedIdentity::virtual_key(
            &KeyHash::from_token(presented),
            &context.key,
        )),
        KeyIdentity::NoMasterKey { .. } => Err(Error::Unconfigured),
        KeyIdentity::UserAuthRoute | KeyIdentity::CachedAdmin { .. } => Err(Error::NotNative),
    }
}

/// What this server answers a credential the core did not authenticate with. The core's own text
/// is the proxy's, which answers instead wherever one stands behind this server.
fn rejected(error: core::Error) -> Error {
    match error {
        core::Error::HostDeclined => Error::NotNative,
        core::Error::HostFailed => Error::Unavailable(Source::new(error)),
        core::Error::Key(KeyDenial::Expired { .. }) => Error::Expired,
        core::Error::Key(
            KeyDenial::MasterKeyOnlyRoute { .. }
            | KeyDenial::UserAuthDisabled
            | KeyDenial::UserDeactivated { .. },
        )
        | core::Error::PremiumRequired(_)
        | core::Error::BudgetExceeded(_) => Error::Forbidden,
        core::Error::Key(
            KeyDenial::MissingKey
            | KeyDenial::MalformedKey
            | KeyDenial::NoDb
            | KeyDenial::NoDbConfigured
            | KeyDenial::NotVirtualKey { .. }
            | KeyDenial::NotFound { .. }
            | KeyDenial::Blocked
            | KeyDenial::NoToken,
        )
        | core::Error::Jwt(_) => Error::InvalidToken,
    }
}

impl VerifiedIdentity {
    /// The virtual key stored under `hash`, with what its row `key` says of it.
    pub fn virtual_key(hash: &KeyHash, key: &impl HeldKey) -> Self {
        let models = key.models();
        Self {
            principal: Principal::new(
                VIRTUAL_KEY.into(),
                hash.as_str().into(),
                PrincipalKind::Service,
            ),
            authentication: Authentication {
                method: AuthenticationMethod::VirtualKey,
                verifier: VIRTUAL_KEY.into(),
                credential_id: hash.as_str().into(),
                expires_at: key.expires_at(),
                limits: key.limits(),
                priority: key.priority(),
            },
            restrictions: match models.is_empty() {
                true => Permissions::All,
                false => Permissions::Models(models),
            },
        }
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
            limits: KeyLimits::default(),
            priority: None,
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
