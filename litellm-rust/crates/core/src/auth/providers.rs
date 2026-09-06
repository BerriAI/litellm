use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};

use super::{
    AuthorizationInput, AuthorizationMutation, AuthorizationPreparation, AuthorizationProvider,
    Clock, CredentialProvenance, OperationFuture, RequestAuthorizer, SecretString, TokenCredential,
    TokenProvider,
};
use crate::Error;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum CredentialKind {
    StaticHeader,
    Bearer,
}

pub enum CredentialSpec {
    StaticHeader {
        name: HeaderName,
        value: SecretString,
        conflicts: Vec<HeaderName>,
    },
    Bearer {
        provider: Arc<dyn TokenProvider>,
        conflicts: Vec<HeaderName>,
    },
}

impl CredentialSpec {
    pub fn kind(&self) -> CredentialKind {
        match self {
            Self::StaticHeader { .. } => CredentialKind::StaticHeader,
            Self::Bearer { .. } => CredentialKind::Bearer,
        }
    }
}

type CredentialFactory =
    Box<dyn FnOnce() -> OperationFuture<'static, Option<CredentialSpec>> + Send + 'static>;

pub struct CredentialCandidate {
    provenance: CredentialProvenance,
    kind: CredentialKind,
    factory: CredentialFactory,
}

impl CredentialCandidate {
    pub fn new(
        provenance: CredentialProvenance,
        kind: CredentialKind,
        factory: impl FnOnce() -> OperationFuture<'static, CredentialSpec> + Send + 'static,
    ) -> Self {
        Self {
            provenance,
            kind,
            factory: Box::new(move || {
                let future = factory();
                Box::pin(async move { future.await.map(Some) })
            }),
        }
    }

    pub fn discover(
        provenance: CredentialProvenance,
        kind: CredentialKind,
        factory: impl FnOnce() -> OperationFuture<'static, Option<CredentialSpec>> + Send + 'static,
    ) -> Self {
        Self {
            provenance,
            kind,
            factory: Box::new(factory),
        }
    }
}

pub trait CredentialAdapter: Send + Sync {
    fn kind(&self) -> CredentialKind;

    fn build(
        &self,
        spec: CredentialSpec,
        provenance: CredentialProvenance,
        clock: Arc<dyn Clock>,
    ) -> Result<Arc<dyn AuthorizationProvider>, Error>;
}

pub struct CredentialAdapterRegistry {
    adapters: HashMap<CredentialKind, Arc<dyn CredentialAdapter>>,
    clock: Arc<dyn Clock>,
}

impl CredentialAdapterRegistry {
    pub fn with_builtin_adapters(clock: Arc<dyn Clock>) -> Self {
        let adapters: [Arc<dyn CredentialAdapter>; 2] = [
            Arc::new(StaticHeaderCredentialAdapter),
            Arc::new(BearerCredentialAdapter),
        ];
        Self {
            adapters: adapters
                .into_iter()
                .map(|adapter| (adapter.kind(), adapter))
                .collect(),
            clock,
        }
    }

    pub fn new(
        adapters: impl IntoIterator<Item = Arc<dyn CredentialAdapter>>,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, Error> {
        let adapters: Vec<_> = adapters.into_iter().collect();
        let unique: HashSet<_> = adapters.iter().map(|adapter| adapter.kind()).collect();
        if unique.len() != adapters.len() {
            return Err(Error::Auth("duplicate credential adapter".into()));
        }
        Ok(Self {
            adapters: adapters
                .into_iter()
                .map(|adapter| (adapter.kind(), adapter))
                .collect(),
            clock,
        })
    }

    pub fn resolve(
        &self,
        candidates: impl IntoIterator<Item = CredentialCandidate>,
    ) -> OperationFuture<'_, Arc<dyn AuthorizationProvider>> {
        let candidates: Vec<_> = candidates.into_iter().collect();
        Box::pin(async move {
            for candidate in candidates {
                let expected_kind = candidate.kind;
                let provenance = candidate.provenance;
                let Some(spec) = (candidate.factory)().await? else {
                    continue;
                };
                if spec.kind() != expected_kind {
                    return Err(Error::Auth(
                        "credential factory returned the wrong credential kind".into(),
                    ));
                }
                let adapter = self
                    .adapters
                    .get(&expected_kind)
                    .ok_or_else(|| Error::Auth("unsupported credential kind".into()))?;
                return adapter.build(spec, provenance, self.clock.clone());
            }
            Err(Error::Auth("missing credential".into()))
        })
    }
}

#[derive(Clone)]
struct HeaderSnapshot {
    name: HeaderName,
    value: HeaderValue,
    declared: Vec<HeaderName>,
    conflicts: Vec<HeaderName>,
}

impl HeaderSnapshot {
    fn new(
        name: HeaderName,
        value: SecretString,
        conflicts: Vec<HeaderName>,
    ) -> Result<Self, Error> {
        let mut header = HeaderValue::from_str(value.expose())
            .map_err(|_| Error::Auth("credential cannot be used in an HTTP header".into()))?;
        header.set_sensitive(true);
        let declared = std::iter::once(name.clone())
            .chain(conflicts.iter().cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        Ok(Self {
            name,
            value: header,
            declared,
            conflicts,
        })
    }

    fn preparation(self: Arc<Self>, provenance: CredentialProvenance) -> AuthorizationPreparation {
        AuthorizationPreparation {
            authorizer: self.clone(),
            visible_headers: HeaderMap::from_iter([(self.name.clone(), self.value.clone())]),
            remove_headers: self.conflicts.clone(),
            provenance: Some(provenance),
        }
    }
}

impl RequestAuthorizer for HeaderSnapshot {
    fn declared_headers(&self) -> &[HeaderName] {
        &self.declared
    }

    fn authorize<'a>(
        &'a self,
        _: AuthorizationInput<'a>,
    ) -> OperationFuture<'a, AuthorizationMutation> {
        Box::pin(async move {
            Ok(AuthorizationMutation {
                set_headers: vec![(self.name.clone(), self.value.clone())],
                remove_headers: self.conflicts.clone(),
            })
        })
    }
}

pub struct StaticHeaderAuthorizationProvider {
    name: HeaderName,
    value: SecretString,
    conflicts: Vec<HeaderName>,
    provenance: CredentialProvenance,
}

impl StaticHeaderAuthorizationProvider {
    pub fn new(
        name: HeaderName,
        value: SecretString,
        conflicts: Vec<HeaderName>,
        provenance: CredentialProvenance,
    ) -> Self {
        Self {
            name,
            value,
            conflicts,
            provenance,
        }
    }
}

impl AuthorizationProvider for StaticHeaderAuthorizationProvider {
    fn prepare(&self) -> OperationFuture<'_, AuthorizationPreparation> {
        Box::pin(async move {
            HeaderSnapshot::new(
                self.name.clone(),
                self.value.clone(),
                self.conflicts.clone(),
            )
            .map(Arc::new)
            .map(|snapshot| snapshot.preparation(self.provenance.clone()))
        })
    }
}

pub struct BearerAuthorizationProvider {
    provider: Arc<dyn TokenProvider>,
    clock: Arc<dyn Clock>,
    conflicts: Vec<HeaderName>,
    provenance: CredentialProvenance,
}

impl BearerAuthorizationProvider {
    pub fn new(
        provider: Arc<dyn TokenProvider>,
        clock: Arc<dyn Clock>,
        conflicts: Vec<HeaderName>,
        provenance: CredentialProvenance,
    ) -> Self {
        Self {
            provider,
            clock,
            conflicts,
            provenance,
        }
    }
}

impl AuthorizationProvider for BearerAuthorizationProvider {
    fn prepare(&self) -> OperationFuture<'_, AuthorizationPreparation> {
        Box::pin(async move {
            let token = match self.provider.token().await? {
                TokenCredential::KnownExpiry(lease) if lease.expires_at > self.clock.now() => {
                    lease.token
                }
                TokenCredential::KnownExpiry(_) => {
                    return Err(Error::Auth(
                        "credential provider returned an expired token".into(),
                    ));
                }
                TokenCredential::NoStore(token) => token,
            };
            if token.expose().trim().is_empty() {
                return Err(Error::Auth(
                    "credential provider returned an empty token".into(),
                ));
            }
            let value = SecretString::new(format!("Bearer {}", token.expose()));
            HeaderSnapshot::new(AUTHORIZATION, value, self.conflicts.clone())
                .map(Arc::new)
                .map(|snapshot| snapshot.preparation(self.provenance.clone()))
        })
    }
}

pub struct StaticHeaderCredentialAdapter;

impl CredentialAdapter for StaticHeaderCredentialAdapter {
    fn kind(&self) -> CredentialKind {
        CredentialKind::StaticHeader
    }

    fn build(
        &self,
        spec: CredentialSpec,
        provenance: CredentialProvenance,
        _: Arc<dyn Clock>,
    ) -> Result<Arc<dyn AuthorizationProvider>, Error> {
        match spec {
            CredentialSpec::StaticHeader {
                name,
                value,
                conflicts,
            } => Ok(Arc::new(StaticHeaderAuthorizationProvider::new(
                name, value, conflicts, provenance,
            ))),
            CredentialSpec::Bearer { .. } => Err(Error::Auth(
                "static header adapter received a bearer credential".into(),
            )),
        }
    }
}

pub struct BearerCredentialAdapter;

impl CredentialAdapter for BearerCredentialAdapter {
    fn kind(&self) -> CredentialKind {
        CredentialKind::Bearer
    }

    fn build(
        &self,
        spec: CredentialSpec,
        provenance: CredentialProvenance,
        clock: Arc<dyn Clock>,
    ) -> Result<Arc<dyn AuthorizationProvider>, Error> {
        match spec {
            CredentialSpec::Bearer {
                provider,
                conflicts,
            } => Ok(Arc::new(BearerAuthorizationProvider::new(
                provider, clock, conflicts, provenance,
            ))),
            CredentialSpec::StaticHeader { .. } => Err(Error::Auth(
                "bearer adapter received a static header credential".into(),
            )),
        }
    }
}
