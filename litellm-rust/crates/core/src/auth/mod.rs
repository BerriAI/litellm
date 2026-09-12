mod credential;
pub mod error;
pub(crate) mod vertex;
pub use error::AuthError;
pub(crate) mod http;
mod policy;
mod secret;
mod token;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InputSource {
    Request,
    #[default]
    Deployment,
    Environment,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Sourced<T> {
    value: T,
    source: InputSource,
}

impl<T> Sourced<T> {
    pub fn new(value: T, source: InputSource) -> Self {
        Self { value, source }
    }

    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn source(&self) -> InputSource {
        self.source
    }

    pub fn into_value(self) -> T {
        self.value
    }

    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> Sourced<U> {
        Sourced::new(map(self.value), self.source)
    }
}

pub use credential::{
    CredentialFileRef, CredentialLookup, CredentialLookupFuture, CredentialPlan,
    CredentialPlanResolution, CredentialRef, CredentialResolver, CredentialResolverHandle,
};
pub use http::{CredentialPlacement, RequestAuth};
pub use policy::{CredentialPlanKind, CredentialRule, ExistingHeaderBehavior, ProviderAuthPolicy};
pub use secret::SecretValue;
pub use token::{ResolvedCredential, TokenFuture, TokenProvider, TokenProviderHandle};
