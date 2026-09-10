pub mod azure;
mod credential;
pub mod error;
pub use error::AuthError;
pub(crate) mod http;
mod policy;
mod secret;
mod token;

pub use credential::{
    CredentialFileRef, CredentialLookup, CredentialLookupFuture, CredentialPlan,
    CredentialPlanResolution, CredentialRef, CredentialResolver, CredentialResolverHandle,
};
pub use http::{CredentialPlacement, RequestAuth};
pub use policy::{CredentialPlanKind, CredentialRule, ExistingHeaderBehavior, ProviderAuthPolicy};
pub use secret::SecretValue;
pub use token::{ResolvedCredential, TokenFuture, TokenProvider, TokenProviderHandle};
