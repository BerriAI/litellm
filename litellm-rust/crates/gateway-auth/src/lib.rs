mod authentication;
mod authorization;
mod error;
mod http;
mod identity;
pub mod keys;
pub mod policy;
pub mod store;
mod ui;

use sha2::{Digest, Sha256};

pub use authentication::{
    Auth, AuthFuture, Authenticator, Clock, Credential, IdentityResolver, LocalAdministrator,
    MasterKeyAuthenticator, SystemClock,
};
pub use authorization::{
    AccessRequest, AuthenticatedRequest, AuthorizedOperation, Authorizer, McpAction,
    NoAdditionalPolicy, Permissions, UiAction,
};
pub use error::{Error, KeyError, UiAuthError};
pub use http::{Bearer, CredentialExtractor, RequireMasterKey, authenticate};
pub use identity::{
    AuthenticatedCaller, Authentication, AuthenticationMethod, Principal, PrincipalKind,
    ResolvedIdentity, SharedCaller, VerifiedIdentity,
};
pub use keys::KeyHash;
pub use policy::{KeyPolicy, ModelScope, RouteScope, TeamPolicy};
pub use store::{
    Cached, InvalidInvalidation, Invalidation, KeyCache, KeyDigest, KeyRecord, KeyStore,
    MokaKeyCache, StoreError, StoreFuture, decode_invalidation,
};
pub use ui::{UI_CSRF_KEY, UiAuthSession, UiBackend, UiCredentials, UiSession, UiUser};

pub fn hash_token(token: &str) -> String {
    format!("{:x}", Sha256::digest(token.as_bytes()))
}
