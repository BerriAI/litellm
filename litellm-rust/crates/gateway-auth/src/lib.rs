mod authentication;
mod authorization;
mod error;
mod http;
mod identity;
pub mod keys;
mod ui;
mod virtual_keys;

use sha2::{Digest, Sha256};

pub use authentication::{
    Auth, AuthFuture, Authenticator, Clock, Credential, IdentityResolver, LocalAdministrator,
    MasterKeyAuthenticator, SystemClock,
};
pub use authorization::{
    AccessRequest, AuthenticatedRequest, AuthorizedOperation, Authorizer, KeyGrant, McpAction,
    NoAdditionalPolicy, Permissions, UiAction,
};
pub use error::{Error, KeyError, UiAuthError};
pub use http::{ApiKeyHeaders, Bearer, CredentialExtractor, RequireMasterKey, authenticate};
pub use identity::{
    AuthenticatedCaller, Authentication, AuthenticationMethod, Principal, PrincipalKind,
    ResolvedIdentity, SharedCaller, VerifiedIdentity,
};
pub use ui::{UI_CSRF_KEY, UiAuthSession, UiBackend, UiCredentials, UiSession, UiUser};
pub use virtual_keys::{CachedKeyLookup, KeyCacheSettings, VirtualKeyAuthenticator};

pub fn hash_token(token: &str) -> String {
    format!("{:x}", Sha256::digest(token.as_bytes()))
}
