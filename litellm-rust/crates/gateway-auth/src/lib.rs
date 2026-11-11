mod api_key;
mod authentication;
mod authorization;
mod core_auth;
mod error;
mod http;
mod identity;
pub mod keys;
mod ui;

use sha2::{Digest, Sha256};

pub use api_key::ApiKey;
pub use authentication::{
    Auth, AuthFuture, Authenticator, Clock, Credential, IdentityResolver, LocalAdministrator,
    MasterKeyAuthenticator, SystemClock, master_key,
};
pub use authorization::{
    AccessRequest, AuthenticatedRequest, AuthorizedOperation, Authorizer, McpAction,
    NoAdditionalPolicy, Permissions, UiAction,
};
pub use core_auth::CoreAuthenticator;
pub use error::Error;
pub use http::{ApiKeyHeaders, CredentialExtractor, authenticate};
pub use identity::{
    AuthenticatedCaller, Authentication, AuthenticationMethod, Principal, PrincipalKind,
    ResolvedIdentity, SharedCaller, VerifiedIdentity,
};
pub use ui::{UI_CSRF_KEY, UiAuthSession, UiBackend, UiCredentials, UiSession, UiUser};

pub fn hash_token(token: &str) -> String {
    format!("{:x}", Sha256::digest(token.as_bytes()))
}
