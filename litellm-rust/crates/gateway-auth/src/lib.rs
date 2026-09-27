mod authentication;
mod authorization;
mod error;
mod http;
mod identity;
pub mod keys;

use sha2::{Digest, Sha256};

pub use authentication::{
    Auth, AuthFuture, Authenticator, Clock, Credential, IdentityResolver, LocalAdministrator,
    MasterKeyAuthenticator, SystemClock,
};
pub use authorization::{
    AccessRequest, AuthenticatedRequest, AuthorizedOperation, Authorizer, McpAction,
    NoAdditionalPolicy, Permissions, UiAction,
};
pub use error::{Error, KeyError};
pub use http::{Bearer, CredentialExtractor, RequireMasterKey, authenticate};
pub use identity::{
    AuthenticatedCaller, Authentication, AuthenticationMethod, Principal, PrincipalKind,
    ResolvedIdentity, SharedCaller, VerifiedIdentity,
};

pub fn hash_token(token: &str) -> String {
    format!("{:x}", Sha256::digest(token.as_bytes()))
}
