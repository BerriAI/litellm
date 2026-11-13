mod authentication;
mod authorization;
mod error;
mod http;
mod identity;
pub mod keys;
mod ui;

pub use authentication::{
    Auth, AuthFuture, Authenticator, Clock, Credential, IdentityResolver, LocalAdministrator,
    MasterKeyAuthenticator, SystemClock,
};
pub use authorization::{
    AccessRequest, AuthenticatedRequest, AuthorizedOperation, Authorizer, McpAction,
    NoAdditionalPolicy, Permissions, UiAction,
};
pub use error::Error;
pub use http::{Bearer, CredentialExtractor, RequireMasterKey, authenticate};
pub use identity::{
    AuthenticatedCaller, Authentication, AuthenticationMethod, Principal, PrincipalKind,
    ResolvedIdentity, SharedCaller, VerifiedIdentity,
};
pub use ui::{UI_CSRF_KEY, UiAuthSession, UiBackend, UiCredentials, UiSession, UiUser};
