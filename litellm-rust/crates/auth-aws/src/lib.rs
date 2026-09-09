mod cache;
mod credentials;
mod error;
mod role;
mod runtime;
mod signing;

pub use cache::{Clock, CredentialScope, CredentialState, SystemClock};
pub use credentials::{Credentials, session_credentials, static_credentials};
pub use error::Error;
pub use role::{role_identity, same_role_arns};
pub use runtime::{
    AssumeRoleRequest, CredentialFuture, CredentialRuntime, NativeCredentialRuntime,
    WebIdentityRequest, assume_role_credentials, caller_identity, default_credentials,
    profile_credentials, web_identity_credentials,
};
pub use signing::{SigV4Request, sign_v4};
