pub mod azure;
pub(crate) mod http;
mod secret;
mod token;

pub use http::CredentialPlacement;
pub use secret::SecretValue;
pub use token::{ResolvedCredential, TokenCaller, TokenCallerHandle, TokenFuture};
