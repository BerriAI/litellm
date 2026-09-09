mod authorization;
mod credentials;
mod error;
mod secret;
mod services;

pub use authorization::BodyAuthorizationInput;
pub use credentials::{AuthServices, AzureCredentialInputs, CallerCredential, CredentialInputs};
pub use error::AuthServiceError;
pub use secret::{ResolvedCredential, SecretString, SuppliedSecret};
pub use services::{AuthValueLookup, CallerTokenProvider, ExecutionHeaders};

#[cfg(test)]
mod tests;
