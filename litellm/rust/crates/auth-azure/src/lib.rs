mod credential_provider_cache;
mod native;
mod requests_transport;
mod resolve;
mod retry_after;
#[cfg(test)]
mod test_support;
mod token_retry;
mod types;

pub use requests_transport::{
    CertificateRejection, MissingCaBundle, RequestsReason, RequestsSendFailure,
};
pub use resolve::{AzureAuthService, AzureIdentity, SECRET_NAMES};
pub use token_retry::{EntraTokenFailure, TokenResponse};
pub use types::{
    AzureAuthInputs, AzureTokenRetry, ConfigValue, DEFAULT_AZURE_SCOPE, TokenTransport,
};
