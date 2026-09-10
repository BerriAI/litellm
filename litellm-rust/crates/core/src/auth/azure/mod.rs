#![allow(
    dead_code,
    unused_imports,
    reason = "used by the OCR architecture in the next stacked PR"
)]

mod credential_provider_cache;
mod native;
mod resolve;
mod types;

pub(crate) use resolve::AzureAuthService;
pub use types::{AzureAuthInputs, AzureCredentialType, ConfigValue, DEFAULT_AZURE_SCOPE};
