#![forbid(unsafe_code)]

mod error;
mod key_vault;

pub use error::Error;
pub use key_vault::{AzureKeyVault, AzureTokenProvider, NativeAzureTokenProvider};
