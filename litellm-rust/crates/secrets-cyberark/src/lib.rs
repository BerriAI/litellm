#![forbid(unsafe_code)]

mod error;
mod secret_manager;

pub use error::{Error, WriteFailure};
pub use secret_manager::{AuthenticationRetry, CyberArkSecretManager, DeleteOutcome};
