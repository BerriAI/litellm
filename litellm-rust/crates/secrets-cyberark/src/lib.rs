#![forbid(unsafe_code)]

mod error;
mod secret_manager;

pub use error::Error;
pub use secret_manager::{AuthenticationRetry, CyberArkSecretManager, DeleteOutcome, WriteFailure};
