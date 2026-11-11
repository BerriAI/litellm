#![forbid(unsafe_code)]

mod auth;
mod error;
pub mod kms;
pub mod secret_manager;

pub use error::Error;
pub use kms::{AwsKms, load_aws_kms};
pub use secret_manager::{AwsSecretWriteSettings, AwsSecretsManagerV2, RotationResponse};
