#![forbid(unsafe_code)]

mod base_secret_manager;
mod cache;
mod config;
mod context;
mod error;
mod value;

pub use base_secret_manager::{
    BaseSecretManager, RotationError, SecretDeleter, SecretRotator, SecretWriter,
    async_rotate_secret, validate_secret_name,
};
pub use cache::SecretCache;
pub use config::{AccessMode, KeyManagementSettings, KeyManagementSystem};
pub use context::{
    AwsOperationContext, AzureOperationContext, CyberarkOperationContext, GoogleOperationContext,
    HashicorpOperationContext, SecretOperationContext, SecretWriteContext,
};
pub use error::Error;
pub use litellm_auth_types::SecretValue;
pub use value::{PythonSecretRead, Secret};
