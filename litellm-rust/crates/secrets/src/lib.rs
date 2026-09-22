#![forbid(unsafe_code)]

mod error;
mod handler;
mod oidc;
mod resolver;
mod state;

pub use error::Error;
pub use handler::{ExternalSecretManager, SecretManager, get_secret_from_manager};
pub use litellm_secrets_types::{
    AccessMode, KeyManagementSettings, KeyManagementSystem, Secret, SecretValue,
};
pub use oidc::{OidcProvider, OidcReference, OidcResolver};
pub use resolver::{FailurePolicy, SecretResolver};
pub use state::{SecretManagerState, secret_manager_would_be_consulted};

#[cfg(feature = "aws")]
pub use litellm_secrets_aws as aws;
#[cfg(feature = "azure")]
pub use litellm_secrets_azure as azure;
#[cfg(feature = "cyberark")]
pub use litellm_secrets_cyberark as cyberark;
#[cfg(feature = "google")]
pub use litellm_secrets_google as google;
#[cfg(feature = "hashicorp")]
pub use litellm_secrets_hashicorp as hashicorp;
