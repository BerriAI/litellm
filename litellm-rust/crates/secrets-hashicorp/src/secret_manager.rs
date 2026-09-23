mod client;
mod raw;
mod read;
mod write;

use std::{
    collections::HashMap,
    fmt,
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::{
    BaseSecretManager, HashicorpOperationContext, RotationError, SecretCache, SecretDeleter,
    SecretRotator, SecretValue, SecretWriteContext, SecretWriter, async_rotate_secret,
    validate_secret_name,
};
use rustify::errors::ClientError as RustifyClientError;
use serde_json::Value;
use tokio::sync::Mutex;
use vaultrs::{
    api,
    api::kv2::requests::SetSecretRequestOptions,
    auth::approle,
    client::{Identity, VaultClient, VaultClientSettingsBuilder},
    error::ClientError,
    kv2,
};

use crate::{Error, HashicorpVaultConfig, TlsCertAuth, cert_login::CertLoginRequest};

pub use raw::RawOperationError;

const CACHE_CAPACITY: u64 = 200;

#[derive(Clone)]
struct CachedClient {
    client: Arc<VaultClient>,
    expires_at: Option<Instant>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SecretLocation {
    pub namespace: Option<String>,
    pub mount: String,
    pub path: String,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    location: SecretLocation,
    data_key: String,
}

#[derive(Clone)]
pub struct HashicorpVault {
    config: HashicorpVaultConfig,
    cache: SecretCache<CacheKey, SecretValue>,
    auth_client: Arc<Mutex<Option<CachedClient>>>,
}

impl fmt::Debug for HashicorpVault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HashicorpVault")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl HashicorpVault {
    pub fn new(
        environment: Arc<dyn Lookup + Send + Sync>,
        enterprise_enabled: bool,
    ) -> Result<Self, Error> {
        let config: HashicorpVaultConfig =
            HashicorpVaultConfig::from_environment(environment.as_ref())?;
        Self::from_config(config, enterprise_enabled)
    }

    pub fn from_config(
        config: HashicorpVaultConfig,
        enterprise_enabled: bool,
    ) -> Result<Self, Error> {
        if !enterprise_enabled {
            return Err(Error::EnterpriseRequired);
        }
        let cache = SecretCache::new(CACHE_CAPACITY, config.refresh_interval);
        Ok(Self {
            config,
            cache,
            auth_client: Arc::new(Mutex::new(None)),
        })
    }

    pub fn secret_location(&self, secret_name: &str) -> Result<SecretLocation, Error> {
        self.secret_location_with_context(secret_name, &HashicorpOperationContext::default())
    }

    pub fn secret_location_with_context(
        &self,
        secret_name: &str,
        context: &HashicorpOperationContext,
    ) -> Result<SecretLocation, Error> {
        validate_secret_name(secret_name).map_err(Error::InvalidSecretName)?;
        let path: String = [
            context
                .path_prefix
                .as_deref()
                .or(self.config.path_prefix.as_deref())
                .and_then(path_component),
            Some(secret_name.to_owned()),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<String>>()
        .join("/");
        Ok(SecretLocation {
            namespace: context
                .namespace
                .as_deref()
                .or(self.config.secret_namespace())
                .and_then(path_component),
            mount: context
                .mount
                .as_deref()
                .or(Some(self.config.mount.as_str()))
                .and_then(path_component)
                .unwrap_or_else(|| "secret".to_owned()),
            path,
        })
    }

    pub fn config(&self) -> &HashicorpVaultConfig {
        &self.config
    }
}

#[derive(Clone, Copy)]
enum ErrorContext {
    Login,
    Read,
    Secret,
}

fn path_component(value: &str) -> Option<String> {
    let value: &str = value.trim().trim_matches('/');
    (!value.is_empty()).then(|| value.to_owned())
}

fn data_key(context: &HashicorpOperationContext) -> String {
    context
        .data_key
        .as_deref()
        .map(str::trim)
        .filter(|data_key| !data_key.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| "key".to_owned())
}

async fn with_timeout<T>(
    context: &HashicorpOperationContext,
    operation: impl Future<Output = Result<T, Error>>,
) -> Result<T, Error> {
    match context.timeout {
        Some(timeout) => tokio::time::timeout(timeout, operation)
            .await
            .map_err(|_| Error::Timeout)?,
        None => operation.await,
    }
}

fn map_api_error(error: ClientError, context: ErrorContext) -> Error {
    match error {
        ClientError::APIError { code, .. } => match context {
            ErrorContext::Login => Error::LoginStatus { status: code },
            ErrorContext::Read | ErrorContext::Secret => Error::Status { status: code },
        },
        ClientError::JsonParseError { source } => match context {
            ErrorContext::Login => Error::MalformedLogin,
            ErrorContext::Read => Error::MalformedPayload,
            ErrorContext::Secret => Error::Client(ClientError::JsonParseError { source }),
        },
        ClientError::ResponseEmptyError | ClientError::ResponseDataEmptyError => {
            malformed_response(context)
        }
        ClientError::RestClientError { source } => match source {
            RustifyClientError::ServerResponseError { code, .. } => match context {
                ErrorContext::Login => Error::LoginStatus { status: code },
                ErrorContext::Read | ErrorContext::Secret => Error::Status { status: code },
            },
            RustifyClientError::ResponseParseError { .. } => malformed_response(context),
            source => Error::Client(ClientError::RestClientError { source }),
        },
        error => Error::Client(error),
    }
}

fn api_status(error: &ClientError) -> Option<u16> {
    match error {
        ClientError::APIError { code, .. } => Some(*code),
        ClientError::RestClientError {
            source: RustifyClientError::ServerResponseError { code, .. },
        } => Some(*code),
        _ => None,
    }
}

fn malformed_response(context: ErrorContext) -> Error {
    match context {
        ErrorContext::Login => Error::MalformedLogin,
        ErrorContext::Read => Error::MalformedPayload,
        ErrorContext::Secret => Error::Client(ClientError::ResponseDataEmptyError),
    }
}
