use std::{
    collections::HashMap,
    fmt,
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::{
    BaseSecretManager, HashicorpOperationContext, SecretOperationContext, SecretValue,
    SecretWriteContext, async_rotate_secret, validate_secret_name,
};
use moka::future::Cache;
use rustify::errors::ClientError as RustifyClientError;
use serde_json::Value;
use tokio::sync::Mutex;
use vaultrs::{
    api,
    auth::approle,
    client::{Identity, VaultClient, VaultClientSettingsBuilder},
    error::ClientError,
    kv2,
};

use crate::{Error, HashicorpVaultConfig, TlsCertAuth, cert_login::CertLoginRequest};

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
    cache: Cache<CacheKey, SecretValue>,
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
        let cache: Cache<CacheKey, SecretValue> = Cache::builder()
            .max_capacity(CACHE_CAPACITY)
            .time_to_live(config.refresh_interval)
            .build();
        Ok(Self {
            config,
            cache,
            auth_client: Arc::new(Mutex::new(None)),
        })
    }

    pub fn secret_location(&self, secret_name: &str) -> Result<SecretLocation, Error> {
        self.secret_location_with_context(secret_name, &SecretOperationContext::default())
    }

    pub fn secret_location_with_context(
        &self,
        secret_name: &str,
        context: &SecretOperationContext,
    ) -> Result<SecretLocation, Error> {
        validate_secret_name(secret_name).map_err(Error::InvalidSecretName)?;
        let operation: Option<&HashicorpOperationContext> = hashicorp_context(context)?;
        let path: String = [
            operation
                .and_then(|operation| operation.path_prefix.as_deref())
                .and_then(path_component)
                .or_else(|| self.config.path_prefix.clone()),
            Some(secret_name.to_owned()),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<String>>()
        .join("/");
        Ok(SecretLocation {
            namespace: self.config.secret_namespace().map(str::to_owned),
            mount: operation
                .and_then(|operation| operation.mount.as_deref())
                .and_then(path_component)
                .unwrap_or_else(|| self.config.mount.clone()),
            path,
        })
    }

    pub fn config(&self) -> &HashicorpVaultConfig {
        &self.config
    }

    pub async fn async_read_secret(&self, secret_name: &str) -> Result<Option<SecretValue>, Error> {
        self.async_read_secret_with_context(secret_name, &SecretOperationContext::default())
            .await
    }

    pub async fn async_read_secret_with_context(
        &self,
        secret_name: &str,
        context: &SecretOperationContext,
    ) -> Result<Option<SecretValue>, Error> {
        let location: SecretLocation = self.secret_location_with_context(secret_name, context)?;
        let data_key: String = data_key(context)?;
        let cache_key = CacheKey {
            location: location.clone(),
            data_key: data_key.clone(),
        };
        if let Some(value) = self.cache.get(&cache_key).await {
            return Ok(Some(value));
        }
        let data: Option<HashMap<String, Value>> = with_timeout(context, async {
            let client: Arc<VaultClient> = self.vault_client().await?;
            match kv2::read(client.as_ref(), &location.mount, &location.path).await {
                Ok(data) => Ok(Some(data)),
                Err(error) if api_status(&error) == Some(404) => Ok(None),
                Err(error) => Err(map_api_error(error, ErrorContext::Read)),
            }
        })
        .await?;
        let Some(data) = data else {
            return Ok(None);
        };
        let Some(value) = data.get(&data_key) else {
            return Ok(None);
        };
        let value: &str = value.as_str().ok_or(Error::NonStringValue)?;
        let value: SecretValue = SecretValue::new(value);
        self.cache.insert(cache_key, value.clone()).await;
        Ok(Some(value))
    }

    pub async fn async_write_secret(
        &self,
        secret_name: &str,
        value: SecretValue,
        description: Option<&str>,
    ) -> Result<Value, Error> {
        self.async_write_secret_with_context(
            secret_name,
            &value,
            &SecretWriteContext {
                description: description.map(str::to_owned),
                ..SecretWriteContext::default()
            },
        )
        .await
    }

    pub async fn async_write_secret_with_context(
        &self,
        secret_name: &str,
        value: &SecretValue,
        context: &SecretWriteContext,
    ) -> Result<Value, Error> {
        let location: SecretLocation =
            self.secret_location_with_context(secret_name, &context.operation)?;
        let data_key: String = data_key(&context.operation)?;
        let data: HashMap<String, Value> = match context.description.as_deref() {
            Some(description) => [
                (data_key, Value::String(value.expose().to_owned())),
                (
                    "description".to_owned(),
                    Value::String(description.to_owned()),
                ),
            ]
            .into_iter()
            .collect(),
            None => [(data_key, Value::String(value.expose().to_owned()))]
                .into_iter()
                .collect(),
        };
        let metadata = with_timeout(&context.operation, async {
            let client: Arc<VaultClient> = self.vault_client().await?;
            kv2::set(client.as_ref(), &location.mount, &location.path, &data)
                .await
                .map_err(|error| map_api_error(error, ErrorContext::Secret))
        })
        .await?;
        self.cache.invalidate_all();
        serde_json::to_value(metadata)
            .map_err(|source| Error::Client(ClientError::JsonParseError { source }))
    }

    pub async fn async_delete_secret(&self, secret_name: &str) -> Result<(), Error> {
        self.async_delete_secret_with_context(secret_name, &SecretOperationContext::default())
            .await
    }

    pub async fn async_delete_secret_with_context(
        &self,
        secret_name: &str,
        context: &SecretOperationContext,
    ) -> Result<(), Error> {
        let location: SecretLocation = self.secret_location_with_context(secret_name, context)?;
        with_timeout(context, async {
            let client: Arc<VaultClient> = self.vault_client().await?;
            kv2::delete_latest(client.as_ref(), &location.mount, &location.path)
                .await
                .map_err(|error| map_api_error(error, ErrorContext::Secret))
        })
        .await?;
        self.cache.invalidate_all();
        Ok(())
    }

    pub async fn async_rotate_secret(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
    ) -> Result<Value, Error> {
        self.async_rotate_secret_with_context(
            current_name,
            new_name,
            value,
            &SecretOperationContext::default(),
        )
        .await
    }

    pub async fn async_rotate_secret_with_context(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
        context: &SecretOperationContext,
    ) -> Result<Value, Error> {
        async_rotate_secret(self, current_name, new_name, value, context).await
    }

    async fn vault_client(&self) -> Result<Arc<VaultClient>, Error> {
        let mut cached = self.auth_client.lock().await;
        if let Some(entry) = cached.as_ref()
            && entry
                .expires_at
                .is_none_or(|expires_at| expires_at > Instant::now())
        {
            return Ok(entry.client.clone());
        }

        let (client, expires_at): (VaultClient, Option<Instant>) =
            match (self.config.approle.as_ref(), self.config.tls_cert.as_ref()) {
                (Some(approle), _) => {
                    let login_client: VaultClient =
                        self.build_client(self.config.login_namespace(), "")?;
                    let auth = approle::login(
                        &login_client,
                        &approle.mount_path,
                        &approle.role_id,
                        approle.secret_id.expose(),
                    )
                    .await
                    .map_err(|error| map_api_error(error, ErrorContext::Login))?;
                    (
                        self.build_client(self.config.secret_namespace(), &auth.client_token)?,
                        token_expiry(auth.lease_duration),
                    )
                }
                (None, Some(tls)) => {
                    let login_client: VaultClient =
                        self.build_client(self.config.login_namespace(), "")?;
                    let endpoint: CertLoginRequest = CertLoginRequest::new(tls.role.as_deref());
                    let auth = api::auth(&login_client, endpoint)
                        .await
                        .map_err(|error| map_api_error(error, ErrorContext::Login))?;
                    (
                        self.build_client(self.config.secret_namespace(), &auth.client_token)?,
                        token_expiry(auth.lease_duration),
                    )
                }
                (None, None) => {
                    let token: SecretValue =
                        self.config.token.clone().ok_or(Error::NoAuthConfigured)?;
                    (
                        self.build_client(self.config.secret_namespace(), token.expose())?,
                        None,
                    )
                }
            };
        let client: Arc<VaultClient> = Arc::new(client);
        *cached = Some(CachedClient {
            client: client.clone(),
            expires_at,
        });
        Ok(client)
    }

    fn build_client(&self, namespace: Option<&str>, token: &str) -> Result<VaultClient, Error> {
        let settings = VaultClientSettingsBuilder::default()
            .address(&self.config.address)
            .token(token.to_owned())
            .namespace(namespace.map(str::to_owned))
            .identity(identity_for(self.config.tls_cert.as_ref())?)
            .ca_certs(Vec::new())
            .verify(true)
            .build()
            .map_err(|message| Error::ClientSettings {
                message: message.to_string(),
            })?;
        VaultClient::new(settings).map_err(Error::Client)
    }
}

impl BaseSecretManager for HashicorpVault {
    type Error = Error;
    type WriteResponse = Value;
    type DeleteResponse = ();

    async fn async_read_secret(
        &self,
        name: &str,
        context: &SecretOperationContext,
    ) -> Result<Option<SecretValue>, Error> {
        HashicorpVault::async_read_secret_with_context(self, name, context).await
    }

    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext,
    ) -> Result<Value, Error> {
        HashicorpVault::async_write_secret_with_context(self, name, value, context).await
    }

    async fn async_delete_secret(
        &self,
        name: &str,
        _recovery_window_in_days: Option<u32>,
        context: &SecretOperationContext,
    ) -> Result<(), Error> {
        HashicorpVault::async_delete_secret_with_context(self, name, context).await
    }
}

#[derive(Clone, Copy)]
enum ErrorContext {
    Login,
    Read,
    Secret,
}

fn hashicorp_context(
    context: &SecretOperationContext,
) -> Result<Option<&HashicorpOperationContext>, Error> {
    match context {
        SecretOperationContext::Hashicorp(context) => Ok(Some(context)),
        SecretOperationContext::Default => Ok(None),
        SecretOperationContext::Aws(_) | SecretOperationContext::Cyberark(_) => {
            Err(Error::InvalidOperationContext)
        }
    }
}

fn path_component(value: &str) -> Option<String> {
    let value: &str = value.trim().trim_matches('/');
    (!value.is_empty()).then(|| value.to_owned())
}

fn data_key(context: &SecretOperationContext) -> Result<String, Error> {
    Ok(hashicorp_context(context)?
        .and_then(|context| context.data_key.as_deref())
        .map(str::trim)
        .filter(|data_key| !data_key.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| "key".to_owned()))
}

async fn with_timeout<T>(
    context: &SecretOperationContext,
    operation: impl Future<Output = Result<T, Error>>,
) -> Result<T, Error> {
    match context.timeout() {
        Some(timeout) => tokio::time::timeout(timeout, operation)
            .await
            .map_err(|_| Error::Timeout)?,
        None => operation.await,
    }
}

fn identity_for(tls: Option<&TlsCertAuth>) -> Result<Option<Identity>, Error> {
    tls.map(|tls| {
        let cert: Vec<u8> = std::fs::read(&tls.cert_path).map_err(|source| Error::TlsIdentity {
            path: tls.cert_path.clone(),
            message: source.to_string(),
        })?;
        let key: Vec<u8> = std::fs::read(&tls.key_path).map_err(|source| Error::TlsIdentity {
            path: tls.key_path.clone(),
            message: source.to_string(),
        })?;
        Identity::from_pem(&[cert.as_slice(), key.as_slice()].concat()).map_err(|source| {
            Error::TlsIdentity {
                path: tls.cert_path.clone(),
                message: source.to_string(),
            }
        })
    })
    .transpose()
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

fn token_expiry(lease_duration: u64) -> Option<Instant> {
    (lease_duration > 0).then(|| Instant::now() + Duration::from_secs(lease_duration))
}
