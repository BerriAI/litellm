use std::{
    collections::HashMap,
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::{
    BaseSecretManager, SecretValue, async_rotate_secret, validate_secret_name,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecretLocation {
    pub namespace: Option<String>,
    pub mount: String,
    pub path: String,
}

#[derive(Clone)]
pub struct HashicorpVault {
    config: HashicorpVaultConfig,
    cache: Cache<String, SecretValue>,
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
        let cache: Cache<String, SecretValue> = Cache::builder()
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
        validate_secret_name(secret_name).map_err(Error::InvalidSecretName)?;
        let path: String = [
            self.config.path_prefix.clone(),
            Some(secret_name.to_owned()),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<String>>()
        .join("/");
        Ok(SecretLocation {
            namespace: self.config.secret_namespace().map(str::to_owned),
            mount: self.config.mount.clone(),
            path,
        })
    }

    pub fn config(&self) -> &HashicorpVaultConfig {
        &self.config
    }

    pub async fn async_read_secret(&self, secret_name: &str) -> Result<Option<SecretValue>, Error> {
        let location: SecretLocation = self.secret_location(secret_name)?;
        let cache_key: String = cache_key(&location);
        if let Some(value) = self.cache.get(&cache_key).await {
            return Ok(Some(value));
        }
        let client: Arc<VaultClient> = self.vault_client().await?;
        let data: HashMap<String, Value> =
            match kv2::read(client.as_ref(), &location.mount, &location.path).await {
                Ok(data) => data,
                Err(error) if api_status(&error) == Some(404) => return Ok(None),
                Err(error) => return Err(map_api_error(error, ErrorContext::Read)),
            };
        let Some(value) = data.get("key") else {
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
        let location: SecretLocation = self.secret_location(secret_name)?;
        let cache_key: String = cache_key(&location);
        let data: HashMap<String, Value> = match description {
            Some(description) => [
                ("key".to_owned(), Value::String(value.expose().to_owned())),
                (
                    "description".to_owned(),
                    Value::String(description.to_owned()),
                ),
            ]
            .into_iter()
            .collect(),
            None => [("key".to_owned(), Value::String(value.expose().to_owned()))]
                .into_iter()
                .collect(),
        };
        let client: Arc<VaultClient> = self.vault_client().await?;
        let metadata = kv2::set(client.as_ref(), &location.mount, &location.path, &data)
            .await
            .map_err(|error| map_api_error(error, ErrorContext::Secret))?;
        self.cache.invalidate(&cache_key).await;
        serde_json::to_value(metadata)
            .map_err(|source| Error::Client(ClientError::JsonParseError { source }))
    }

    pub async fn async_delete_secret(&self, secret_name: &str) -> Result<(), Error> {
        let location: SecretLocation = self.secret_location(secret_name)?;
        let cache_key: String = cache_key(&location);
        let client: Arc<VaultClient> = self.vault_client().await?;
        kv2::delete_latest(client.as_ref(), &location.mount, &location.path)
            .await
            .map_err(|error| map_api_error(error, ErrorContext::Secret))?;
        self.cache.invalidate(&cache_key).await;
        Ok(())
    }

    pub async fn async_rotate_secret(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
    ) -> Result<Value, Error> {
        async_rotate_secret(self, current_name, new_name, value).await
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

    async fn async_read_secret(&self, name: &str) -> Result<Option<SecretValue>, Error> {
        HashicorpVault::async_read_secret(self, name).await
    }

    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        description: Option<&str>,
    ) -> Result<Value, Error> {
        HashicorpVault::async_write_secret(self, name, value.clone(), description).await
    }

    async fn async_delete_secret(
        &self,
        name: &str,
        _recovery_window_in_days: i64,
    ) -> Result<(), Error> {
        HashicorpVault::async_delete_secret(self, name).await
    }
}

#[derive(Clone, Copy)]
enum ErrorContext {
    Login,
    Read,
    Secret,
}

fn cache_key(location: &SecretLocation) -> String {
    format!(
        "{:?}/{}/{}",
        location.namespace, location.mount, location.path
    )
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
