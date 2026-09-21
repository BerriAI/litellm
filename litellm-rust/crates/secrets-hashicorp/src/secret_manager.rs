use std::{
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::{
    BaseSecretManager, SecretValue, async_rotate_secret, validate_secret_name,
};
use moka::future::Cache;
use reqwest::Client;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::{Error, HashicorpVaultConfig};

const CACHE_CAPACITY: u64 = 200;

#[derive(Clone)]
struct CachedToken {
    token: SecretValue,
    expires_at: Option<Instant>,
}

#[derive(Clone)]
pub struct HashicorpVault {
    client: Arc<Client>,
    config: HashicorpVaultConfig,
    cache: Cache<String, SecretValue>,
    auth_token: Arc<Mutex<Option<CachedToken>>>,
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
        if !enterprise_enabled {
            return Err(Error::EnterpriseRequired);
        }
        let config: HashicorpVaultConfig =
            HashicorpVaultConfig::from_environment(environment.as_ref())?;
        let client: Client = client_for_config(&config)?;
        Self::with_client(client, config, enterprise_enabled)
    }

    pub fn with_client(
        client: Client,
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
            client: Arc::new(client),
            config,
            cache,
            auth_token: Arc::new(Mutex::new(None)),
        })
    }

    pub fn secret_url(&self, secret_name: &str) -> Result<String, Error> {
        validate_secret_name(secret_name).map_err(Error::InvalidSecretName)?;
        let namespace: String = self
            .config
            .secret_namespace()
            .map(|value| format!("{value}/"))
            .unwrap_or_default();
        let path_prefix: String = self
            .config
            .path_prefix
            .as_deref()
            .map(|value| format!("{value}/"))
            .unwrap_or_default();
        Ok(format!(
            "{}/v1/{}{}/data/{}{}",
            self.config.address, namespace, self.config.mount, path_prefix, secret_name
        ))
    }

    pub fn login_url(&self) -> Option<String> {
        self.config.approle.as_ref().map_or_else(
            || {
                self.config
                    .tls_cert
                    .as_ref()
                    .map(|_| format!("{}/v1/auth/cert/login", self.config.address))
            },
            |approle| {
                Some(format!(
                    "{}/v1/auth/{}/login",
                    self.config.address, approle.mount_path
                ))
            },
        )
    }

    pub fn config(&self) -> &HashicorpVaultConfig {
        &self.config
    }

    pub async fn async_read_secret(&self, secret_name: &str) -> Result<Option<SecretValue>, Error> {
        let url: String = self.secret_url(secret_name)?;
        if let Some(value) = self.cache.get(&url).await {
            return Ok(Some(value));
        }
        let token: SecretValue = self.vault_token().await?;
        let mut request = self
            .client
            .get(&url)
            .header("X-Vault-Token", token.expose());
        if let Some(namespace) = self.config.secret_namespace() {
            request = request.header("X-Vault-Namespace", namespace);
        }
        let response: reqwest::Response = request.send().await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !response.status().is_success() {
            return Err(Error::Status {
                status: response.status().as_u16(),
            });
        }
        let body = response.bytes().await?;
        let body: Value = serde_json::from_slice(&body).map_err(|_| Error::MalformedPayload)?;
        let data: &Value = body
            .get("data")
            .and_then(Value::as_object)
            .and_then(|value| value.get("data"))
            .ok_or(Error::MalformedPayload)?;
        let data: &serde_json::Map<String, Value> =
            data.as_object().ok_or(Error::MalformedPayload)?;
        let Some(value) = data.get("key") else {
            return Ok(None);
        };
        let value: &str = value.as_str().ok_or(Error::NonStringValue)?;
        let value: SecretValue = SecretValue::new(value);
        self.cache.insert(url, value.clone()).await;
        Ok(Some(value))
    }

    pub async fn async_write_secret(
        &self,
        secret_name: &str,
        value: SecretValue,
        description: Option<&str>,
    ) -> Result<Value, Error> {
        let url: String = self.secret_url(secret_name)?;
        let data: Value = match description {
            Some(description) => json!({"key": value.expose(), "description": description}),
            None => json!({"key": value.expose()}),
        };
        let token: SecretValue = self.vault_token().await?;
        let mut request = self
            .client
            .post(&url)
            .header("X-Vault-Token", token.expose());
        if let Some(namespace) = self.config.secret_namespace() {
            request = request.header("X-Vault-Namespace", namespace);
        }
        let response: reqwest::Response = request.json(&json!({"data": data})).send().await?;
        if !response.status().is_success() {
            return Err(Error::Status {
                status: response.status().as_u16(),
            });
        }
        self.cache.invalidate(&url).await;
        let body = response.bytes().await?;
        if body.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&body).map_err(|_| Error::MalformedPayload)
    }

    pub async fn async_delete_secret(&self, secret_name: &str) -> Result<(), Error> {
        let url: String = self.secret_url(secret_name)?;
        let token: SecretValue = self.vault_token().await?;
        let mut request = self
            .client
            .delete(&url)
            .header("X-Vault-Token", token.expose());
        if let Some(namespace) = self.config.secret_namespace() {
            request = request.header("X-Vault-Namespace", namespace);
        }
        let response: reqwest::Response = request.send().await?;
        if !response.status().is_success() {
            return Err(Error::Status {
                status: response.status().as_u16(),
            });
        }
        self.cache.invalidate(&url).await;
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

    async fn vault_token(&self) -> Result<SecretValue, Error> {
        let mut cached: tokio::sync::MutexGuard<'_, Option<CachedToken>> =
            self.auth_token.lock().await;
        if let Some(entry) = cached.as_ref()
            && entry
                .expires_at
                .is_none_or(|expires_at| expires_at > Instant::now())
        {
            return Ok(entry.token.clone());
        }
        let Some(login_url) = self.login_url() else {
            let Some(token) = self.config.token.clone() else {
                return Err(Error::NoAuthConfigured);
            };
            return Ok(token);
        };
        let body: Value = match (self.config.approle.as_ref(), self.config.tls_cert.as_ref()) {
            (Some(approle), _) => {
                json!({"role_id": approle.role_id, "secret_id": approle.secret_id.expose()})
            }
            (None, Some(tls)) => tls
                .role
                .as_deref()
                .map_or_else(|| json!({}), |role| json!({"name": role})),
            (None, None) => {
                let Some(token) = self.config.token.clone() else {
                    return Err(Error::NoAuthConfigured);
                };
                return Ok(token);
            }
        };
        let mut request = self.client.post(login_url).json(&body);
        if let Some(namespace) = self.config.login_namespace() {
            request = request.header("X-Vault-Namespace", namespace);
        }
        let response: reqwest::Response = request.send().await?;
        if !response.status().is_success() {
            return Err(Error::LoginStatus {
                status: response.status().as_u16(),
            });
        }
        let body = response.bytes().await?;
        let payload: Value = serde_json::from_slice(&body).map_err(|_| Error::MalformedLogin)?;
        let auth: &serde_json::Map<String, Value> = payload
            .get("auth")
            .and_then(Value::as_object)
            .ok_or(Error::MalformedLogin)?;
        let token: SecretValue = SecretValue::new(
            auth.get("client_token")
                .and_then(Value::as_str)
                .ok_or(Error::MalformedLogin)?,
        );
        let lease_duration: u64 = auth
            .get("lease_duration")
            .and_then(Value::as_u64)
            .ok_or(Error::MalformedLogin)?;
        let expires_at: Option<Instant> =
            (lease_duration > 0).then(|| Instant::now() + Duration::from_secs(lease_duration));
        *cached = Some(CachedToken {
            token: token.clone(),
            expires_at,
        });
        Ok(token)
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

fn client_for_config(config: &HashicorpVaultConfig) -> Result<Client, Error> {
    let mut builder: reqwest::ClientBuilder = Client::builder();
    if let Some(tls) = config.tls_cert.as_ref() {
        let cert: Vec<u8> = std::fs::read(&tls.cert_path).map_err(|source| Error::TlsIdentity {
            path: tls.cert_path.clone(),
            message: source.to_string(),
        })?;
        let key: Vec<u8> = std::fs::read(&tls.key_path).map_err(|source| Error::TlsIdentity {
            path: tls.key_path.clone(),
            message: source.to_string(),
        })?;
        let identity: reqwest::Identity = reqwest::Identity::from_pem(
            &[cert.as_slice(), key.as_slice()].concat(),
        )
        .map_err(|source| Error::TlsIdentity {
            path: tls.cert_path.clone(),
            message: source.to_string(),
        })?;
        builder = builder.identity(identity);
    }
    builder.build().map_err(Error::Request)
}
