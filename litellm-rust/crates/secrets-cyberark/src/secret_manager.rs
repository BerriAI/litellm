use std::{fs, sync::Arc, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::{
    BaseSecretManager, SecretOperationContext, SecretValue, SecretWriteContext,
    validate_secret_name,
};
use moka::future::Cache;
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};

use crate::Error;

const CYBERARK_API_BASE: &str = "CYBERARK_API_BASE";
const CYBERARK_ACCOUNT: &str = "CYBERARK_ACCOUNT";
const CYBERARK_USERNAME: &str = "CYBERARK_USERNAME";
const CYBERARK_API_KEY: &str = "CYBERARK_API_KEY";
const CYBERARK_CLIENT_CERT: &str = "CYBERARK_CLIENT_CERT";
const CYBERARK_CLIENT_KEY: &str = "CYBERARK_CLIENT_KEY";
const CYBERARK_SSL_VERIFY: &str = "CYBERARK_SSL_VERIFY";
const CYBERARK_REFRESH_INTERVAL: &str = "CYBERARK_REFRESH_INTERVAL";
const DEFAULT_API_BASE: &str = "http://127.0.0.1:8080";
const DEFAULT_ACCOUNT: &str = "default";
const DEFAULT_USERNAME: &str = "admin";
const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(300);
const SECRET_NAME_SAFE: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~');

#[derive(Clone)]
pub struct CyberArkSecretManager {
    client: reqwest::Client,
    endpoint: reqwest::Url,
    account: String,
    username: String,
    api_key: SecretValue,
    token: Cache<(), SecretValue>,
    secrets: Cache<String, SecretValue>,
    authentication_lock: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeleteOutcome {
    NotSupported,
}

impl CyberArkSecretManager {
    pub fn with_client(
        client: reqwest::Client,
        endpoint: reqwest::Url,
        account: String,
        username: String,
        api_key: SecretValue,
        refresh_interval: Option<Duration>,
    ) -> Self {
        let endpoint = normalize_endpoint(endpoint);
        let ttl = refresh_interval
            .filter(|interval| !interval.is_zero())
            .unwrap_or(DEFAULT_REFRESH_INTERVAL);
        let token = Cache::builder().time_to_live(ttl).build();
        let secrets = Cache::builder().time_to_live(ttl).build();
        Self {
            client,
            endpoint,
            account,
            username,
            api_key,
            token,
            secrets,
            authentication_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    pub fn new(
        environment: Arc<dyn Lookup + Send + Sync>,
        enterprise_enabled: bool,
    ) -> Result<Self, Error> {
        let api_key = environment.get(CYBERARK_API_KEY).unwrap_or_default();
        let cert = environment.get(CYBERARK_CLIENT_CERT).unwrap_or_default();
        let key = environment.get(CYBERARK_CLIENT_KEY).unwrap_or_default();
        if api_key.is_empty() && (cert.is_empty() || key.is_empty()) {
            return Err(Error::MissingCredentials);
        }
        if !enterprise_enabled {
            return Err(Error::EnterpriseRequired);
        }
        let verify = environment
            .get(CYBERARK_SSL_VERIFY)
            .map(|value| !value.trim().eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        let mut builder = reqwest::Client::builder();
        if !verify {
            tracing::warn!(
                "CyberArk SSL verification is disabled. This is insecure and should only be used for testing with self-signed certificates."
            );
            builder = builder.danger_accept_invalid_certs(true);
        }
        if !cert.is_empty() && !key.is_empty() {
            let certificate = fs::read(cert).map_err(|_| Error::ClientCertificate)?;
            let private_key = fs::read(key).map_err(|_| Error::ClientCertificate)?;
            let identity = reqwest::Identity::from_pem(&[certificate, private_key].concat())
                .map_err(|_| Error::ClientCertificate)?;
            builder = builder.identity(identity);
        }
        let client = builder.build()?;
        let endpoint = reqwest::Url::parse(
            &environment
                .get(CYBERARK_API_BASE)
                .unwrap_or_else(|| DEFAULT_API_BASE.to_owned()),
        )
        .map_err(|_| Error::Endpoint)?;
        let account = environment
            .get(CYBERARK_ACCOUNT)
            .unwrap_or_else(|| DEFAULT_ACCOUNT.to_owned());
        let username = environment
            .get(CYBERARK_USERNAME)
            .unwrap_or_else(|| DEFAULT_USERNAME.to_owned());
        let refresh_interval = environment
            .get(CYBERARK_REFRESH_INTERVAL)
            .map(|value| {
                value
                    .parse::<u64>()
                    .map(Duration::from_secs)
                    .map_err(|_| Error::RefreshInterval)
            })
            .transpose()?;
        Ok(Self::with_client(
            client,
            endpoint,
            account,
            username,
            SecretValue::new(api_key),
            refresh_interval,
        ))
    }

    fn secret_url(&self, name: &str) -> Result<reqwest::Url, Error> {
        let encoded = utf8_percent_encode(name, SECRET_NAME_SAFE);
        self.endpoint
            .join(&format!("secrets/{}/variable/{}", self.account, encoded))
            .map_err(|_| Error::Endpoint)
    }

    async fn authenticate(&self, context: &SecretOperationContext) -> Result<SecretValue, Error> {
        if let Some(token) = self.token.get(&()).await {
            return Ok(token);
        }
        let _guard = self.authentication_lock.lock().await;
        if let Some(token) = self.token.get(&()).await {
            return Ok(token);
        }
        let url = self
            .endpoint
            .join(&format!(
                "authn/{}/{}/authenticate",
                self.account, self.username
            ))
            .map_err(|_| Error::Endpoint)?;
        let response = with_timeout(
            self.client.post(url).body(self.api_key.expose().to_owned()),
            context,
        )
        .send()
        .await?;
        if !response.status().is_success() {
            return Err(Error::AuthStatus(response.status().as_u16()));
        }
        let token = SecretValue::new(STANDARD.encode(response.text().await?));
        self.token.insert((), token.clone()).await;
        Ok(token)
    }

    async fn authorization_header(
        &self,
        context: &SecretOperationContext,
    ) -> Result<String, Error> {
        Ok(format!(
            "Token token=\"{}\"",
            self.authenticate(context).await?.expose()
        ))
    }

    pub async fn async_read_secret(&self, name: &str) -> Result<Option<SecretValue>, Error> {
        self.async_read_secret_with_context(name, &SecretOperationContext::default())
            .await
    }

    pub async fn async_read_secret_with_context(
        &self,
        name: &str,
        context: &SecretOperationContext,
    ) -> Result<Option<SecretValue>, Error> {
        if let Some(value) = self.secrets.get(name).await {
            return Ok(Some(value));
        }
        let response = with_timeout(
            self.client
                .get(self.secret_url(name)?)
                .header("Authorization", self.authorization_header(context).await?),
            context,
        )
        .send()
        .await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !response.status().is_success() {
            return Err(Error::Status(response.status().as_u16()));
        }
        let value = SecretValue::new(response.text().await?);
        self.secrets.insert(name.to_owned(), value.clone()).await;
        Ok(Some(value))
    }

    pub async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        description: Option<&str>,
    ) -> Result<(), Error> {
        self.async_write_secret_with_context(
            name,
            value,
            description,
            &SecretOperationContext::default(),
        )
        .await
    }

    pub async fn async_write_secret_with_context(
        &self,
        name: &str,
        value: &SecretValue,
        _description: Option<&str>,
        context: &SecretOperationContext,
    ) -> Result<(), Error> {
        validate_secret_name(name)?;
        self.ensure_variable_exists(name, context).await;
        let response = with_timeout(
            self.client
                .post(self.secret_url(name)?)
                .header("Authorization", self.authorization_header(context).await?)
                .body(value.expose().to_owned()),
            context,
        )
        .send()
        .await?;
        if !response.status().is_success() {
            return Err(Error::Status(response.status().as_u16()));
        }
        self.secrets.insert(name.to_owned(), value.clone()).await;
        Ok(())
    }

    async fn ensure_variable_exists(&self, name: &str, context: &SecretOperationContext) {
        let policy_url = self
            .endpoint
            .join(&format!("policies/{}/policy/root", self.account));
        let Ok(policy_url) = policy_url else {
            tracing::warn!("Could not build CyberArk policy endpoint");
            return;
        };
        let Ok(authorization) = self.authorization_header(context).await else {
            tracing::warn!("Could not authenticate while ensuring CyberArk variable exists");
            return;
        };
        let body = format!(
            "- !variable {}\n",
            serde_json::to_string(name).expect("serializing a string cannot fail")
        );
        let response = with_timeout(
            self.client
                .post(policy_url)
                .header("Authorization", authorization)
                .header("Content-Type", "application/x-yaml")
                .body(body),
            context,
        )
        .send()
        .await;
        match response {
            Ok(response) if response.status().is_success() => {}
            Ok(response)
                if matches!(
                    response.status(),
                    reqwest::StatusCode::CONFLICT | reqwest::StatusCode::UNPROCESSABLE_ENTITY
                ) =>
            {
                tracing::debug!(
                    "CyberArk variable policy already exists or conflicts: {}",
                    response.status()
                );
            }
            Ok(response) => {
                tracing::warn!(
                    "Could not ensure CyberArk variable exists: {}",
                    response.status()
                );
            }
            Err(error) => {
                tracing::warn!("Error ensuring CyberArk variable exists: {error}");
            }
        }
    }

    pub async fn async_delete_secret(
        &self,
        name: &str,
        recovery_window_in_days: Option<u32>,
    ) -> Result<DeleteOutcome, Error> {
        self.async_delete_secret_with_context(
            name,
            recovery_window_in_days,
            &SecretOperationContext::default(),
        )
        .await
    }

    pub async fn async_delete_secret_with_context(
        &self,
        name: &str,
        _recovery_window_in_days: Option<u32>,
        _context: &SecretOperationContext,
    ) -> Result<DeleteOutcome, Error> {
        tracing::warn!(
            "CyberArk Conjur does not support direct secret deletion. Secrets must be removed through policy updates."
        );
        self.secrets.invalidate(name).await;
        Ok(DeleteOutcome::NotSupported)
    }
}

impl BaseSecretManager for CyberArkSecretManager {
    type Error = Error;
    type WriteResponse = ();
    type DeleteResponse = DeleteOutcome;

    async fn async_read_secret(
        &self,
        name: &str,
        context: &SecretOperationContext,
    ) -> Result<Option<SecretValue>, Error> {
        self.async_read_secret_with_context(name, context).await
    }

    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext,
    ) -> Result<(), Error> {
        self.async_write_secret_with_context(
            name,
            value,
            context.description.as_deref(),
            &context.operation,
        )
        .await
    }

    async fn async_delete_secret(
        &self,
        name: &str,
        recovery_window_in_days: Option<u32>,
        context: &SecretOperationContext,
    ) -> Result<DeleteOutcome, Error> {
        self.async_delete_secret_with_context(name, recovery_window_in_days, context)
            .await
    }
}

fn with_timeout(
    request: reqwest::RequestBuilder,
    context: &SecretOperationContext,
) -> reqwest::RequestBuilder {
    match context.timeout() {
        Some(timeout) => request.timeout(timeout),
        None => request,
    }
}

fn normalize_endpoint(mut endpoint: reqwest::Url) -> reqwest::Url {
    if !endpoint.path().ends_with('/') {
        endpoint.set_path(&format!("{}/", endpoint.path()));
    }
    endpoint
}
