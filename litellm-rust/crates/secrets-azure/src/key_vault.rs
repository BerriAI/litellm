use std::sync::Arc;

use litellm_auth_azure::{AzureAuthInputs, AzureAuthService, ConfigValue};
use litellm_auth_types::{InputSource, Sourced};
use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::{AzureOperationContext, BaseSecretManager, Secret, SecretValue};
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC};
use serde::Deserialize;

use crate::Error;

const AZURE_KEY_VAULT_URI: &str = "AZURE_KEY_VAULT_URI";
const API_VERSION: &str = "7.4";
const PATH_SEGMENT: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~');

#[derive(Clone)]
pub struct AzureKeyVault {
    client: reqwest::Client,
    vault: reqwest::Url,
    auth: Arc<AzureAuthService>,
    inputs: Arc<AzureAuthInputs>,
    environment: Arc<dyn Lookup + Send + Sync>,
}

#[derive(Deserialize)]
struct SecretResponse {
    value: Option<String>,
}

impl AzureKeyVault {
    pub fn with_client(
        client: reqwest::Client,
        vault: reqwest::Url,
        environment: Arc<dyn Lookup + Send + Sync>,
    ) -> Result<Self, Error> {
        if vault.host_str().is_none() {
            return Err(Error::VaultUri);
        }
        let inputs = AzureAuthInputs {
            azure_scope: ConfigValue::Value(Sourced::new(
                scope_for(&vault),
                InputSource::Deployment,
            )),
            enable_azure_ad_token_refresh: Sourced::new(true, InputSource::Deployment),
            ..AzureAuthInputs::default()
        };
        Ok(Self {
            client,
            vault,
            auth: Arc::new(AzureAuthService::default()),
            inputs: Arc::new(inputs),
            environment,
        })
    }

    pub fn new(environment: Arc<dyn Lookup + Send + Sync>) -> Result<Self, Error> {
        let value = environment
            .get(AZURE_KEY_VAULT_URI)
            .ok_or(Error::MissingEnvironment(AZURE_KEY_VAULT_URI))?;
        let vault = reqwest::Url::parse(&value).map_err(|_| Error::VaultUri)?;
        if vault.scheme() != "https" || vault.host_str().is_none() {
            return Err(Error::VaultUri);
        }
        Self::with_client(reqwest::Client::new(), vault, environment)
    }

    pub fn scope(&self) -> &str {
        self.inputs
            .azure_scope
            .as_value()
            .map(|value| value.value().as_str())
            .unwrap_or_default()
    }

    pub async fn get_secret(&self, name: &str) -> Result<Option<Secret>, Error> {
        BaseSecretManager::async_read_secret(self, name, &AzureOperationContext::default())
            .await
            .map(|value| value.map(Secret::String))
    }

    async fn read(&self, name: &str) -> Result<Option<SecretValue>, Error> {
        let token = self
            .auth
            .get_azure_ad_token(&self.inputs, &|key| self.environment.get(key))
            .await?
            .ok_or(Error::MissingCredentials)?;
        let encoded_name = percent_encoding::utf8_percent_encode(name, PATH_SEGMENT);
        let url = self
            .vault
            .join(&format!("secrets/{encoded_name}?api-version={API_VERSION}"))
            .map_err(|_| Error::VaultUri)?;
        let response = self
            .client
            .get(url)
            .bearer_auth(token.value().secret().expose())
            .header(reqwest::header::ACCEPT, "application/json")
            .send()
            .await
            .map_err(Error::Http)?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if response.status() != reqwest::StatusCode::OK {
            return Err(Error::Status(response.status().as_u16()));
        }
        let payload: SecretResponse = response.json().await.map_err(Error::Http)?;
        let value = payload.value.ok_or(Error::MissingValue)?;
        Ok(Some(SecretValue::new(value)))
    }
}

fn scope_for(vault: &reqwest::Url) -> String {
    let host = vault.host_str().unwrap_or_default();
    let resource = host
        .split_once('.')
        .map_or(host, |(_, remainder)| remainder);
    format!("https://{resource}/.default")
}

impl BaseSecretManager for AzureKeyVault {
    type Error = Error;
    type Context = AzureOperationContext;

    async fn async_read_secret(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> Result<Option<SecretValue>, Error> {
        match context.timeout {
            Some(timeout) => tokio::time::timeout(timeout, self.read(name))
                .await
                .map_err(|_| Error::Timeout)?,
            None => self.read(name).await,
        }
    }
}

pub trait AzureTokenProvider: Send + Sync {
    fn get_token<'a>(
        &'a self,
        scope: &'a str,
        environment: &'a (dyn Lookup + Send + Sync),
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<SecretValue, Error>> + Send + 'a>>;
}

#[derive(Default)]
pub struct NativeAzureTokenProvider {
    auth: AzureAuthService,
}

impl AzureTokenProvider for NativeAzureTokenProvider {
    fn get_token<'a>(
        &'a self,
        scope: &'a str,
        environment: &'a (dyn Lookup + Send + Sync),
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<SecretValue, Error>> + Send + 'a>>
    {
        Box::pin(async move {
            let inputs = AzureAuthInputs {
                azure_scope: ConfigValue::Value(Sourced::new(
                    scope.to_owned(),
                    InputSource::Deployment,
                )),
                enable_azure_ad_token_refresh: Sourced::new(true, InputSource::Deployment),
                ..Default::default()
            };
            self.auth
                .get_azure_ad_token(&inputs, &|name| environment.get(name))
                .await?
                .map(|token| SecretValue::new(token.value().secret().expose()))
                .ok_or(Error::MissingCredentials)
        })
    }
}
