use std::sync::Arc;

use litellm_auth_azure::{AzureAuthInputs, AzureAuthService};
use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::{Secret, SecretValue};
use serde::Deserialize;

use crate::Error;

const AZURE_KEY_VAULT_URI: &str = "AZURE_KEY_VAULT_URI";
const API_VERSION: &str = "7.4";

#[derive(Clone)]
pub struct AzureKeyVault {
    client: reqwest::Client,
    vault: reqwest::Url,
    auth: Arc<AzureAuthService>,
    inputs: AzureAuthInputs,
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
        let scope = scope_for(&vault);
        let inputs = AzureAuthInputs::from_sourced_optional_params(
            serde_json::json!({
                "azure_scope": scope,
                "enable_azure_ad_token_refresh": true,
            })
            .as_object()
            .expect("static Azure auth inputs object"),
            &std::collections::BTreeMap::new(),
        )?;
        Ok(Self {
            client,
            vault,
            auth: Arc::new(AzureAuthService::default()),
            inputs,
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

    pub async fn get_secret_from_azure_key_vault(
        &self,
        name: &str,
    ) -> Result<Option<Secret>, Error> {
        let token = self
            .auth
            .get_azure_ad_token(&self.inputs, &|key| self.environment.get(key))
            .await?
            .ok_or(Error::MissingCredentials)?;
        let encoded_name = encode_name(name);
        let url = self
            .vault
            .join(&format!("secrets/{encoded_name}?api-version={API_VERSION}"))
            .map_err(|_| Error::VaultUri)?;
        let response = self
            .client
            .get(url)
            .bearer_auth(token.value().secret().expose())
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
        Ok(Some(Secret::String(SecretValue::new(value))))
    }
}

fn scope_for(vault: &reqwest::Url) -> String {
    let host = vault.host_str().unwrap_or_default();
    let resource = host
        .split_once('.')
        .map_or(host, |(_, remainder)| remainder);
    format!("https://{resource}/.default")
}

fn encode_name(name: &str) -> String {
    percent_encoding::utf8_percent_encode(name, percent_encoding::NON_ALPHANUMERIC)
        .to_string()
        .replace("%2D", "-")
        .replace("%2E", ".")
        .replace("%5F", "_")
        .replace("%7E", "~")
}
