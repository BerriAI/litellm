use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{Map, Value};
use strum::EnumString;
use veil::Redact;

use crate::error::AuthError;
use crate::providers::auth::secret::SecretValue;
use crate::providers::auth::token::TokenCallerHandle;

pub const DEFAULT_AZURE_SCOPE: &str = "https://cognitiveservices.azure.com/.default";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ConfigValue<T> {
    #[default]
    Absent,
    ExplicitNone,
    Value(T),
}

impl<T> ConfigValue<T> {
    pub fn as_value(&self) -> Option<&T> {
        match self {
            Self::Value(value) => Some(value),
            Self::Absent | Self::ExplicitNone => None,
        }
    }
}

#[derive(Clone, Copy, Debug, EnumString, PartialEq, Eq, Hash)]
pub enum AzureCredentialType {
    ClientSecretCredential,
    ManagedIdentityCredential,
    DefaultAzureCredential,
    DeploymentIdentityCredential,
    WorkloadIdentityCredential,
}

#[derive(Clone, Debug, Default)]
pub struct AzureAuthInputs {
    pub azure_ad_token: ConfigValue<SecretValue>,
    pub azure_ad_token_provider: Option<TokenCallerHandle>,
    pub tenant_id: ConfigValue<String>,
    pub client_id: ConfigValue<String>,
    pub client_secret: ConfigValue<SecretValue>,
    pub azure_scope: ConfigValue<String>,
    pub azure_authority_host: ConfigValue<String>,
    pub azure_credential: ConfigValue<String>,
    pub federated_token_file: ConfigValue<String>,
    pub enable_azure_ad_token_refresh: bool,
}

impl AzureAuthInputs {
    pub fn from_optional_params(params: &Map<String, Value>) -> Result<Self, AuthError> {
        Ok(Self {
            azure_ad_token: secret_config(params, "azure_ad_token")?,
            azure_ad_token_provider: None,
            tenant_id: string_config(params, "tenant_id")?,
            client_id: string_config(params, "client_id")?,
            client_secret: secret_config(params, "client_secret")?,
            azure_scope: string_config(params, "azure_scope")?,
            azure_authority_host: string_config(params, "azure_authority_host")?,
            azure_credential: string_config(params, "azure_credential")?,
            federated_token_file: string_config(params, "azure_federated_token_file")?,
            enable_azure_ad_token_refresh: params
                .get("enable_azure_ad_token_refresh")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }
}

fn string_config(
    params: &Map<String, Value>,
    name: &str,
) -> Result<ConfigValue<String>, AuthError> {
    match params.get(name) {
        None => Ok(ConfigValue::Absent),
        Some(Value::Null) => Ok(ConfigValue::ExplicitNone),
        Some(Value::String(value)) => Ok(ConfigValue::Value(value.clone())),
        Some(_) => Err(AuthError::InvalidConfiguration(format!(
            "{name} must be a string or null"
        ))),
    }
}

fn secret_config(
    params: &Map<String, Value>,
    name: &str,
) -> Result<ConfigValue<SecretValue>, AuthError> {
    Ok(match string_config(params, name)? {
        ConfigValue::Absent => ConfigValue::Absent,
        ConfigValue::ExplicitNone => ConfigValue::ExplicitNone,
        ConfigValue::Value(value) => ConfigValue::Value(SecretValue::new(value)),
    })
}

pub(crate) type LookupFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<SecretValue>, AuthError>> + Send + 'a>>;

pub(crate) trait AzureValueLookup: std::fmt::Debug + Send + Sync {
    fn resolve<'a>(&'a self, reference: &'a str) -> LookupFuture<'a>;
}

#[derive(Clone, Redact)]
pub(crate) struct AzureValueLookupHandle(#[redact(with = "[REDACTED]")] Arc<dyn AzureValueLookup>);

impl AzureValueLookupHandle {
    pub(crate) fn new(lookup: Arc<dyn AzureValueLookup>) -> Self {
        Self(lookup)
    }

    pub(crate) async fn resolve(&self, reference: &str) -> Result<Option<SecretValue>, AuthError> {
        self.0.resolve(reference).await
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ProcessAzureValueLookup;

impl AzureValueLookup for ProcessAzureValueLookup {
    fn resolve<'a>(&'a self, reference: &'a str) -> LookupFuture<'a> {
        Box::pin(async move { resolve_process_value(reference) })
    }
}

fn resolve_process_value(reference: &str) -> Result<Option<SecretValue>, AuthError> {
    if let Some(name) = reference.strip_prefix("oidc/env/") {
        return Ok(std::env::var(name).ok().map(SecretValue::new));
    }
    if let Some(name) = reference.strip_prefix("oidc/env_path/") {
        let Some(path) = std::env::var(name).ok() else {
            return Ok(None);
        };
        return read_secret_file(&path, false).map(Some);
    }
    if let Some(path) = reference.strip_prefix("oidc/file/") {
        return read_secret_file(path, true).map(Some);
    }
    if reference.starts_with("oidc/") {
        return Err(AuthError::InvalidConfiguration(format!(
            "unsupported OIDC reference: {reference}"
        )));
    }
    Ok(None)
}

fn read_secret_file(path: &str, enforce_allowlist: bool) -> Result<SecretValue, AuthError> {
    let requested = std::path::Path::new(path);
    if !requested.is_absolute() {
        return Err(AuthError::InvalidConfiguration(
            "oidc/file path must be absolute".to_string(),
        ));
    }
    let resolved = requested.canonicalize().map_err(|error| {
        AuthError::Acquisition(format!("failed to resolve Azure assertion file: {error}"))
    })?;
    if enforce_allowlist {
        let allowed = oidc_allowed_directories()?;
        if !allowed
            .iter()
            .any(|directory| resolved.starts_with(directory))
        {
            return Err(AuthError::InvalidConfiguration(
                "oidc/file path is outside the allowed credential directories".to_string(),
            ));
        }
    }
    std::fs::read_to_string(resolved)
        .map(SecretValue::new)
        .map_err(|error| {
            AuthError::Acquisition(format!("failed to read Azure assertion file: {error}"))
        })
}

fn oidc_allowed_directories() -> Result<Vec<std::path::PathBuf>, AuthError> {
    let configured = std::env::var("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS").ok();
    let raw = configured
        .as_deref()
        .map(|value| {
            value
                .split(',')
                .filter(|part| !part.trim().is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec!["/var/run/secrets", "/run/secrets"]);
    raw.into_iter()
        .map(|path| {
            let candidate = std::path::Path::new(path.trim());
            if !candidate.is_absolute() {
                return Err(AuthError::InvalidConfiguration(
                    "OIDC allowed credential directories must be absolute".to_string(),
                ));
            }
            candidate.canonicalize().map_err(|error| {
                AuthError::InvalidConfiguration(format!(
                    "failed to resolve OIDC allowed credential directory: {error}"
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{AzureAuthInputs, AzureCredentialType, ConfigValue};

    #[test]
    fn selector_parsing_is_exact() {
        assert_eq!(
            "ClientSecretCredential".parse::<AzureCredentialType>(),
            Ok(AzureCredentialType::ClientSecretCredential)
        );
        assert!(
            "clientsecretcredential"
                .parse::<AzureCredentialType>()
                .is_err()
        );
    }

    #[test]
    fn defaults_preserve_absence() {
        let inputs = AzureAuthInputs::default();

        assert_eq!(inputs.tenant_id, ConfigValue::Absent);
        assert_eq!(inputs.azure_ad_token, ConfigValue::Absent);
    }

    #[test]
    fn parsing_distinguishes_null_empty_and_absent() {
        let params = json!({"tenant_id": null, "client_id": ""});
        let inputs = AzureAuthInputs::from_optional_params(params.as_object().unwrap()).unwrap();

        assert_eq!(inputs.tenant_id, ConfigValue::ExplicitNone);
        assert_eq!(inputs.client_id, ConfigValue::Value(String::new()));
        assert_eq!(inputs.client_secret, ConfigValue::Absent);
    }

    #[test]
    fn debug_does_not_expose_secrets() {
        let params = json!({"azure_ad_token": "token-value", "client_secret": "secret-value"});
        let inputs = AzureAuthInputs::from_optional_params(params.as_object().unwrap()).unwrap();
        let debug = format!("{inputs:?}");

        assert!(!debug.contains("token-value"));
        assert!(!debug.contains("secret-value"));
    }
}
