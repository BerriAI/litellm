use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use azure_core::credentials::{Secret, TokenCredential};
use azure_core::http::ClientMethodOptions;
use azure_identity::{
    ClientAssertion, ClientAssertionCredential, ClientSecretCredential, DeveloperToolsCredential,
    ManagedIdentityCredential, ManagedIdentityCredentialOptions, UserAssignedId,
    WorkloadIdentityCredential, WorkloadIdentityCredentialOptions,
};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::Error;
use crate::auth::{
    AuthHeaderKind, Environment, ExpiringToken, ResolvedAuth, SecretString, SystemClock,
    TokenCache, TokenProvider,
};
use crate::constants::{AZURE_AUTH_DEFAULT_SCOPE, AZURE_AUTH_TOKEN_REFRESH_WINDOW_SECS};

static TOKENS: OnceLock<TokenCache> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzureCredentialKind {
    ApiKey,
    StaticToken,
    ClientSecret,
    ClientAssertion,
    WorkloadIdentity,
    ManagedIdentity,
    Default,
}

pub struct AzureAuthInput<'a> {
    pub api_key: Option<&'a str>,
    pub api_key_env: &'static str,
    pub api_key_kind: AuthHeaderKind,
    pub params: &'a Map<String, Value>,
    pub environment: &'a dyn Environment,
}

enum AzureTokenCredential {
    ClientSecret(Arc<ClientSecretCredential>),
    ClientAssertion(Arc<ClientAssertionCredential<StaticClientAssertion>>),
    WorkloadIdentity(Arc<WorkloadIdentityCredential>),
    ManagedIdentity(Arc<ManagedIdentityCredential>),
    Default {
        workload_identity: Option<Arc<WorkloadIdentityCredential>>,
        managed_identity: Arc<ManagedIdentityCredential>,
        developer_tools: Option<Arc<DeveloperToolsCredential>>,
    },
}

impl AzureTokenCredential {
    async fn token(&self, scope: &str) -> azure_core::Result<azure_core::credentials::AccessToken> {
        match self {
            Self::ClientSecret(value) => value.get_token(&[scope], None).await,
            Self::ClientAssertion(value) => value.get_token(&[scope], None).await,
            Self::WorkloadIdentity(value) => value.get_token(&[scope], None).await,
            Self::ManagedIdentity(value) => value.get_token(&[scope], None).await,
            Self::Default {
                workload_identity,
                managed_identity,
                developer_tools,
            } => {
                if let Some(value) = workload_identity
                    && let Ok(token) = value.get_token(&[scope], None).await
                {
                    return Ok(token);
                }
                match managed_identity.get_token(&[scope], None).await {
                    Ok(token) => Ok(token),
                    Err(managed_identity_error) => match developer_tools {
                        Some(value) => value.get_token(&[scope], None).await,
                        None => Err(managed_identity_error),
                    },
                }
            }
        }
    }
}

#[derive(Clone)]
struct StaticClientAssertion(SecretString);

impl std::fmt::Debug for StaticClientAssertion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("[redacted]")
    }
}

#[async_trait]
impl ClientAssertion for StaticClientAssertion {
    async fn secret(
        &self,
        _options: Option<ClientMethodOptions<'_>>,
    ) -> azure_core::Result<String> {
        Ok(self.0.expose().to_string())
    }
}

struct AzureTokenProvider {
    credential: AzureTokenCredential,
    scope: String,
}

#[async_trait]
impl TokenProvider for AzureTokenProvider {
    async fn token(&self) -> Result<ExpiringToken, Error> {
        let token = self
            .credential
            .token(&self.scope)
            .await
            .map_err(|_| Error::Auth("Azure credential refresh failed".to_string()))?;
        let seconds = token
            .expires_on
            .unix_timestamp()
            .saturating_sub(
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64,
            )
            .max(0) as u64;
        Ok(ExpiringToken {
            token: SecretString::new(token.token.secret()),
            expires_at: Instant::now() + Duration::from_secs(seconds),
        })
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn param_or_env(
    params: &Map<String, Value>,
    param: &str,
    env: &str,
    environment: &dyn Environment,
) -> Option<String> {
    params
        .get(param)
        .and_then(Value::as_str)
        .and_then(|value| non_empty(Some(value)))
        .map(str::to_string)
        .or_else(|| {
            environment
                .get(env)
                .filter(|value| !value.trim().is_empty())
        })
}

fn refresh_enabled(params: &Map<String, Value>) -> bool {
    params
        .get("azure_ad_token_refresh")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub fn classify_azure_credential(input: &AzureAuthInput<'_>) -> Result<AzureCredentialKind, Error> {
    if non_empty(input.api_key).is_some()
        || input
            .environment
            .get(input.api_key_env)
            .is_some_and(|value| !value.trim().is_empty())
    {
        return Ok(AzureCredentialKind::ApiKey);
    }
    let azure_ad_token = param_or_env(
        input.params,
        "azure_ad_token",
        "AZURE_AD_TOKEN",
        input.environment,
    );
    if azure_ad_token
        .as_deref()
        .is_some_and(|token| !token.starts_with("oidc/"))
    {
        return Ok(AzureCredentialKind::StaticToken);
    }
    let tenant = param_or_env(
        input.params,
        "tenant_id",
        "AZURE_TENANT_ID",
        input.environment,
    );
    let client = param_or_env(
        input.params,
        "client_id",
        "AZURE_CLIENT_ID",
        input.environment,
    );
    let secret = param_or_env(
        input.params,
        "client_secret",
        "AZURE_CLIENT_SECRET",
        input.environment,
    );
    if tenant.is_some() && client.is_some() && secret.is_some() {
        return Ok(AzureCredentialKind::ClientSecret);
    }
    if azure_ad_token
        .as_deref()
        .is_some_and(|token| token.starts_with("oidc/"))
    {
        if tenant.is_some()
            && client.is_some()
            && param_or_env(
                input.params,
                "azure_ad_client_assertion",
                "AZURE_AD_CLIENT_ASSERTION",
                input.environment,
            )
            .is_some()
        {
            return Ok(AzureCredentialKind::ClientAssertion);
        }
        return Err(Error::Auth(
            "Azure OIDC authentication requires tenant_id, client_id, and a resolved client assertion"
                .to_string(),
        ));
    }
    if !refresh_enabled(input.params) {
        return Err(Error::Auth(
            "Missing Azure credentials; set an API key, azure_ad_token, service principal, or enable Azure token refresh"
                .to_string(),
        ));
    }
    let credential = input.environment.get("AZURE_CREDENTIAL");
    if credential.as_deref() == Some("WorkloadIdentityCredential") {
        return Ok(AzureCredentialKind::WorkloadIdentity);
    }
    if credential.as_deref() == Some("ManagedIdentityCredential")
        || input.environment.get("IDENTITY_ENDPOINT").is_some()
        || input.environment.get("MSI_ENDPOINT").is_some()
    {
        return Ok(AzureCredentialKind::ManagedIdentity);
    }
    Ok(AzureCredentialKind::Default)
}

fn cache_key(kind: AzureCredentialKind, values: &[&str]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{kind:?}"));
    values.iter().for_each(|value| {
        hasher.update(value.len().to_le_bytes());
        hasher.update(value.as_bytes());
    });
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn token_credential(
    kind: AzureCredentialKind,
    params: &Map<String, Value>,
    environment: &dyn Environment,
) -> Result<(String, AzureTokenCredential), Error> {
    let scope = param_or_env(params, "azure_scope", "AZURE_SCOPE", environment)
        .unwrap_or_else(|| AZURE_AUTH_DEFAULT_SCOPE.to_string());
    let tenant = param_or_env(params, "tenant_id", "AZURE_TENANT_ID", environment);
    let client = param_or_env(params, "client_id", "AZURE_CLIENT_ID", environment);
    let token_file = environment
        .get("AZURE_FEDERATED_TOKEN_FILE")
        .filter(|value| !value.trim().is_empty());
    let result = match kind {
        AzureCredentialKind::ClientSecret => {
            let tenant =
                tenant.ok_or_else(|| Error::Auth("Azure tenant ID is missing".to_string()))?;
            let client =
                client.ok_or_else(|| Error::Auth("Azure client ID is missing".to_string()))?;
            let secret = param_or_env(params, "client_secret", "AZURE_CLIENT_SECRET", environment)
                .ok_or_else(|| Error::Auth("Azure client secret is missing".to_string()))?;
            let key = cache_key(kind, &[&tenant, &client, &secret, &scope]);
            let credential =
                ClientSecretCredential::new(&tenant, client, Secret::new(secret), None)
                    .map_err(|_| Error::Auth("Azure service principal is invalid".to_string()))?;
            (key, AzureTokenCredential::ClientSecret(credential))
        }
        AzureCredentialKind::ClientAssertion => {
            let tenant =
                tenant.ok_or_else(|| Error::Auth("Azure tenant ID is missing".to_string()))?;
            let client =
                client.ok_or_else(|| Error::Auth("Azure client ID is missing".to_string()))?;
            let assertion = param_or_env(
                params,
                "azure_ad_client_assertion",
                "AZURE_AD_CLIENT_ASSERTION",
                environment,
            )
            .ok_or_else(|| Error::Auth("Azure client assertion is missing".to_string()))?;
            let key = cache_key(kind, &[&tenant, &client, &assertion, &scope]);
            let credential = ClientAssertionCredential::new(
                tenant,
                client,
                StaticClientAssertion(SecretString::new(assertion)),
                None,
            )
            .map_err(|_| Error::Auth("Azure client assertion is invalid".to_string()))?;
            (key, AzureTokenCredential::ClientAssertion(credential))
        }
        AzureCredentialKind::WorkloadIdentity => {
            let credential =
                WorkloadIdentityCredential::new(Some(WorkloadIdentityCredentialOptions {
                    tenant_id: tenant.clone(),
                    client_id: client.clone(),
                    token_file_path: token_file.clone().map(PathBuf::from),
                    ..Default::default()
                }))
                .map_err(|_| Error::Auth("Azure workload identity is invalid".to_string()))?;
            (
                cache_key(
                    kind,
                    &[
                        tenant.as_deref().unwrap_or(""),
                        client.as_deref().unwrap_or(""),
                        token_file.as_deref().unwrap_or(""),
                        &scope,
                    ],
                ),
                AzureTokenCredential::WorkloadIdentity(credential),
            )
        }
        AzureCredentialKind::ManagedIdentity => {
            let options = ManagedIdentityCredentialOptions {
                user_assigned_id: client.clone().map(UserAssignedId::ClientId),
                ..Default::default()
            };
            let credential = ManagedIdentityCredential::new(Some(options))
                .map_err(|_| Error::Auth("Azure managed identity is unavailable".to_string()))?;
            (
                cache_key(kind, &[client.as_deref().unwrap_or("system"), &scope]),
                AzureTokenCredential::ManagedIdentity(credential),
            )
        }
        AzureCredentialKind::Default => {
            let workload_identity = match (&tenant, &client, &token_file) {
                (Some(tenant_id), Some(client_id), Some(token_file_path)) => {
                    WorkloadIdentityCredential::new(Some(WorkloadIdentityCredentialOptions {
                        tenant_id: Some(tenant_id.clone()),
                        client_id: Some(client_id.clone()),
                        token_file_path: Some(PathBuf::from(token_file_path)),
                        ..Default::default()
                    }))
                    .ok()
                }
                _ => None,
            };
            let options = ManagedIdentityCredentialOptions {
                user_assigned_id: client.clone().map(UserAssignedId::ClientId),
                ..Default::default()
            };
            let managed_identity = ManagedIdentityCredential::new(Some(options)).map_err(|_| {
                Error::Auth("Azure default credentials are unavailable".to_string())
            })?;
            let developer_tools = DeveloperToolsCredential::new(None).ok();
            (
                cache_key(
                    kind,
                    &[
                        tenant.as_deref().unwrap_or(""),
                        client.as_deref().unwrap_or("system"),
                        token_file.as_deref().unwrap_or(""),
                        &scope,
                    ],
                ),
                AzureTokenCredential::Default {
                    workload_identity,
                    managed_identity,
                    developer_tools,
                },
            )
        }
        AzureCredentialKind::ApiKey | AzureCredentialKind::StaticToken => {
            return Err(Error::Auth(
                "Azure credential classification failed".to_string(),
            ));
        }
    };
    Ok((format!("{}:{scope}", result.0), result.1))
}

pub async fn resolve_azure_auth(input: AzureAuthInput<'_>) -> Result<ResolvedAuth, Error> {
    let kind = classify_azure_credential(&input)?;
    match kind {
        AzureCredentialKind::ApiKey => {
            let key = non_empty(input.api_key)
                .map(str::to_string)
                .or_else(|| input.environment.get(input.api_key_env))
                .ok_or_else(|| Error::Auth("Azure API key is missing".to_string()))?;
            Ok(ResolvedAuth::from_credential(input.api_key_kind, key))
        }
        AzureCredentialKind::StaticToken => {
            let token = param_or_env(
                input.params,
                "azure_ad_token",
                "AZURE_AD_TOKEN",
                input.environment,
            )
            .ok_or_else(|| Error::Auth("Azure access token is missing".to_string()))?;
            Ok(ResolvedAuth::Bearer(SecretString::new(token)))
        }
        kind => {
            let scope = param_or_env(
                input.params,
                "azure_scope",
                "AZURE_SCOPE",
                input.environment,
            )
            .unwrap_or_else(|| AZURE_AUTH_DEFAULT_SCOPE.to_string());
            let (key, credential) = token_credential(kind, input.params, input.environment)?;
            let provider = AzureTokenProvider { credential, scope };
            let token = TOKENS
                .get_or_init(|| {
                    TokenCache::new(
                        Duration::from_secs(AZURE_AUTH_TOKEN_REFRESH_WINDOW_SECS),
                        Arc::new(SystemClock),
                    )
                })
                .token(key, &provider)
                .await?;
            Ok(ResolvedAuth::Bearer(token))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_environment(_: &str) -> Option<String> {
        None
    }

    fn input<'a>(
        api_key: Option<&'a str>,
        params: &'a Map<String, Value>,
        env: &'a dyn Environment,
    ) -> AzureAuthInput<'a> {
        AzureAuthInput {
            api_key,
            api_key_env: "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
            api_key_kind: AuthHeaderKind::Header("Ocp-Apim-Subscription-Key"),
            params,
            environment: env,
        }
    }

    #[test]
    fn api_key_precedes_static_token_and_service_principal() {
        let params = Map::from_iter([
            (
                "azure_ad_token".to_string(),
                Value::String("token".to_string()),
            ),
            ("tenant_id".to_string(), Value::String("tenant".to_string())),
            ("client_id".to_string(), Value::String("client".to_string())),
            (
                "client_secret".to_string(),
                Value::String("secret".to_string()),
            ),
        ]);
        assert_eq!(
            classify_azure_credential(&input(Some("key"), &params, &empty_environment)).unwrap(),
            AzureCredentialKind::ApiKey
        );
    }

    #[test]
    fn static_token_precedes_service_principal_and_refresh_requires_opt_in() {
        let token_params = Map::from_iter([
            (
                "azure_ad_token".to_string(),
                Value::String("token".to_string()),
            ),
            ("tenant_id".to_string(), Value::String("tenant".to_string())),
            ("client_id".to_string(), Value::String("client".to_string())),
            (
                "client_secret".to_string(),
                Value::String("secret".to_string()),
            ),
        ]);
        assert_eq!(
            classify_azure_credential(&input(None, &token_params, &empty_environment)).unwrap(),
            AzureCredentialKind::StaticToken
        );
        let params = Map::new();
        assert!(classify_azure_credential(&input(None, &params, &empty_environment)).is_err());
    }

    #[test]
    fn oidc_reference_requires_a_resolved_assertion() {
        let params = Map::from_iter([
            (
                "azure_ad_token".to_string(),
                Value::String("oidc/token-reference".to_string()),
            ),
            ("tenant_id".to_string(), Value::String("tenant".to_string())),
            ("client_id".to_string(), Value::String("client".to_string())),
        ]);
        assert!(classify_azure_credential(&input(None, &params, &empty_environment)).is_err());

        let mut resolved = params;
        resolved.insert(
            "azure_ad_client_assertion".to_string(),
            Value::String("assertion".to_string()),
        );
        assert_eq!(
            classify_azure_credential(&input(None, &resolved, &empty_environment)).unwrap(),
            AzureCredentialKind::ClientAssertion
        );
    }

    #[test]
    fn refresh_uses_managed_identity_signals_or_default_chain() {
        let params = Map::from_iter([("azure_ad_token_refresh".to_string(), Value::Bool(true))]);
        let managed_environment = |name: &str| {
            (name == "IDENTITY_ENDPOINT").then(|| "https://identity.example".to_string())
        };

        assert_eq!(
            classify_azure_credential(&input(None, &params, &managed_environment)).unwrap(),
            AzureCredentialKind::ManagedIdentity
        );
        assert_eq!(
            classify_azure_credential(&input(None, &params, &empty_environment)).unwrap(),
            AzureCredentialKind::Default
        );
    }

    #[test]
    fn cache_keys_preserve_credential_field_boundaries() {
        assert_ne!(
            cache_key(AzureCredentialKind::ClientSecret, &["ab", "c"]),
            cache_key(AzureCredentialKind::ClientSecret, &["a", "bc"])
        );
    }
}
