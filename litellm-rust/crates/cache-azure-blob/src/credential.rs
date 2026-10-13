use std::{
    fmt,
    sync::Arc,
    time::{Duration, SystemTime},
};

use azure_core::{
    credentials::{AccessToken, TokenCredential, TokenRequestOptions},
    error::ErrorKind,
    time::OffsetDateTime,
};
use litellm_auth_azure::{AzureAuthInputs, AzureAuthService};
use litellm_auth_types::ResolvedCredential;

const STATIC_TOKEN_LIFETIME: Duration = Duration::from_secs(300);
const LLM_TOKEN_ENV: &str = "AZURE_AD_TOKEN";

type EnvLookup = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

pub struct AzureBlobCredential {
    service: AzureAuthService,
    env_lookup: EnvLookup,
}

impl fmt::Debug for AzureBlobCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AzureBlobCredential")
    }
}

impl Default for AzureBlobCredential {
    fn default() -> Self {
        Self::new(
            AzureAuthService::default(),
            Arc::new(|name| std::env::var(name).ok()),
        )
    }
}

impl AzureBlobCredential {
    pub fn new(service: AzureAuthService, env_lookup: EnvLookup) -> Self {
        Self {
            service,
            env_lookup,
        }
    }
}

#[async_trait::async_trait]
impl TokenCredential for AzureBlobCredential {
    async fn get_token(
        &self,
        scopes: &[&str],
        _options: Option<TokenRequestOptions<'_>>,
    ) -> azure_core::Result<AccessToken> {
        let env_lookup = &self.env_lookup;
        let lookup = move |name: &str| (name != LLM_TOKEN_ENV).then(|| env_lookup(name)).flatten();
        let credential = self
            .service
            .get_azure_ad_token(
                &AzureAuthInputs::default_credential_for_scope(&scopes.join(" ")),
                &lookup,
            )
            .await
            .map_err(|error| {
                azure_core::Error::with_message(ErrorKind::Credential, error.to_string())
            })?
            .ok_or_else(|| {
                azure_core::Error::with_message(
                    ErrorKind::Credential,
                    "no Azure credential is available for blob storage",
                )
            })?;
        let (token, expires_on) = match credential.into_value() {
            ResolvedCredential::AccessToken { token, expires_on } => (token, expires_on),
            ResolvedCredential::Static(token) => (token, None),
        };
        let expires_on = expires_on.unwrap_or_else(|| SystemTime::now() + STATIC_TOKEN_LIFETIME);
        Ok(AccessToken::new(
            token.expose().to_string(),
            OffsetDateTime::from(expires_on),
        ))
    }
}
