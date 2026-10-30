use litellm_auth_azure::{AzureAuthInputs, AzureAuthService, ConfigValue};
use litellm_auth_types::{InputSource, ResolvedCredential, Sourced};
use litellm_config::AzureAdAuth;

use crate::{CredentialSource, Expiring, SourceFuture};

const AZURE_REDIS_SCOPE: &str = "https://redis.azure.com/.default";

pub(crate) struct AzureAd {
    service: AzureAuthService,
    inputs: AzureAuthInputs,
    username: Option<String>,
}

impl AzureAd {
    pub(crate) fn new(auth: AzureAdAuth) -> Self {
        let configured = |value: Option<String>| match value {
            Some(value) => ConfigValue::Value(Sourced::new(value, InputSource::Deployment)),
            None => ConfigValue::Absent,
        };
        let selector = match (&auth.client_id, &auth.tenant_id, &auth.client_secret) {
            (Some(_), Some(_), Some(_)) => "ClientSecretCredential",
            (Some(_), _, _) => "ManagedIdentityCredential",
            (None, _, _) => "DefaultAzureCredential",
        };
        let inputs = AzureAuthInputs {
            client_id: configured(auth.client_id),
            tenant_id: configured(auth.tenant_id),
            client_secret: match auth.client_secret {
                Some(secret) => ConfigValue::Value(Sourced::new(secret, InputSource::Deployment)),
                None => ConfigValue::Absent,
            },
            azure_credential: configured(Some(selector.to_owned())),
            ..AzureAuthInputs::default_credential_for_scope(AZURE_REDIS_SCOPE)
        };
        Self {
            service: AzureAuthService::default(),
            inputs,
            username: auth.username,
        }
    }
}

impl CredentialSource for AzureAd {
    fn fetch(&self) -> SourceFuture<'_> {
        Box::pin(async move {
            let credential = self
                .service
                .get_azure_ad_token(&self.inputs, &|_| None)
                .await?
                .ok_or("no Azure credential is available for Redis")?
                .into_value();
            let (token, expires_at) = match credential {
                ResolvedCredential::AccessToken { token, expires_on } => (token, expires_on),
                ResolvedCredential::Static(token) => (token, None),
            };
            Ok(Expiring {
                username: self.username.clone(),
                password: token.expose().to_owned(),
                expires_at,
            })
        })
    }
}
