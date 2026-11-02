use litellm_auth_azure::{AzureAuthService, AzureIdentity};
use litellm_auth_types::ResolvedCredential;
use litellm_config::AzureAdAuth;

use crate::Error;
use crate::error::CredentialStage;

use crate::{CredentialSource, Expiring, SharedEnvironment, SourceFuture};

const AZURE_REDIS_SCOPE: &str = "https://redis.azure.com/.default";

pub(crate) struct AzureAd {
    service: AzureAuthService,
    identity: AzureIdentity,
    username: Option<String>,
    lookup: SharedEnvironment,
}

impl AzureAd {
    pub(crate) fn new(auth: AzureAdAuth, lookup: SharedEnvironment) -> Self {
        // _build_azure_credential in litellm/_redis.py picks the credential class the same way
        let identity = match (auth.client_id, auth.tenant_id, auth.client_secret) {
            (Some(client_id), Some(tenant_id), Some(client_secret)) => {
                AzureIdentity::ClientSecret {
                    tenant_id,
                    client_id,
                    client_secret,
                }
            }
            (Some(client_id), _, _) => AzureIdentity::ManagedIdentity { client_id },
            (None, _, _) => AzureIdentity::Default,
        };
        Self {
            service: AzureAuthService::default(),
            identity,
            username: auth.username,
            lookup,
        }
    }
}

impl CredentialSource for AzureAd {
    fn fetch(&self) -> SourceFuture<'_> {
        Box::pin(async move {
            let credential = self
                .service
                .identity_token(&self.identity, AZURE_REDIS_SCOPE, &|name| {
                    self.lookup.get(name)
                })
                .await
                .map_err(Error::credential(CredentialStage::Identity))?
                .ok_or(Error::MissingCredential { provider: "Azure" })?;
            let (token, expires_at) = match credential {
                ResolvedCredential::AccessToken { token, expires_on } => (token, expires_on),
                ResolvedCredential::Static(token) => (token, None),
            };
            Ok(Expiring {
                username: self.username.clone(),
                password: token,
                expires_at,
            })
        })
    }
}
