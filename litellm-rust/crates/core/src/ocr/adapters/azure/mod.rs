mod document_intelligence;
mod mistral;

use std::sync::OnceLock;

use crate::Error;
use crate::auth::azure::{AzureAuthInputs, AzureAuthService};

pub(crate) use document_intelligence::AzureDocumentIntelligenceAdapter;
pub(crate) use mistral::AzureMistralAdapter;

async fn resolve_entra(
    config: &AzureAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Option<String>, Error> {
    static SERVICE: OnceLock<AzureAuthService> = OnceLock::new();
    SERVICE
        .get_or_init(AzureAuthService::default)
        .resolve(config, env_lookup)
        .await
        .map(|credential| credential.map(|credential| credential.secret().expose().to_string()))
        .map_err(Error::from)
}
