mod cohere;
mod document_intelligence;
mod mistral;

use std::sync::OnceLock;

use crate::Error;
use crate::auth::error::AuthConfigurationError;
use crate::auth::{InputSource, Sourced};
use crate::ocr::error::OcrError;
use crate::ocr::types::OcrConnection;
use crate::providers::azure_ai::auth::{AzureAuthInputs, AzureAuthService};

pub(crate) use cohere::AzureCohereAdapter;
pub(crate) use document_intelligence::AzureDocumentIntelligenceAdapter;
pub(crate) use mistral::AzureMistralAdapter;
pub(super) use mistral::validate_environment as validate_ai_environment;

async fn resolve_entra(
    config: &AzureAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Option<Sourced<String>>, Error> {
    static SERVICE: OnceLock<AzureAuthService> = OnceLock::new();
    SERVICE
        .get_or_init(AzureAuthService::default)
        .get_azure_ad_token(config, env_lookup)
        .await
        .or_else(|error| match error {
            crate::AuthError::EmptyAzureToken => Ok(None),
            other => Err(other),
        })
        .map(|credential| {
            credential.map(|credential| {
                let source = credential.source();
                let value = credential.value().secret().expose().to_string();
                Sourced::new(value, source)
            })
        })
        .map_err(Error::from)
}

fn validate_destination(
    connection: &OcrConnection,
    credential_source: InputSource,
) -> Result<(), OcrError> {
    if connection.api_base.is_some()
        && connection.api_base_source == InputSource::Request
        && credential_source != InputSource::Request
    {
        return Err(Error::from(crate::AuthError::Configuration(
            AuthConfigurationError::RequestAzureCredentialDestination,
        ))
        .into());
    }
    Ok(())
}
