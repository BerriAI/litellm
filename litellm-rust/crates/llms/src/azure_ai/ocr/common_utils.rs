use std::sync::OnceLock;

use litellm_auth::{InputSource, Sourced};
use litellm_auth_azure::{AzureAuthInputs, AzureAuthService};

use crate::base_llm::ocr::{
    error::Error,
    transformation::{OcrConnection, PreparedOcrRequest},
};

pub(crate) fn azure_auth_inputs(request: &PreparedOcrRequest) -> Result<AzureAuthInputs, Error> {
    Ok(AzureAuthInputs {
        azure_ad_token_provider: request.azure_ad_token_provider.clone(),
        ..AzureAuthInputs::from_sourced_optional_params(
            &request.optional_params,
            &request.input_sources,
        )?
    }
    .or_configured_token_refresh(request.connection.settings.enable_azure_ad_token_refresh))
}

pub(super) async fn resolve_entra(
    config: &AzureAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Option<Sourced<String>>, Error> {
    static SERVICE: OnceLock<AzureAuthService> = OnceLock::new();
    SERVICE
        .get_or_init(AzureAuthService::default)
        .get_azure_ad_token(config, env_lookup)
        .await
        .or_else(|error| match error {
            litellm_auth::Error::EmptyAzureToken => Ok(None),
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

pub(super) fn validate_destination(
    connection: &OcrConnection,
    credential_source: InputSource,
) -> Result<(), Error> {
    if connection.api_base.is_some()
        && connection.api_base_source == InputSource::Request
        && credential_source != InputSource::Request
    {
        return Err(litellm_auth::Error::RequestAzureCredentialDestination.into());
    }
    Ok(())
}
