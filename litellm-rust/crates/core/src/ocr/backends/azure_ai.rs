use serde_json::{Map, Value};

use crate::Error;
use crate::auth::azure::AzureAuthInputs;
use crate::ocr::backends::OcrBackend;
use crate::ocr::error::OcrError;
use crate::ocr::types::OcrConnection;
use crate::providers::azure_ai::auth;

#[derive(Clone, Debug)]
pub struct AzureBackend;

impl OcrBackend for AzureBackend {
    type Config = AzureAuthInputs;
    const PROVIDER: crate::ocr::registry::OcrProvider = crate::ocr::registry::OcrProvider::AzureAi;

    fn decode_config(params: &Map<String, Value>) -> Result<Self::Config, Error> {
        Ok(AzureAuthInputs::from_optional_params(params)?)
    }
}

pub(crate) fn resolve_integration(
    model: &crate::ocr::registry::OcrModel,
) -> crate::ocr::registry::OcrIntegrationKind {
    let model = model.as_str().to_ascii_lowercase();
    if model.contains("doc-intelligence") || model.contains("documentintelligence") {
        crate::ocr::registry::OcrIntegrationKind::AzureDocumentIntelligence
    } else {
        crate::ocr::registry::OcrIntegrationKind::AzureMistral
    }
}

pub(crate) async fn authenticate_mistral(
    connection: &OcrConnection,
    config: &AzureAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, OcrError> {
    Ok(auth::authenticate(
        connection.extra_headers.clone(),
        connection.api_key.as_deref(),
        Some(config),
        env_lookup,
    )
    .await?)
}

pub(crate) async fn prepare_document_intelligence_auth(
    connection: &OcrConnection,
    config: &AzureAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<(String, Vec<(String, String)>), OcrError> {
    let endpoint =
        auth::resolve_document_intelligence_endpoint(connection.api_base.as_deref(), env_lookup)?;
    let headers = auth::authenticate_document_intelligence(
        connection.extra_headers.clone(),
        connection.api_key.as_deref(),
        Some(config),
        env_lookup,
    )
    .await?;
    Ok((endpoint, headers))
}

pub(crate) fn resolve_mistral_api_base(
    connection: &OcrConnection,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<String, OcrError> {
    Ok(auth::resolve_api_base(
        connection.api_base.as_deref(),
        env_lookup,
    )?)
}
