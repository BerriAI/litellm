use serde_json::{Map, Value};

use crate::Error;
use crate::constants::VERTEX_OCR_DEFAULT_LOCATION;
use crate::ocr::backends::OcrBackend;
use crate::ocr::error::OcrError;
use crate::ocr::types::OcrConnection;
use crate::providers::vertex_ai::auth::{self, VertexAuthInputs};

#[derive(Clone, Debug)]
pub struct VertexBackend;

impl OcrBackend for VertexBackend {
    type Config = VertexAuthInputs;
    const PROVIDER: crate::ocr::registry::OcrProvider = crate::ocr::registry::OcrProvider::VertexAi;

    fn decode_config(params: &Map<String, Value>) -> Result<Self::Config, Error> {
        Ok(VertexAuthInputs::from_optional_params(params)?)
    }
}

pub(crate) fn resolve_integration(
    model: &crate::ocr::registry::OcrModel,
) -> crate::ocr::registry::OcrIntegrationKind {
    if model.as_str().to_ascii_lowercase().contains("deepseek") {
        crate::ocr::registry::OcrIntegrationKind::VertexDeepSeek
    } else {
        crate::ocr::registry::OcrIntegrationKind::VertexMistral
    }
}

pub(crate) async fn authenticate(
    connection: &OcrConnection,
    config: &VertexAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<auth::VertexAuthentication, OcrError> {
    Ok(auth::authenticate(
        connection.extra_headers.clone(),
        connection.api_key.as_deref(),
        config,
        None,
        env_lookup,
    )
    .await?)
}

pub(crate) fn location(
    config: &VertexAuthInputs,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    auth::resolve_location(config, env_lookup)
        .unwrap_or_else(|| VERTEX_OCR_DEFAULT_LOCATION.to_string())
}
