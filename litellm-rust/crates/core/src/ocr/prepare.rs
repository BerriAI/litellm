use super::backends::OcrBackend;
use super::formats::OcrFormat;
use super::registry::OcrIntegration;
use super::types::{OcrConnection, OcrDocument};
use crate::Error;

pub use super::registry::{
    OcrIntegrationKind, OcrIntegrationRequest, OcrModel, OcrProvider, resolve_ocr_integration,
};

pub(crate) struct PreparedOcrRequest<F, B>
where
    F: OcrFormat,
    B: OcrBackend<F>,
{
    pub integration: OcrIntegration<F, B>,
    pub model: String,
    pub document: OcrDocument,
    pub params: F::MappedParams,
    pub backend_config: B::Config,
    pub connection: OcrConnection,
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) fn prepare_ocr_call<F, B>(
    integration: OcrIntegration<F, B>,
    model: String,
    document: OcrDocument,
    params: F::InputParams,
    backend_config: B::Config,
    connection: OcrConnection,
) -> Result<PreparedOcrRequest<F, B>, Error>
where
    F: OcrFormat,
    B: OcrBackend<F>,
{
    let params = integration.format.map_ocr_params(params)?;
    Ok(PreparedOcrRequest {
        integration,
        model,
        document,
        params,
        backend_config,
        connection,
    })
}

pub(crate) fn credential_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

#[cfg(test)]
#[path = "../../tests/ocr_prepare.rs"]
mod tests;
