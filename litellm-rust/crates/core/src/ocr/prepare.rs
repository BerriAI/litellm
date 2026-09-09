use super::backends::{HostConfig, InputParams, MappedParams, OcrIntegration};
use super::formats::OcrFormat;

use super::types::{OcrConnection, OcrDocument};
use crate::Error;

pub use super::registry::{
    OcrIntegrationKind, OcrIntegrationRequest, OcrModel, OcrProvider, resolve_ocr_integration,
};

pub(crate) struct MappedOcrRequest<I>
where
    I: OcrIntegration,
{
    pub integration: I,
    pub model: String,
    pub document: OcrDocument,
    pub params: MappedParams<I>,
    pub backend_config: HostConfig<I>,
    pub connection: OcrConnection,
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) fn prepare_ocr_call<I>(
    integration: I,
    model: String,
    document: OcrDocument,
    params: InputParams<I>,
    backend_config: HostConfig<I>,
    connection: OcrConnection,
) -> Result<MappedOcrRequest<I>, Error>
where
    I: OcrIntegration,
{
    let params = integration.format().map_ocr_params(params)?;
    Ok(MappedOcrRequest {
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
