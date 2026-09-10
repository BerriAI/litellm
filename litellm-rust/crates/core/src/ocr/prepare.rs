use super::adapters::{AdapterConfig, InputParams, MappedParams, OcrAdapter};
use super::types::{OcrConnection, OcrDocument};
use crate::Error;

pub(crate) struct MappedOcrRequest<A: OcrAdapter> {
    pub adapter: A,
    pub model: String,
    pub document: OcrDocument,
    pub params: MappedParams<A>,
    pub config: AdapterConfig<A>,
    pub connection: OcrConnection,
}

pub(crate) fn prepare_ocr_call<A>(
    adapter: A,
    model: String,
    document: OcrDocument,
    params: InputParams<A>,
    config: AdapterConfig<A>,
    connection: OcrConnection,
) -> Result<MappedOcrRequest<A>, Error>
where
    A: OcrAdapter,
{
    let params = A::map_params(params)?;
    Ok(MappedOcrRequest {
        adapter,
        model,
        document,
        params,
        config,
        connection,
    })
}

pub(crate) fn credential_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}
