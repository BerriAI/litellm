use crate::ocr::backends::PreparedOcrBackend;
use crate::ocr::backends::mistral::{self, MistralBackend};
use crate::ocr::error::OcrError;
use crate::ocr::formats::mistral::{MistralOcrFormat, types::MistralOcrParams};
use crate::ocr::types::OcrConnection;

use super::{BackendConfig, OcrIntegration};

#[derive(Clone, Debug)]
pub struct MistralDirect;

impl OcrIntegration for MistralDirect {
    type Backend = MistralBackend;
    type Format = MistralOcrFormat;
    type DocumentPreparation = super::PassThrough;
    const FORMAT: Self::Format = MistralOcrFormat;

    async fn prepare(
        &self,
        connection: &OcrConnection,
        _config: &BackendConfig<Self>,
        _model: &str,
        _params: &MistralOcrParams,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedOcrBackend, OcrError> {
        Ok(PreparedOcrBackend {
            url: mistral::complete_url(connection.api_base.as_deref())?,
            headers: mistral::authenticate(connection, env_lookup)?,
        })
    }
}
