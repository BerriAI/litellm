use crate::ocr::backends::PreparedOcrBackend;
use crate::ocr::backends::reducto::{self, ReductoBackend};
use crate::ocr::error::OcrError;
use crate::ocr::formats::reducto::{ReductoParseLegacyFormat, types::ReductoLegacyParams};
use crate::ocr::types::OcrConnection;

use super::{BackendConfig, OcrIntegration};

#[derive(Clone, Debug)]
pub struct ReductoLegacy;

impl OcrIntegration for ReductoLegacy {
    type Backend = ReductoBackend;
    type Format = ReductoParseLegacyFormat;
    type DocumentPreparation = super::ReductoUpload;
    const GUARDRAIL_STAGE: super::GuardrailStage = super::GuardrailStage::Document;

    async fn prepare(
        &self,
        connection: &OcrConnection,
        _config: &BackendConfig<Self>,
        _model: &str,
        _params: &ReductoLegacyParams,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedOcrBackend, OcrError> {
        Ok(PreparedOcrBackend {
            url: reducto::complete_url(connection.api_base.as_deref(), "parse")?,
            headers: reducto::authenticate(connection, env_lookup)?,
        })
    }

    fn preserve_native_response(&self, _params: &ReductoLegacyParams) -> bool {
        true
    }
}
