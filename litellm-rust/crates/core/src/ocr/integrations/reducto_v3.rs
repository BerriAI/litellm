use crate::ocr::backends::PreparedOcrBackend;
use crate::ocr::backends::reducto::{self, ReductoBackend};
use crate::ocr::error::OcrError;
use crate::ocr::formats::reducto::{ReductoParseV3Format, types::ReductoV3Params};
use crate::ocr::types::OcrConnection;

use super::{BackendConfig, OcrIntegration};

#[derive(Clone, Debug)]
pub struct ReductoV3;

impl OcrIntegration for ReductoV3 {
    type Backend = ReductoBackend;
    type Format = ReductoParseV3Format;
    type DocumentPreparation = super::ReductoUpload;
    const GUARDRAIL_STAGE: super::GuardrailStage = super::GuardrailStage::Document;

    async fn prepare(
        &self,
        connection: &OcrConnection,
        _config: &BackendConfig<Self>,
        _model: &str,
        _params: &ReductoV3Params,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedOcrBackend, OcrError> {
        Ok(PreparedOcrBackend {
            url: reducto::complete_url(connection.api_base.as_deref(), "parse")?,
            headers: reducto::authenticate(connection, env_lookup)?,
        })
    }

    fn preserve_native_response(&self, _params: &ReductoV3Params) -> bool {
        true
    }
}
