use crate::constants::AZURE_AI_OCR_PATH;
use crate::ocr::backends::PreparedOcrBackend;
use crate::ocr::backends::azure_ai::{self, AzureBackend};
use crate::ocr::document::InlineOcrDocument;
use crate::ocr::error::{OcrError, OcrRequestError};
use crate::ocr::formats::mistral::{MistralOcrFormat, types::MistralOcrParams};
use crate::ocr::types::OcrConnection;
use crate::url_utils::ApiUrl;

use super::{BackendConfig, OcrIntegration};

#[derive(Clone, Debug)]
pub struct AzureMistral;

impl OcrIntegration for AzureMistral {
    type Backend = AzureBackend;
    type Format = MistralOcrFormat;
    type DocumentPreparation = super::RequireInline;
    const FORMAT: Self::Format = MistralOcrFormat;

    async fn prepare(
        &self,
        connection: &OcrConnection,
        config: &BackendConfig<Self>,
        _model: &str,
        _params: &MistralOcrParams,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedOcrBackend, OcrError> {
        let base = azure_ai::resolve_mistral_api_base(connection, env_lookup)?;
        let path: Vec<&str> = AZURE_AI_OCR_PATH.trim_matches('/').split('/').collect();
        let url = ApiUrl::parse(&base)
            .and_then(|url| url.complete_path(&path))
            .map(|url| url.into_string())
            .map_err(|_| OcrRequestError::RequestField {
                path: "api_base".into(),
            })?;
        let headers = azure_ai::authenticate_mistral(connection, config, env_lookup).await?;
        Ok(PreparedOcrBackend { url, headers })
    }

    fn validate_request_body(
        &self,
        body: &crate::ocr::formats::mistral::types::MistralOcrRequest,
    ) -> Result<(), OcrRequestError> {
        InlineOcrDocument::try_from(body.document.clone())?;
        Ok(())
    }
}
