use crate::auth::AuthError;
use crate::constants::AZURE_AI_OCR_PATH;
use crate::ocr::error::OcrError;
use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use crate::ocr::transformation::OcrProviderConfig;
use crate::ocr::types::{OcrConnection, OcrDocument, OcrResponseData};
use crate::providers::azure_ai::auth;
use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use crate::providers::mistral::ocr::types::{
    MistralOcrParams, MistralOcrRequest, MistralOcrResponse,
};

pub struct AzureAiOcrConfig;
pub const AZURE_AI_OCR_CONFIG: AzureAiOcrConfig = AzureAiOcrConfig;

impl OcrProviderConfig for AzureAiOcrConfig {
    type InputParams = MistralOcrParams;
    type MappedParams = MistralOcrParams;
    type PreparedDocument = OcrDocument;
    type RequestBody = MistralOcrRequest;
    type ResponseBody = MistralOcrResponse;
    fn map_ocr_params(
        &self,
        params: MistralOcrParams,
    ) -> Result<MistralOcrParams, OcrRequestError> {
        MISTRAL_OCR_CONFIG.map_ocr_params(params)
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        params: &MistralOcrParams,
    ) -> Result<MistralOcrRequest, OcrRequestError> {
        MISTRAL_OCR_CONFIG.transform_ocr_request(model, document, params)
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        response: MistralOcrResponse,
        params: &MistralOcrParams,
    ) -> Result<OcrResponseData, OcrResponseError> {
        MISTRAL_OCR_CONFIG.transform_ocr_response(model, response, params)
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn complete_url(
        &self,
        connection: &OcrConnection,
        _model: &str,
        _params: &MistralOcrParams,
    ) -> Result<String, OcrError> {
        let base = auth::resolve_api_base(connection.api_base.as_deref(), &|name| {
            std::env::var(name).ok()
        })?;
        Ok(format!("{}{AZURE_AI_OCR_PATH}", base.trim_end_matches('/')))
    }

    async fn prepare_document(
        &self,
        document: OcrDocument,
        connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<OcrDocument, OcrError> {
        crate::ocr::client::convert_document_url_to_data_uri(document, connection).await
    }

    async fn authenticate(
        &self,
        connection: &OcrConnection,
    ) -> Result<Vec<(String, String)>, AuthError> {
        auth::authenticate(
            connection.extra_headers.clone(),
            connection.api_key.as_deref(),
            connection.azure_auth.as_ref(),
            &|name| std::env::var(name).ok(),
        )
        .await
    }
}
