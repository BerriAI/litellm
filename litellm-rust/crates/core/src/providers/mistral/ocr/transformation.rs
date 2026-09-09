use super::types::{MistralOcrParams, MistralOcrRequest, MistralOcrResponse};
use crate::auth::AuthError;
use crate::constants::MISTRAL_OCR_API_BASE;
use crate::ocr::error::OcrRequestError;
use crate::ocr::error::{OcrError, OcrResponseError};
use crate::ocr::transformation::{OcrBackend, OcrFormat};
use crate::ocr::types::{OcrConnection, OcrDocument, OcrResponseData};
use crate::providers::mistral::auth;

pub struct MistralOcrBackend;
pub const MISTRAL_OCR_BACKEND: MistralOcrBackend = MistralOcrBackend;

pub fn complete_url(api_base: Option<&str>) -> String {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(MISTRAL_OCR_API_BASE)
        .trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{base}/ocr")
    } else {
        format!("{base}/v1/ocr")
    }
}

pub struct MistralOcrFormat;

impl OcrFormat for MistralOcrFormat {
    type InputParams = MistralOcrParams;
    type MappedParams = MistralOcrParams;
    type PreparedDocument = OcrDocument;
    type RequestBody = MistralOcrRequest;
    type ResponseBody = MistralOcrResponse;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn map_ocr_params(
        &self,
        params: MistralOcrParams,
    ) -> Result<MistralOcrParams, OcrRequestError> {
        Ok(params)
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        params: &MistralOcrParams,
    ) -> Result<MistralOcrRequest, OcrRequestError> {
        Ok(MistralOcrRequest {
            model: model.to_string(),
            document,
            params: params.clone(),
        })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_response(
        &self,
        model: &str,
        response: MistralOcrResponse,
        _params: &MistralOcrParams,
    ) -> Result<OcrResponseData, OcrResponseError> {
        Ok(OcrResponseData {
            document_annotation: response.document_annotation,
            usage_info: response.usage_info,
            ..OcrResponseData::new(
                response.model.unwrap_or_else(|| model.to_string()),
                response.pages,
            )
        })
    }
}

impl OcrBackend for MistralOcrBackend {
    type Format = MistralOcrFormat;
    const FORMAT: Self::Format = MistralOcrFormat;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn complete_url(
        &self,
        connection: &OcrConnection,
        _model: &str,
        _params: &MistralOcrParams,
    ) -> Result<String, OcrError> {
        Ok(complete_url(connection.api_base.as_deref()))
    }

    async fn prepare_document(
        &self,
        _http_client: &reqwest::Client,
        document: OcrDocument,
        _connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<OcrDocument, OcrError> {
        Ok(document)
    }

    async fn authenticate(
        &self,
        connection: &OcrConnection,
    ) -> Result<Vec<(String, String)>, AuthError> {
        auth::validate_environment(
            connection.extra_headers.clone(),
            connection.api_key.as_deref(),
            &|name| std::env::var(name).ok(),
        )
    }
}
