use super::super::OcrAdapter;
use crate::ocr::OcrClient;
use crate::ocr::codecs::reducto::{self, ReductoLegacyParams, ReductoResponse};
use crate::ocr::error::{OcrError, OcrResponseError};
use crate::ocr::prepare::{
    _prepare_ocr_request, build_http_request, credential_env, guardrail_document,
};
use crate::ocr::registry::OcrProvider;
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};
use crate::ocr::wire::DecodedOcrResponse;

#[derive(Clone, Debug)]
pub(crate) struct ReductoLegacyAdapter;

impl OcrAdapter for ReductoLegacyAdapter {
    type ProviderResponse = ReductoResponse;
    const PROVIDER: OcrProvider = OcrProvider::Reducto;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn transform_ocr_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, OcrError> {
        let params: ReductoLegacyParams = _prepare_ocr_request(request)?;
        let headers = super::validate_environment(&request.connection, &credential_env)?;
        let url = super::get_complete_url(request.connection.api_base.as_deref(), "parse")?;
        let document = guardrail_document(request, &url).await?;
        let document =
            super::prepare_document(client, document, &request.connection, &headers).await?;
        let body = reducto::transform_legacy_ocr_request(&request.model, document, &params)?;
        build_http_request(client, request, &url, &headers, &body)
    }

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: Self::ProviderResponse,
    ) -> Result<LiteLLMOcrResponse, OcrResponseError> {
        reducto::transform_ocr_response(&request.model, response)
    }

    async fn read_response(
        &self,
        _client: &OcrClient,
        response: reqwest::Response,
        _url: &str,
        _headers: &[(String, String)],
        _request: &LiteLLMOcrRequest,
    ) -> Result<DecodedOcrResponse<Self::ProviderResponse>, OcrError> {
        crate::ocr::client::read_json_response(response, true).await
    }
}
