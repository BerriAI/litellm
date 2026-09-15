use std::future::Future;

use serde::de::DeserializeOwned;

use crate::ocr::OcrClient;
use crate::ocr::error::{OcrError, OcrResponseError};
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrResponseFormat};
use crate::ocr::wire::DecodedOcrResponse;
use crate::params::OpaqueParams;

pub(crate) trait BaseOcrConfig: Send + Sync + Sized + 'static {
    type ProviderResponse: DeserializeOwned + Send;

    fn get_supported_ocr_params(&self, model: &str) -> &'static [&'static str];

    fn map_ocr_params(&self, model: &str, params: &OpaqueParams) -> OpaqueParams {
        params.retain_supported(self.get_supported_ocr_params(model))
    }

    fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> impl Future<Output = Result<reqwest::Request, OcrError>> + Send;

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: Self::ProviderResponse,
    ) -> Result<LiteLLMOcrResponse, OcrResponseError>;

    fn read_response(
        &self,
        _client: &OcrClient,
        response: reqwest::Response,
        _url: &str,
        _headers: &[(String, String)],
        request: &LiteLLMOcrRequest,
    ) -> impl Future<Output = Result<DecodedOcrResponse<Self::ProviderResponse>, OcrError>> + Send
    {
        async move {
            let bytes = crate::ocr::client::read_response_bytes(
                response,
                request.connection.max_response_bytes,
            )
            .await?;
            crate::ocr::handler::post_call(&request.hooks, &bytes).await?;
            Ok(crate::ocr::wire::decode_response(
                &bytes,
                request.response_format()? == OcrResponseFormat::Native,
            )?)
        }
    }
}
