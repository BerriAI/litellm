use std::future::Future;

use serde::de::DeserializeOwned;

use crate::ocr::OcrClient;
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrResponseFormat};
use crate::ocr::wire::DecodedOcrResponse;
use crate::params::OpaqueParams;

pub(crate) trait BaseOcrConfig: Send + Sync + Sized + 'static {
    type ProviderResponse: DeserializeOwned + Send;

    fn get_supported_ocr_params(&self, model: &str) -> &'static [&'static str];

    fn map_ocr_params(&self, _model: &str, params: &OpaqueParams) -> OpaqueParams {
        params
            .provider_params()
            .without(&["extra_body", "model", "document"])
    }

    fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> impl Future<Output = Result<reqwest::Request, crate::ocr::Error>> + Send;

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: Self::ProviderResponse,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error>;

    fn read_response(
        &self,
        _client: &OcrClient,
        response: reqwest::Response,
        _url: &str,
        _headers: &[(String, String)],
        request: &LiteLLMOcrRequest,
    ) -> impl Future<Output = Result<DecodedOcrResponse<Self::ProviderResponse>, crate::ocr::Error>> + Send
    {
        async move {
            let bytes = crate::ocr::client::read_response_bytes(
                response,
                request.connection.max_response_bytes,
            )
            .await?;
            crate::ocr::handler::post_call(&request.hooks, &bytes).await?;
            crate::ocr::wire::decode_response(
                &bytes,
                request.response_format()? == OcrResponseFormat::Native,
            )
        }
    }
}
