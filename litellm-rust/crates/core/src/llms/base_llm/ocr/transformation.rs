use std::future::Future;
use std::sync::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::ocr::OcrClient;
use crate::ocr::hooks::OcrHooks;
use crate::ocr::types::{LiteLLMOcrResponse, OcrConnection, OcrDocument, OcrResponseFormat};
use crate::params::OpaqueParams;

pub(crate) trait BaseOcrConfig: Send + Sync + Sized + 'static {
    type OcrParams: DeserializeOwned + Send + Sync;
    type ProviderRequest: Serialize + DeserializeOwned + Send;
    type ProviderResponse: DeserializeOwned + Send;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &[]
    }

    fn map_ocr_params(
        &self,
        _non_default_params: &OpaqueParams,
        optional_params: &OpaqueParams,
        _model: &str,
    ) -> Result<Self::OcrParams, crate::ocr::Error> {
        crate::ocr::wire::decode_request_value(
            serde_json::Value::Object(optional_params.clone().into()),
            "optional_params",
        )
    }

    fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &Self::OcrParams,
        headers: &[(String, String)],
        context: OcrRequestContext<'_>,
    ) -> impl Future<Output = Result<Self::ProviderRequest, crate::ocr::Error>> + Send;

    fn normalize_response(
        &self,
        model: &str,
        response: Self::ProviderResponse,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error>;

    fn async_transform_ocr_response(
        &self,
        model: &str,
        raw_response: reqwest::Response,
        context: OcrResponseContext<'_>,
    ) -> impl Future<Output = Result<LiteLLMOcrResponse, crate::ocr::Error>> + Send {
        async move {
            let bytes = crate::ocr::client::read_response_bytes(
                raw_response,
                context.connection.max_response_bytes,
            )
            .await?;
            crate::ocr::handler::post_call(context.hooks, &bytes).await?;
            let decoded = crate::ocr::wire::decode_response::<Self::ProviderResponse>(
                &bytes,
                context.request_format == OcrResponseFormat::Native,
            )?;
            Ok(LiteLLMOcrResponse {
                provider_native_response: decoded.native,
                ..self.normalize_response(model, decoded.data)?
            })
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct OcrRequestContext<'a> {
    pub client: &'a OcrClient,
    pub connection: &'a OcrConnection,
}

#[derive(Clone, Copy)]
pub(crate) struct OcrResponseContext<'a> {
    pub client: &'a OcrClient,
    pub connection: &'a OcrConnection,
    pub hooks: &'a Arc<dyn OcrHooks>,
    pub request_format: OcrResponseFormat,
    pub url: &'a str,
    pub headers: &'a [(String, String)],
}
