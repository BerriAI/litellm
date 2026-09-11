use std::future::Future;

use serde::de::DeserializeOwned;

use super::OcrClient;
use super::error::{OcrError, OcrResponseError};
use super::registry::OcrProvider;
use super::types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrResponseFormat};
use super::wire::DecodedOcrResponse;

mod azure;
mod mistral;
mod reducto;

pub(crate) use azure::{AzureDocumentIntelligenceAdapter, AzureMistralAdapter};
pub(crate) use mistral::MistralAdapter;
pub(crate) use reducto::{ReductoLegacyAdapter, ReductoV3Adapter};

/// Converts a complete LiteLLM OCR call to provider HTTP and normalizes its response.
pub(crate) trait OcrAdapter: Send + Sync + Sized + 'static {
    /// Provider JSON schema; direct and Vertex Mistral share `MistralOcrResponse`.
    type ProviderResponse: DeserializeOwned + Send;

    const PROVIDER: OcrProvider;

    /// Prepares the complete provider HTTP request.
    /// `request` contains the model, document, connection, and unmapped caller options.
    /// `client` supplies reusable provider and document HTTP clients.
    /// Returns the complete HTTP request, whereas Python returns body data.
    fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> impl Future<Output = Result<reqwest::Request, OcrError>> + Send;

    /// Python: `transform_ocr_response`.
    /// `request` supplies caller context, including the fallback model.
    /// `response` is the decoded provider payload; the output is the shared LiteLLM schema.
    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: Self::ProviderResponse,
    ) -> Result<LiteLLMOcrResponse, OcrResponseError>;

    /// Decodes provider HTTP; adapters may override this to poll asynchronous operations.
    /// Python performs that polling inside `async_transform_ocr_response`.
    /// `client` is reused for polling; `response` is the initial HTTP response.
    /// `url` and `headers` describe the submitted call; `request` supplies limits and format.
    fn read_response(
        &self,
        _client: &OcrClient,
        response: reqwest::Response,
        _url: &str,
        _headers: &[(String, String)],
        request: &LiteLLMOcrRequest,
    ) -> impl Future<Output = Result<DecodedOcrResponse<Self::ProviderResponse>, OcrError>> + Send
    {
        let retain_native = request
            .response_format()
            .map(|format| format == OcrResponseFormat::Native);
        async move { super::client::read_json_response(response, retain_native?).await }
    }
}

macro_rules! for_each_ocr_adapter {
    ($callback:ident) => {
        $callback! {
            Mistral, $crate::ocr::adapters::MistralAdapter, $crate::ocr::adapters::MistralAdapter, Mistral;
            AzureMistral, $crate::ocr::adapters::AzureMistralAdapter, $crate::ocr::adapters::AzureMistralAdapter, AzureAi;
            AzureDocumentIntelligence, $crate::ocr::adapters::AzureDocumentIntelligenceAdapter, $crate::ocr::adapters::AzureDocumentIntelligenceAdapter, AzureAi;
            ReductoLegacy, $crate::ocr::adapters::ReductoLegacyAdapter, $crate::ocr::adapters::ReductoLegacyAdapter, Reducto;
            ReductoV3, $crate::ocr::adapters::ReductoV3Adapter, $crate::ocr::adapters::ReductoV3Adapter, Reducto;
        }
    };
}

pub(crate) use for_each_ocr_adapter;
