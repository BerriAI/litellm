use std::future::Future;
use std::sync::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::call_arguments::{CallArguments, parse_options};
use crate::ocr::OcrClient;
use crate::ocr::hooks::OcrHooks;
use crate::ocr::types::{
    LiteLLMOcrResponse, OcrConnection, OcrCredentialInputs, OcrDocument, OcrResponseFormat,
    PreparedOcrRequest, ResolvedOcrCredentials,
};

const HEALTH_CHECK_PDF_DATA_URI: &str = "data:application/pdf;base64,JVBERi0xLjQKJeLjz9MKMyAwIG9iago8PC9UeXBlIC9QYWdlCi9QYXJlbnQgMSAwIFIKL01lZGlhQm94IFswIDAgNjEyIDc5Ml0KL0NvbnRlbnRzIDQgMCBSCi9SZXNvdXJjZXMgPDwvRm9udCA8PC9GMSAyIDAgUj4+Pj4+PgplbmRvYmoKNCAwIG9iago8PC9MZW5ndGggNDQ+PgpzdHJlYW0KQlQKL0YxIDI0IFRmCjEwMCA3MDAgVGQKKHRlc3QpIFRqCkVUCmVuZHN0cmVhbQplbmRvYmoKMiAwIG9iago8PC9UeXBlIC9Gb250Ci9TdWJ0eXBlIC9UeXBlMQovQmFzZUZvbnQgL0hlbHZldGljYT4+CmVuZG9iagoxIDAgb2JqCjw8L1R5cGUgL1BhZ2VzCi9LaWRzIFszIDAgUl0KL0NvdW50IDE+PgplbmRvYmoKNSAwIG9iago8PC9UeXBlIC9DYXRhbG9nCi9QYWdlcyAxIDAgUj4+CmVuZG9iagp0cmFpbGVyCjw8L1NpemUgNgovUm9vdCA1IDAgUj4+CnN0YXJ0eHJlZgozMjQKJSVFT0Y=";

pub(crate) trait BaseOcrConfig: Send + Sync + Sized + 'static {
    type OcrParams: DeserializeOwned + Send + Sync;
    type ProviderRequest: Serialize + Send;
    type Environment: Send + Sync;

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        None
    }

    fn resolve_connection_params(&self, inputs: OcrCredentialInputs) -> ResolvedOcrCredentials {
        ResolvedOcrCredentials {
            api_key: inputs
                .dynamic_api_key
                .filter(|value| !value.value().is_empty())
                .or(inputs.api_key),
            api_base: inputs
                .dynamic_api_base
                .filter(|value| !value.value().is_empty())
                .or(inputs.api_base),
        }
    }

    fn get_health_check_document(&self) -> OcrDocument {
        OcrDocument::DocumentUrl {
            document_url: HEALTH_CHECK_PDF_DATA_URI.into(),
            extra_fields: Default::default(),
        }
    }

    fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
    ) -> impl Future<Output = Result<Self::Environment, crate::ocr::Error>> + Send;

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        optional_params: &Self::OcrParams,
        environment: &Self::Environment,
    ) -> Result<String, crate::ocr::Error>;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &[]
    }

    fn map_ocr_params(
        &self,
        arguments: &CallArguments,
        model: &str,
    ) -> Result<Self::OcrParams, crate::ocr::Error> {
        Ok(parse_options(
            &arguments
                .select(self.get_supported_ocr_params(model))
                .into(),
        )?)
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &Self::OcrParams,
        headers: &[(String, String)],
    ) -> Result<Self::ProviderRequest, crate::ocr::Error>;

    fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &Self::OcrParams,
        headers: &[(String, String)],
        _context: OcrRequestContext<'_>,
    ) -> impl Future<Output = Result<Self::ProviderRequest, crate::ocr::Error>> + Send {
        async move { self.transform_ocr_request(model, document, optional_params, headers) }
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: OcrResponseFormat,
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
            self.transform_ocr_response(model, &bytes, context.request_format)
        }
    }

    fn get_error_class(
        &self,
        error_message: String,
        status_code: u16,
        headers: Vec<(String, String)>,
    ) -> crate::ocr::Error {
        crate::ocr::Error::Provider {
            status: status_code,
            body: error_message,
            headers,
        }
    }
}

pub(crate) fn decode_and_normalize_response<T: DeserializeOwned>(
    model: &str,
    raw_response: &[u8],
    request_format: OcrResponseFormat,
    normalize: impl FnOnce(&str, T) -> Result<LiteLLMOcrResponse, crate::ocr::Error>,
) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
    let decoded = crate::ocr::wire::decode_response(
        raw_response,
        request_format == OcrResponseFormat::Native,
    )?;
    Ok(LiteLLMOcrResponse {
        provider_native_response: decoded.native,
        ..normalize(model, decoded.data)?
    })
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
