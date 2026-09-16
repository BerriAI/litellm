use std::future::Future;
use std::sync::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::call_arguments::CallArguments;
use crate::ocr::OcrClient;
use crate::ocr::hooks::OcrHooks;
use crate::ocr::types::{
    LiteLLMOcrResponse, OcrConnection, OcrCredentialInputs, OcrDocument, OcrResponseFormat,
    PreparedOcrRequest, ResolvedOcrCredentials,
};

/// Output of `validate_environment`: whatever a provider resolves up front
/// (headers at minimum; Vertex also carries the project id).
pub(crate) trait OcrEnvironment: Send + Sync {
    fn headers(&self) -> &[(String, String)];
}

impl OcrEnvironment for Vec<(String, String)> {
    fn headers(&self) -> &[(String, String)] {
        self
    }
}

const HEALTH_CHECK_PDF_DATA_URI: &str = "data:application/pdf;base64,JVBERi0xLjQKJeLjz9MKMyAwIG9iago8PC9UeXBlIC9QYWdlCi9QYXJlbnQgMSAwIFIKL01lZGlhQm94IFswIDAgNjEyIDc5Ml0KL0NvbnRlbnRzIDQgMCBSCi9SZXNvdXJjZXMgPDwvRm9udCA8PC9GMSAyIDAgUj4+Pj4+PgplbmRvYmoKNCAwIG9iago8PC9MZW5ndGggNDQ+PgpzdHJlYW0KQlQKL0YxIDI0IFRmCjEwMCA3MDAgVGQKKHRlc3QpIFRqCkVUCmVuZHN0cmVhbQplbmRvYmoKMiAwIG9iago8PC9UeXBlIC9Gb250Ci9TdWJ0eXBlIC9UeXBlMQovQmFzZUZvbnQgL0hlbHZldGljYT4+CmVuZG9iagoxIDAgb2JqCjw8L1R5cGUgL1BhZ2VzCi9LaWRzIFszIDAgUl0KL0NvdW50IDE+PgplbmRvYmoKNSAwIG9iago8PC9UeXBlIC9DYXRhbG9nCi9QYWdlcyAxIDAgUj4+CmVuZG9iagp0cmFpbGVyCjw8L1NpemUgNgovUm9vdCA1IDAgUj4+CnN0YXJ0eHJlZgozMjQKJSVFT0Y=";

pub(crate) trait BaseOcrConfig: Send + Sync + Sized + 'static {
    type OcrParams: Send + Sync;
    type ProviderRequest: Serialize + Send;
    type Environment: OcrEnvironment;

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
    ) -> Result<Self::OcrParams, crate::ocr::Error>;

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

    /// Whether the `document` field in the outgoing body is owned by the
    /// provider transform and must survive guardrail body rewrites.
    /// Providers that inline remote URLs return `false` for remote documents
    /// so a hook may still replace the fetched payload.
    fn retains_document(&self, _document: &OcrDocument) -> bool {
        true
    }

    /// Provider-specific check applied to the composed body, both before and
    /// after guardrail hooks. Defaults to accepting any body.
    fn validate_request_body(&self, _body: &Value) -> Result<(), crate::ocr::Error> {
        Ok(())
    }

    /// Rust counterpart of `BaseLLMHTTPHandler._async_prepare_ocr_request`:
    /// map params, validate environment, build URL, transform, compose body.
    fn prepare_request(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
    ) -> impl Future<Output = Result<reqwest::Request, crate::ocr::Error>> + Send {
        async move {
            let params = self.map_ocr_params(&request.optional_params, &request.model)?;
            let environment = self.validate_environment(request, client).await?;
            let url = self.get_complete_url(request, &params, &environment)?;
            let headers = environment.headers();
            let body = self
                .async_transform_ocr_request(
                    &request.model,
                    request.document.clone(),
                    &params,
                    headers,
                    OcrRequestContext {
                        client,
                        connection: &request.connection,
                    },
                )
                .await?;
            crate::ocr::prepare::transform_request_body(
                client,
                request,
                &url,
                headers,
                self.retains_document(&request.document),
                body,
                |body| self.validate_request_body(body),
            )
            .await
        }
    }
}

pub(crate) fn decode_and_normalize_response<T: DeserializeOwned>(
    model: &str,
    raw_response: &[u8],
    request_format: OcrResponseFormat,
    normalize: impl FnOnce(&str, T) -> Result<LiteLLMOcrResponse, crate::ocr::Error>,
) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
    let decoded = crate::ocr::json::decode_response(
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
