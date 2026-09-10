use crate::auth::azure::AzureAuthInputs;
use crate::constants::{AZURE_AI_OCR_PATH, AZURE_DI_API_VERSION};
use crate::ocr::backends::{BackendConfig, OcrBackend, OcrIntegration, PreparedOcrBackend};
use crate::ocr::error::{OcrError, OcrRequestError};
use crate::ocr::formats::document_intelligence::{
    AzureDocumentIntelligenceOcrFormat,
    types::{AzureDocumentIntelligenceOperation, DocumentIntelligenceParams},
};
use crate::ocr::formats::mistral::{MistralOcrFormat, types::MistralOcrParams};
use crate::ocr::types::{OcrConnection, OcrDocument, OcrRequestFormat};
use crate::ocr::wire::DecodedOcrResponse;
use crate::providers::azure_ai::auth;
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};

fn encode_model_id(model: &str) -> Result<String, OcrRequestError> {
    const PATH_SEGMENT: &AsciiSet = &NON_ALPHANUMERIC
        .remove(b'-')
        .remove(b'_')
        .remove(b'.')
        .remove(b'~');
    let model = model.rsplit('/').next().unwrap_or(model);
    if matches!(model, "." | "..") {
        return Err(OcrRequestError::DotModel);
    }
    Ok(utf8_percent_encode(model, PATH_SEGMENT).to_string())
}

#[derive(Clone, Debug)]
pub struct AzureMistral;

impl OcrIntegration for AzureMistral {
    type Backend = AzureBackend;
    type Format = MistralOcrFormat;
    type PreparedDocument = crate::ocr::document::InlineOcrDocument;
    const FORMAT: Self::Format = MistralOcrFormat;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn prepare(
        &self,
        connection: &OcrConnection,
        config: &BackendConfig<Self>,
        _model: &str,
        _params: &MistralOcrParams,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedOcrBackend, OcrError> {
        let base = auth::resolve_api_base(connection.api_base.as_deref(), env_lookup)?;
        let headers = auth::authenticate(
            connection.extra_headers.clone(),
            connection.api_key.as_deref(),
            Some(config),
            env_lookup,
        )
        .await?;
        Ok(PreparedOcrBackend {
            url: format!("{}{AZURE_AI_OCR_PATH}", base.trim_end_matches('/')),
            headers,
        })
    }

    fn validate_request_body(
        &self,
        body: &crate::ocr::formats::mistral::types::MistralOcrRequest,
    ) -> Result<(), crate::ocr::error::OcrRequestError> {
        crate::ocr::document::InlineOcrDocument::try_from(body.document.clone())?;
        Ok(())
    }

    async fn prepare_document(
        &self,
        client: &crate::ocr::OcrClient,
        document: OcrDocument,
        connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<Self::PreparedDocument, OcrError> {
        crate::ocr::document::inline_remote_document(
            client.document_fetcher(),
            document,
            connection,
        )
        .await
    }
}

#[derive(Clone, Debug)]
pub struct AzureDocumentIntelligence;

impl OcrIntegration for AzureDocumentIntelligence {
    type Backend = AzureBackend;
    type Format = AzureDocumentIntelligenceOcrFormat;
    type PreparedDocument = OcrDocument;
    const FORMAT: Self::Format = AzureDocumentIntelligenceOcrFormat;

    async fn prepare(
        &self,
        connection: &OcrConnection,
        config: &BackendConfig<Self>,
        model: &str,
        params: &DocumentIntelligenceParams,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedOcrBackend, OcrError> {
        let endpoint = auth::resolve_document_intelligence_endpoint(
            connection.api_base.as_deref(),
            env_lookup,
        )?;
        let mut url = format!(
            "{}/documentintelligence/documentModels/{}:analyze?api-version={}",
            endpoint.trim_end_matches('/'),
            encode_model_id(model)?,
            AZURE_DI_API_VERSION
        );
        if let Some(pages) = &params.pages {
            url.push_str("&pages=");
            url.push_str(&pages.0);
        }
        if let Some(features) = &params.features {
            url.push_str("&features=");
            url.push_str(&features.0);
        }
        let headers = auth::authenticate_document_intelligence(
            connection.extra_headers.clone(),
            connection.api_key.as_deref(),
            Some(config),
            env_lookup,
        )
        .await?;
        Ok(PreparedOcrBackend { url, headers })
    }

    async fn prepare_document(
        &self,
        _client: &crate::ocr::OcrClient,
        document: OcrDocument,
        _connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<Self::PreparedDocument, OcrError> {
        Ok(document)
    }

    fn preserve_native_response(&self, params: &DocumentIntelligenceParams) -> bool {
        params.request_format == OcrRequestFormat::Native
    }

    fn supports_native_request_format(&self) -> bool {
        true
    }

    async fn read_response(
        &self,
        client: &crate::ocr::OcrClient,
        response: reqwest::Response,
        url: &str,
        headers: &[(String, String)],
        connection: &OcrConnection,
        params: &DocumentIntelligenceParams,
    ) -> Result<DecodedOcrResponse<AzureDocumentIntelligenceOperation>, OcrError> {
        super::azure_document_intelligence::polling::read_operation_response(
            client.provider_http(),
            response,
            url,
            headers,
            connection,
            params.request_format == OcrRequestFormat::Native,
        )
        .await
    }
}

#[derive(Clone, Debug)]
pub struct AzureBackend;

impl OcrBackend for AzureBackend {
    type Config = AzureAuthInputs;
    const PROVIDER: crate::ocr::registry::OcrProvider = crate::ocr::registry::OcrProvider::AzureAi;
}
