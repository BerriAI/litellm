use crate::auth::AuthError;
use crate::constants::{AZURE_AI_OCR_PATH, AZURE_DI_API_VERSION};
use crate::ocr::backends::OcrBackend;
use crate::ocr::error::OcrError;
use crate::ocr::formats::document_intelligence::{
    AzureDocumentIntelligenceOcrFormat,
    types::{AzureDocumentIntelligenceOperation, DocumentIntelligenceParams},
};
use crate::ocr::formats::mistral::{MistralOcrFormat, types::MistralOcrParams};
use crate::ocr::types::{OcrConnection, OcrDocument, OcrRequestFormat};
use crate::ocr::wire::{DecodedOcrResponse, encode_model_id};
use crate::providers::azure_ai::auth;

pub struct AzureMistralOcrBackend;

impl OcrBackend<MistralOcrFormat> for AzureMistralOcrBackend {
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
        client: &crate::ocr::OcrClient,
        document: OcrDocument,
        connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<OcrDocument, OcrError> {
        crate::ocr::document::inline_remote_document(
            client.document_fetcher(),
            document,
            connection,
        )
        .await
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

pub struct AzureDocumentIntelligenceOcrBackend;

impl OcrBackend<AzureDocumentIntelligenceOcrFormat> for AzureDocumentIntelligenceOcrBackend {
    fn complete_url(
        &self,
        connection: &OcrConnection,
        model: &str,
        params: &DocumentIntelligenceParams,
    ) -> Result<String, OcrError> {
        let endpoint = auth::resolve_document_intelligence_endpoint(
            connection.api_base.as_deref(),
            &|name| std::env::var(name).ok(),
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
        Ok(url)
    }

    async fn prepare_document(
        &self,
        _client: &crate::ocr::OcrClient,
        document: OcrDocument,
        _connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<OcrDocument, OcrError> {
        Ok(document)
    }

    fn preserve_native_response(&self, params: &DocumentIntelligenceParams) -> bool {
        params.request_format == OcrRequestFormat::Native
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

    async fn authenticate(
        &self,
        connection: &OcrConnection,
    ) -> Result<Vec<(String, String)>, AuthError> {
        auth::authenticate_document_intelligence(
            connection.extra_headers.clone(),
            connection.api_key.as_deref(),
            connection.azure_auth.as_ref(),
            &|name| std::env::var(name).ok(),
        )
        .await
    }
}
