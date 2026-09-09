use crate::auth::AuthError;
use crate::constants::AZURE_AI_OCR_PATH;
use crate::ocr::error::OcrError;
use crate::ocr::transformation::OcrBackend;
use crate::ocr::types::{OcrConnection, OcrDocument};
use crate::providers::azure_ai::auth;
use crate::providers::mistral::ocr::transformation::MistralOcrFormat;
use crate::providers::mistral::ocr::types::MistralOcrParams;

pub struct AzureMistralOcrBackend;
pub const AZURE_MISTRAL_OCR_BACKEND: AzureMistralOcrBackend = AzureMistralOcrBackend;

impl OcrBackend for AzureMistralOcrBackend {
    type Format = MistralOcrFormat;
    const FORMAT: Self::Format = MistralOcrFormat;

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
        http_client: &reqwest::Client,
        document: OcrDocument,
        connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<OcrDocument, OcrError> {
        crate::ocr::client::convert_document_url_to_data_uri(http_client, document, connection)
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
