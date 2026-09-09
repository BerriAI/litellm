use crate::auth::AuthError;
use crate::constants::MISTRAL_OCR_API_BASE;
use crate::ocr::backends::OcrBackend;
use crate::ocr::error::OcrError;
use crate::ocr::formats::mistral::{MistralOcrFormat, types::MistralOcrParams};
use crate::ocr::types::{OcrConnection, OcrDocument};
use crate::providers::mistral::auth;

pub struct MistralOcrBackend;

pub fn complete_url(api_base: Option<&str>) -> String {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(MISTRAL_OCR_API_BASE)
        .trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{base}/ocr")
    } else {
        format!("{base}/v1/ocr")
    }
}

impl OcrBackend<MistralOcrFormat> for MistralOcrBackend {
    fn complete_url(
        &self,
        connection: &OcrConnection,
        _model: &str,
        _params: &MistralOcrParams,
    ) -> Result<String, OcrError> {
        Ok(complete_url(connection.api_base.as_deref()))
    }

    async fn prepare_document(
        &self,
        _http_client: &reqwest::Client,
        document: OcrDocument,
        _connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<OcrDocument, OcrError> {
        Ok(document)
    }

    async fn authenticate(
        &self,
        connection: &OcrConnection,
    ) -> Result<Vec<(String, String)>, AuthError> {
        auth::validate_environment(
            connection.extra_headers.clone(),
            connection.api_key.as_deref(),
            &|name| std::env::var(name).ok(),
        )
    }
}
