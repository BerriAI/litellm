use crate::constants::MISTRAL_OCR_API_BASE;
use crate::ocr::backends::{OcrBackend, PreparedOcrBackend};
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
    type Config = ();

    fn provider_name(&self) -> &'static str {
        "mistral"
    }

    async fn prepare(
        &self,
        connection: &OcrConnection,
        _config: &Self::Config,
        _model: &str,
        _params: &MistralOcrParams,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedOcrBackend, OcrError> {
        let headers = auth::validate_environment(
            connection.extra_headers.clone(),
            connection.api_key.as_deref(),
            env_lookup,
        )?;
        Ok(PreparedOcrBackend {
            url: complete_url(connection.api_base.as_deref()),
            headers,
        })
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
}
