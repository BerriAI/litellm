use crate::constants::MISTRAL_OCR_API_BASE;
use crate::ocr::backends::{BackendConfig, OcrBackend, OcrIntegration, PreparedOcrBackend};
use crate::ocr::error::{OcrError, OcrRequestError};
use crate::ocr::formats::mistral::{MistralOcrFormat, types::MistralOcrParams};
use crate::ocr::types::{OcrConnection, OcrDocument};
use crate::providers::mistral::auth;
use crate::url_utils::ApiUrl;

pub fn complete_url(api_base: Option<&str>) -> Result<String, OcrError> {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(MISTRAL_OCR_API_BASE);
    ApiUrl::parse(base)
        .and_then(|url| url.complete_path(&["v1", "ocr"]))
        .map(|url| url.into_string())
        .map_err(|_| {
            OcrRequestError::RequestField {
                path: "api_base".into(),
            }
            .into()
        })
}

#[derive(Clone, Debug)]
pub struct MistralDirect;

impl OcrIntegration for MistralDirect {
    type Backend = MistralBackend;
    type Format = MistralOcrFormat;
    type PreparedDocument = OcrDocument;
    const FORMAT: Self::Format = MistralOcrFormat;

    async fn prepare(
        &self,
        connection: &OcrConnection,
        _config: &BackendConfig<Self>,
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
            url: complete_url(connection.api_base.as_deref())?,
            headers,
        })
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
}

#[derive(Clone, Debug)]
pub struct MistralBackend;

impl OcrBackend for MistralBackend {
    type Config = ();
    const PROVIDER: crate::ocr::registry::OcrProvider = crate::ocr::registry::OcrProvider::Mistral;
}
