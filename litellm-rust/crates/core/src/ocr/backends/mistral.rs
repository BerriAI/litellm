use serde_json::{Map, Value};

use crate::Error;
use crate::constants::MISTRAL_OCR_API_BASE;
use crate::ocr::backends::OcrBackend;
use crate::ocr::error::{OcrError, OcrRequestError};
use crate::ocr::types::OcrConnection;
use crate::providers::mistral::auth;
use crate::url_utils::ApiUrl;

#[derive(Clone, Debug)]
pub struct MistralBackend;

impl OcrBackend for MistralBackend {
    type Config = ();
    const PROVIDER: crate::ocr::registry::OcrProvider = crate::ocr::registry::OcrProvider::Mistral;

    fn decode_config(_params: &Map<String, Value>) -> Result<Self::Config, Error> {
        Ok(())
    }
}

pub(crate) fn complete_url(api_base: Option<&str>) -> Result<String, OcrError> {
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

pub(crate) fn authenticate(
    connection: &OcrConnection,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, OcrError> {
    Ok(auth::validate_environment(
        connection.extra_headers.clone(),
        connection.api_key.as_deref(),
        env_lookup,
    )?)
}
