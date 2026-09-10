use serde_json::{Map, Value};

use super::{OcrAdapter, PreparedAdapter};
use crate::Error;
use crate::constants::MISTRAL_OCR_API_BASE;
use crate::ocr::codecs::mistral::{self, MistralOcrParams, MistralOcrRequest, MistralOcrResponse};
use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
use crate::ocr::registry::OcrProvider;
use crate::ocr::types::{OcrConnection, OcrDocument, OcrResponseData};
use crate::url_utils::ApiUrl;

const MISTRAL_API_KEY_ENV: &str = "MISTRAL_API_KEY";
const MISSING_KEY_MESSAGE: &str = "Missing Mistral API Key - A call is being made to Mistral but no key is set either in the environment variables or via params";

#[derive(Clone, Debug)]
pub(crate) struct MistralAdapter;

impl OcrAdapter for MistralAdapter {
    type Config = ();
    type InputParams = MistralOcrParams;
    type Params = MistralOcrParams;
    type RequestBody = MistralOcrRequest;
    type ResponseBody = MistralOcrResponse;

    const PROVIDER: OcrProvider = OcrProvider::Mistral;

    fn decode_config(_params: &Map<String, Value>) -> Result<Self::Config, Error> {
        Ok(())
    }

    fn map_params(params: Self::InputParams) -> Result<Self::Params, OcrRequestError> {
        Ok(params)
    }

    fn build_request(
        model: &str,
        document: OcrDocument,
        params: &Self::Params,
    ) -> Result<Self::RequestBody, OcrRequestError> {
        mistral::encode_request(model, document, params)
    }

    fn normalize_response(
        model: &str,
        response: Self::ResponseBody,
        params: &Self::Params,
    ) -> Result<OcrResponseData, OcrResponseError> {
        mistral::decode_response(model, response, params)
    }

    async fn prepare(
        &self,
        connection: &OcrConnection,
        _config: &Self::Config,
        _model: &str,
        _params: &Self::Params,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedAdapter, OcrError> {
        Ok(PreparedAdapter {
            url: complete_url(connection.api_base.as_deref())?,
            headers: authenticate(connection, env_lookup)?,
        })
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

fn authenticate(
    connection: &OcrConnection,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, OcrError> {
    if crate::http_utils::has_header(&connection.extra_headers, "authorization") {
        return Ok(connection.extra_headers.clone());
    }
    let api_key = connection
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(MISTRAL_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
        .ok_or_else(|| Error::Auth(MISSING_KEY_MESSAGE.to_string()))?;
    Ok(
        std::iter::once(("Authorization".into(), format!("Bearer {api_key}")))
            .chain(connection.extra_headers.clone())
            .collect(),
    )
}
