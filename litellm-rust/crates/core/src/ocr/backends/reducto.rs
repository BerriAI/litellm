use crate::auth::AuthError;
use crate::constants::{REDUCTO_ID_PREFIX, REDUCTO_OCR_API_BASE};
use crate::ocr::backends::OcrBackend;
use crate::ocr::document::InlineDocument;
use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
use crate::ocr::formats::reducto::{
    ReductoParseLegacyFormat, ReductoParseV3Format,
    types::{ReductoFileId, ReductoLegacyParams, ReductoUploadResponse, ReductoV3Params},
};
use crate::ocr::types::{OcrConnection, OcrDocument};
use crate::providers::reducto::auth;

pub struct ReductoOcrBackend;

pub fn normalize_api_base(api_base: Option<&str>) -> &str {
    api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(REDUCTO_OCR_API_BASE)
        .trim_end_matches('/')
}

async fn prepare_reducto_document(
    http_client: &reqwest::Client,
    document: OcrDocument,
    connection: &OcrConnection,
    headers: &[(String, String)],
) -> Result<ReductoFileId, OcrError> {
    let source = document.source();
    if source.starts_with(REDUCTO_ID_PREFIX) {
        return Ok(ReductoFileId(source.to_string()));
    }
    let document = InlineDocument::parse(source)?.ok_or(OcrRequestError::ReductoSource)?;
    let mime = document.mime_type().to_string();
    let bytes = document.decode(crate::constants::OCR_INLINE_MAX_BYTES)?;
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name("document")
        .mime_str(&mime)
        .map_err(|_| OcrRequestError::InvalidDataUri)?;
    let mut builder = http_client
        .post(format!(
            "{}/upload",
            normalize_api_base(connection.api_base.as_deref())
        ))
        .multipart(reqwest::multipart::Form::new().part("file", part))
        .timeout(connection.timeout);
    for (name, value) in headers {
        if !name.eq_ignore_ascii_case("content-type")
            && !name.eq_ignore_ascii_case("content-length")
        {
            builder = builder.header(name, value);
        }
    }
    let response = crate::http_utils::http_request(builder)
        .await
        .map_err(crate::ocr::client::network_error)?;
    let response = crate::ocr::wire::read_json_response::<ReductoUploadResponse>(response, false)
        .await?
        .data;
    if response.file_id.is_empty() {
        return Err(OcrResponseError::ResponseField {
            path: "file_id".into(),
        }
        .into());
    }
    Ok(ReductoFileId(response.file_id))
}

macro_rules! impl_reducto_backend {
    ($format:ty, $params:ty) => {
        impl OcrBackend<$format> for ReductoOcrBackend {
            fn complete_url(
                &self,
                connection: &OcrConnection,
                _model: &str,
                _params: &$params,
            ) -> Result<String, OcrError> {
                Ok(format!(
                    "{}/parse",
                    normalize_api_base(connection.api_base.as_deref())
                ))
            }

            async fn prepare_document(
                &self,
                http_client: &reqwest::Client,
                document: OcrDocument,
                connection: &OcrConnection,
                headers: &[(String, String)],
            ) -> Result<ReductoFileId, OcrError> {
                prepare_reducto_document(http_client, document, connection, headers).await
            }

            fn guard_document_before_preparation(&self) -> bool {
                true
            }

            fn preserve_native_response(&self, _params: &$params) -> bool {
                true
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
    };
}

impl_reducto_backend!(ReductoParseV3Format, ReductoV3Params);
impl_reducto_backend!(ReductoParseLegacyFormat, ReductoLegacyParams);
