use crate::Error;
use crate::constants::{REDUCTO_ID_PREFIX, REDUCTO_OCR_API_BASE};
use crate::ocr::backends::OcrBackend;
use crate::ocr::document::{DocumentPreparation, InlineDocument};
use crate::ocr::error::{OcrError, OcrRequestError};
use crate::ocr::formats::reducto::types::{ReductoFileId, ReductoUploadResponse};
use crate::ocr::types::{OcrConnection, OcrDocument};
use crate::providers::reducto::auth;
use crate::url_utils::ApiUrl;
use serde_json::{Map, Value};

#[derive(Clone, Debug)]
pub struct ReductoBackend;

impl OcrBackend for ReductoBackend {
    type Config = ();
    const PROVIDER: crate::ocr::registry::OcrProvider = crate::ocr::registry::OcrProvider::Reducto;

    fn decode_config(_params: &Map<String, Value>) -> Result<Self::Config, Error> {
        Ok(())
    }
}

pub(crate) fn resolve_integration(
    model: &crate::ocr::registry::OcrModel,
) -> crate::ocr::registry::OcrIntegrationKind {
    match model {
        crate::ocr::registry::OcrModel::ReductoLegacy => {
            crate::ocr::registry::OcrIntegrationKind::ReductoLegacy
        }
        _ => crate::ocr::registry::OcrIntegrationKind::ReductoV3,
    }
}

pub(crate) fn complete_url(api_base: Option<&str>, path: &str) -> Result<String, OcrError> {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(REDUCTO_OCR_API_BASE);
    ApiUrl::parse(base)
        .and_then(|url| url.complete_path(&[path]))
        .map(|url| url.into_string())
        .map_err(|_| {
            OcrRequestError::RequestField {
                path: "api_base".into(),
            }
            .into()
        })
}

pub struct ReductoUpload;

impl DocumentPreparation for ReductoUpload {
    type Output = ReductoFileId;

    async fn prepare(
        client: &crate::ocr::OcrClient,
        document: OcrDocument,
        connection: &OcrConnection,
        headers: &[(String, String)],
    ) -> Result<Self::Output, OcrError> {
        let source = document.source();
        if source.starts_with(REDUCTO_ID_PREFIX) {
            return Ok(ReductoFileId(source.to_string()));
        }
        let document = InlineDocument::parse(source)?.ok_or(OcrRequestError::ReductoSource)?;
        upload_document(client, document, connection, headers).await
    }
}

async fn upload_document(
    client: &crate::ocr::OcrClient,
    document: InlineDocument<'_>,
    connection: &OcrConnection,
    headers: &[(String, String)],
) -> Result<ReductoFileId, OcrError> {
    let mime = document.mime_type().to_string();
    let bytes = document.decode(crate::constants::OCR_INLINE_MAX_BYTES)?;
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name("document")
        .mime_str(&mime)
        .map_err(|_| OcrRequestError::InvalidDataUri)?;
    let builder = client
        .provider_http()
        .post(complete_url(connection.api_base.as_deref(), "upload")?)
        .multipart(reqwest::multipart::Form::new().part("file", part))
        .timeout(connection.timeout);
    let builder = crate::http_utils::with_headers(
        builder,
        headers,
        crate::http_utils::HeaderPolicy::Except(&["content-type", "content-length"]),
    );
    let response = crate::http_utils::http_request(builder)
        .await
        .map_err(crate::error::TransportError::from)?;
    let response = crate::ocr::client::read_json_response::<ReductoUploadResponse>(response, false)
        .await?
        .data;
    response.try_into()
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
