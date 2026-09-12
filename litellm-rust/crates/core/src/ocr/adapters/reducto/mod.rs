mod legacy;
mod v3;

use crate::Error;
use crate::constants::{REDUCTO_API_BASE, REDUCTO_API_KEY_ENV, REDUCTO_ID_PREFIX};
use crate::ocr::document::InlineDocument;
use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
use crate::ocr::types::{OcrConnection, OcrDocument};
use crate::url_utils::ApiUrl;

pub(crate) use legacy::ReductoLegacyAdapter;
pub(crate) use v3::ReductoV3Adapter;

pub(super) fn get_complete_url(api_base: Option<&str>, path: &str) -> Result<String, OcrError> {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(REDUCTO_API_BASE);
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

pub(super) fn validate_environment(
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
        .or_else(|| {
            env_lookup(REDUCTO_API_KEY_ENV)
                .map(|key| key.trim().to_string())
                .filter(|key| !key.is_empty())
        })
        .ok_or(Error::MissingReductoApiKey)?;
    Ok(
        std::iter::once(("Authorization".into(), format!("Bearer {api_key}")))
            .chain(connection.extra_headers.clone())
            .collect(),
    )
}

pub(super) async fn prepare_document(
    client: &crate::ocr::OcrClient,
    document: OcrDocument,
    connection: &OcrConnection,
    headers: &[(String, String)],
) -> Result<OcrDocument, OcrError> {
    if document.source().starts_with(REDUCTO_ID_PREFIX) {
        if document.source()[REDUCTO_ID_PREFIX.len()..]
            .trim()
            .is_empty()
        {
            return Err(OcrRequestError::RequestField {
                path: "document file id".into(),
            }
            .into());
        }
        return Ok(document);
    }
    let inline = InlineDocument::parse(document.source())?.ok_or(OcrRequestError::ReductoSource)?;
    let mime = inline.mime_type().to_string();
    let bytes = inline.decode(crate::constants::OCR_INLINE_MAX_BYTES)?;
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name("document")
        .mime_str(&mime)
        .map_err(|_| OcrRequestError::InvalidDataUri)?;
    let builder = client
        .provider_http()
        .post(get_complete_url(connection.api_base.as_deref(), "upload")?)
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
    let uploaded = crate::ocr::client::read_json_response::<
        crate::ocr::codecs::reducto::ReductoUploadResponse,
    >(response, false)
    .await?
    .data;
    let file_id = uploaded
        .file_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty());
    let Some(file_id) = file_id else {
        return Err(OcrResponseError::ResponseField {
            path: "file_id".into(),
        }
        .into());
    };
    Ok(document.with_source(file_id.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_key_precedes_environment_key() {
        let connection = OcrConnection {
            api_key: Some("passed-key".into()),
            ..Default::default()
        };
        let headers = validate_environment(&connection, &|_| Some("env-key".into())).unwrap();
        assert_eq!(headers[0].1, "Bearer passed-key");
    }

    #[test]
    fn blank_explicit_key_uses_environment_key() {
        let connection = OcrConnection {
            api_key: Some(" ".into()),
            ..Default::default()
        };
        let headers = validate_environment(&connection, &|_| Some(" env-key ".into())).unwrap();
        assert_eq!(headers[0].1, "Bearer env-key");
    }

    #[test]
    fn existing_authorization_skips_key_lookup() {
        let connection = OcrConnection {
            extra_headers: vec![("authorization".into(), "Bearer existing".into())],
            ..Default::default()
        };
        assert_eq!(
            validate_environment(&connection, &|_| None).unwrap(),
            connection.extra_headers
        );
    }
}
