use base64::{Engine, engine::general_purpose::STANDARD};
use data_url::mime::Mime;
use data_url::{DataUrl, DataUrlError, forgiving_base64::DecodeError};
use reqwest::Url;
use serde_json::{Map, Value};

use super::error::{OcrError, OcrRequestError, OcrResponseError};
use super::types::{OcrConnection, OcrDocument};
use crate::constants::OCR_MAX_FETCH_REDIRECTS;
use crate::error::{MediaError, TransportError};
use crate::media::{DownloadPolicy, MediaFetcher};

pub fn encode_file_document(
    bytes: &[u8],
    file_name: Option<&str>,
    mime_type: Option<&str>,
) -> Result<Value, OcrRequestError> {
    if bytes.is_empty() {
        return Err(OcrRequestError::RequestField {
            path: "document.file".into(),
        });
    }
    let mime_type = mime_type.map(str::trim);
    if mime_type.is_some_and(|value| !valid_mime_type(value)) {
        return Err(OcrRequestError::RequestField {
            path: "document.mime_type".into(),
        });
    }
    let mime_type = mime_type
        .map(str::to_string)
        .or_else(|| file_name.and_then(mime_type_for_name).map(str::to_string))
        .unwrap_or_else(|| "application/octet-stream".into());
    let source = format!("data:{mime_type};base64,{}", STANDARD.encode(bytes));
    let (kind, field) = if mime_type.starts_with("image/") {
        ("image_url", "image_url")
    } else {
        ("document_url", "document_url")
    };
    Ok(Value::Object(Map::from_iter([
        ("type".into(), Value::String(kind.into())),
        (field.into(), Value::String(source)),
    ])))
}

fn valid_mime_type(value: &str) -> bool {
    let Some((kind, subtype)) = value.split_once('/') else {
        return false;
    };
    !kind.is_empty()
        && !subtype.is_empty()
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'.' | b'+' | b'-' | b'_')
        })
}

fn mime_type_for_name(name: &str) -> Option<&'static str> {
    let extension = name.rsplit_once('.')?.1;
    match extension.to_ascii_lowercase().as_str() {
        "pdf" => Some("application/pdf"),
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "tiff" | "tif" => Some("image/tiff"),
        "bmp" => Some("image/bmp"),
        _ => None,
    }
}

pub(crate) struct InlineDocument<'a>(DataUrl<'a>);

impl<'a> InlineDocument<'a> {
    pub(crate) fn parse(source: &'a str) -> Result<Option<Self>, OcrRequestError> {
        match DataUrl::process(source) {
            Ok(url) => Ok(Some(Self(url))),
            Err(DataUrlError::NotADataUrl) => Ok(None),
            Err(DataUrlError::NoComma) => Err(OcrRequestError::InvalidDataUri),
        }
    }

    pub(crate) fn mime_type(&self) -> &Mime {
        self.0.mime_type()
    }

    pub(crate) fn decode(&self, max_bytes: usize) -> Result<Vec<u8>, OcrRequestError> {
        let mut body = Vec::new();
        self.0
            .decode(|bytes| {
                if bytes.len() > max_bytes.saturating_sub(body.len()) {
                    return Err(OcrRequestError::InlineDocumentTooLarge);
                }
                body.extend_from_slice(bytes);
                Ok(())
            })
            .map_err(|error| match error {
                DecodeError::InvalidBase64(_) => OcrRequestError::InvalidDataUri,
                DecodeError::WriteError(error) => error,
            })?;
        Ok(body)
    }
}

pub(crate) fn validate_inline_document(document: &OcrDocument) -> Result<(), OcrRequestError> {
    let inline =
        InlineDocument::parse(document.source())?.ok_or(OcrRequestError::InvalidDataUri)?;
    inline.decode(crate::constants::OCR_INLINE_MAX_BYTES)?;
    Ok(())
}

pub(crate) async fn inline_remote_document(
    fetcher: &MediaFetcher,
    document: OcrDocument,
    connection: &OcrConnection,
) -> Result<OcrDocument, OcrError> {
    let source = document.source();
    if !source.starts_with("http://") && !source.starts_with("https://") {
        validate_inline_document(&document)?;
        return Ok(document);
    }
    let url = Url::parse(source).map_err(|_| OcrRequestError::RequestField {
        path: "document URL".into(),
    })?;
    let downloaded = fetcher
        .fetch(
            url,
            DownloadPolicy {
                timeout: connection.timeout,
                max_bytes: connection.max_download_bytes,
                max_redirects: OCR_MAX_FETCH_REDIRECTS,
            },
        )
        .await
        .map_err(map_media_error)?;
    let result = document.with_source(format!(
        "data:{};base64,{}",
        downloaded.content_type,
        STANDARD.encode(downloaded.bytes)
    ));
    validate_inline_document(&result)?;
    Ok(result)
}

fn map_media_error(error: MediaError) -> OcrError {
    match error {
        MediaError::BlockedUrl => OcrRequestError::BlockedDocumentUrl.into(),
        MediaError::DownloadDisabled => OcrRequestError::DownloadDisabled.into(),
        MediaError::DownloadTooLarge => OcrRequestError::DownloadTooLarge.into(),
        MediaError::TooManyRedirects => OcrRequestError::TooManyRedirects.into(),
        MediaError::MissingRedirectLocation => OcrResponseError::MissingRedirectLocation.into(),
        MediaError::InvalidRedirect => OcrResponseError::InvalidRedirect.into(),
        MediaError::Http(status) => TransportError::Http {
            status,
            body: "OCR document download failed".into(),
        }
        .into(),
        MediaError::Timeout => {
            TransportError::Network("OCR document download timed out".into()).into()
        }
        MediaError::Transport(error) => error.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;

    fn document(source: &str) -> OcrDocument {
        OcrDocument::DocumentUrl {
            document_url: source.into(),
            extra_fields: Map::new(),
        }
    }

    #[test]
    fn file_bytes_are_encoded_with_core_owned_mime_policy() {
        assert_eq!(
            encode_file_document(b"abc", Some("scan.png"), None).unwrap(),
            serde_json::json!({
                "type": "image_url",
                "image_url": "data:image/png;base64,YWJj"
            })
        );
        assert_eq!(
            encode_file_document(b"abc", None, Some("application/pdf")).unwrap(),
            serde_json::json!({
                "type": "document_url",
                "document_url": "data:application/pdf;base64,YWJj"
            })
        );
    }

    #[test]
    fn file_encoding_rejects_empty_bytes_and_invalid_explicit_mime() {
        assert!(encode_file_document(b"", None, None).is_err());
        assert!(encode_file_document(b"abc", None, Some("text/plain;bad")).is_err());
    }

    #[test]
    fn decodes_data_urls_and_limits_decoded_size() {
        for (source, expected) in [
            ("data:application/pdf;base64,YWJj", b"abc".as_slice()),
            ("DATA:application/pdf;BASE64,YWI", b"ab".as_slice()),
            ("data:,a%20b%00%FF", b"a b\0\xff".as_slice()),
        ] {
            let inline = InlineDocument::parse(source).unwrap().unwrap();
            assert_eq!(inline.decode(expected.len()).unwrap(), expected);
            assert_eq!(
                inline.decode(expected.len() - 1),
                Err(OcrRequestError::InlineDocumentTooLarge)
            );
        }
    }

    #[test]
    fn preserves_mime_parameters_and_standard_default() {
        let inline = InlineDocument::parse("data:application/pdf;version=1.7;base64,YQ==")
            .unwrap()
            .unwrap();
        assert!(inline.mime_type().matches("application", "pdf"));
        assert_eq!(inline.mime_type().get_parameter("version"), Some("1.7"));
        let default = InlineDocument::parse("data:,a").unwrap().unwrap();
        assert!(default.mime_type().matches("text", "plain"));
        assert_eq!(
            default.mime_type().get_parameter("charset"),
            Some("US-ASCII")
        );
    }

    #[test]
    fn rejects_invalid_inline_documents() {
        for source in [
            "https://example.com/document.pdf",
            "data:application/pdf;base64",
            "data:application/pdf;base64,INVALID!",
        ] {
            assert!(validate_inline_document(&document(source)).is_err());
        }
    }

    #[tokio::test]
    async fn remote_conversion_preserves_kind_and_isolates_provider_credentials() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0_u8; 2048];
            let count = socket.read(&mut request).await.unwrap();
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: image/png; charset=binary\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc")
                .await
                .unwrap();
            String::from_utf8_lossy(&request[..count]).into_owned()
        });
        let mut provider_headers = reqwest::header::HeaderMap::new();
        provider_headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_static("Bearer provider-secret"),
        );
        let provider_http = reqwest::Client::builder()
            .default_headers(provider_headers)
            .build()
            .unwrap();
        let document_http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let client = super::super::OcrClient::for_test(provider_http, document_http);
        let converted = inline_remote_document(
            client.document_fetcher(),
            OcrDocument::ImageUrl {
                image_url: format!("http://{address}/image"),
                extra_fields: Map::from_iter([("detail".into(), serde_json::json!("high"))]),
            },
            &OcrConnection::default(),
        )
        .await
        .unwrap();
        let request = server.await.unwrap();

        assert_eq!(
            converted,
            OcrDocument::ImageUrl {
                image_url: "data:image/png;base64,YWJj".into(),
                extra_fields: Map::from_iter([("detail".into(), serde_json::json!("high"))]),
            }
        );
        assert!(!request.to_ascii_lowercase().contains("authorization"));
        assert!(!request.contains("provider-secret"));
    }
}
