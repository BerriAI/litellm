use std::collections::BTreeMap as Map;
use std::io::Read;
use std::path::Path;

use base64::{Engine, engine::general_purpose::STANDARD};
use data_url::mime::Mime;
use data_url::{DataUrl, DataUrlError, forgiving_base64::DecodeError};
use reqwest::Url;

use super::Error as OcrError;
use super::Error as OcrRequestError;
use super::Error as OcrResponseError;
use super::types::{OcrConnection, OcrDocument, OcrDocumentInput};
use crate::constants::{OCR_INLINE_MAX_BYTES, OCR_MAX_FETCH_REDIRECTS};
use crate::media::Error as MediaError;
use crate::media::{DownloadPolicy, MediaFetcher};
use crate::transport::Error as TransportError;

pub fn prepare_document(input: OcrDocumentInput) -> Result<OcrDocument, super::Error> {
    match input {
        OcrDocumentInput::Document(document) => Ok(document),
        OcrDocumentInput::Path { path, mime_type } => {
            read_path_document(&path, mime_type.as_deref())
        }
        OcrDocumentInput::Bytes {
            bytes,
            file_name,
            mime_type,
        } => Ok(encode_file_document(
            &bytes,
            file_name.as_deref(),
            mime_type.as_deref(),
        )?),
        OcrDocumentInput::HostReader { .. } => Err(super::Error::InvalidRequest(
            "OCR file reader was not read by the host".into(),
        )),
    }
}

pub fn read_path_document(
    path: &Path,
    mime_type: Option<&str>,
) -> Result<OcrDocument, super::Error> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .and_then(|file| {
            file.take(OCR_INLINE_MAX_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
        })
        .map_err(|source| super::Error::FileRead {
            path: path.to_owned(),
            source: std::sync::Arc::new(source),
        })?;
    let name = path.file_name().map(|name| name.to_string_lossy());
    encode_file_document(&bytes, name.as_deref(), mime_type)
}

pub fn encode_file_document(
    bytes: &[u8],
    file_name: Option<&str>,
    mime_type: Option<&str>,
) -> Result<OcrDocument, OcrRequestError> {
    if bytes.is_empty() {
        return Err(OcrRequestError::EmptyFile);
    }
    if bytes.len() > OCR_INLINE_MAX_BYTES {
        return Err(OcrRequestError::InlineDocumentTooLarge);
    }
    if let Some(value) = mime_type
        && !valid_mime_type(value)
    {
        return Err(OcrRequestError::InvalidMimeType(value.into()));
    }
    let mime_type = mime_type
        .map(str::to_string)
        .or_else(|| file_name.map(|name| mime_type_for_name(name).to_string()))
        .unwrap_or_else(|| "application/octet-stream".into());
    let source = format!("data:{mime_type};base64,{}", STANDARD.encode(bytes));
    Ok(if mime_type.starts_with("image/") {
        OcrDocument::ImageUrl {
            image_url: source,
            extra_fields: Map::new(),
        }
    } else {
        OcrDocument::DocumentUrl {
            document_url: source,
            extra_fields: Map::new(),
        }
    })
}

fn valid_mime_type(value: &str) -> bool {
    let Some((kind, subtype)) = value.split_once('/') else {
        return false;
    };
    !kind.is_empty()
        && !subtype.is_empty()
        && kind.chars().chain(subtype.chars()).all(|character| {
            character.is_alphanumeric() || matches!(character, '.' | '+' | '-' | '_')
        })
}

pub fn mime_type_for_name(name: &str) -> &'static str {
    let extension = std::path::Path::new(name)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    match extension.to_ascii_lowercase().as_str() {
        "pdf" => "application/pdf",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "tiff" | "tif" => "image/tiff",
        "bmp" => "image/bmp",
        _ => mime_guess::from_path(name)
            .first_raw()
            .unwrap_or("application/octet-stream"),
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
    if !document.is_remote() {
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
        MediaError::BlockedUrl => OcrRequestError::BlockedDocumentUrl,
        MediaError::DownloadDisabled => OcrRequestError::DownloadDisabled,
        MediaError::DownloadTooLarge => OcrRequestError::DownloadTooLarge,
        MediaError::TooManyRedirects => OcrRequestError::TooManyRedirects,
        MediaError::MissingRedirectLocation => OcrResponseError::MissingRedirectLocation,
        MediaError::InvalidRedirect => OcrResponseError::InvalidRedirect,
        MediaError::Http(status) => TransportError::Http {
            status,
            body: "OCR document download failed".into(),
        }
        .into(),
        MediaError::Timeout => TransportError::Http {
            status: 408,
            body: "OCR document download timed out".into(),
        }
        .into(),
        MediaError::Transport(error) => error.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap as Map;

    use super::*;

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
            OcrDocument::ImageUrl {
                image_url: "data:image/png;base64,YWJj".into(),
                extra_fields: Map::new(),
            }
        );
        assert_eq!(
            encode_file_document(b"abc", None, Some("application/pdf")).unwrap(),
            document("data:application/pdf;base64,YWJj")
        );
    }

    #[test]
    fn file_name_mime_mapping_matches_python() {
        for (name, expected) in [
            ("document.pdf", "application/pdf"),
            ("image.png", "image/png"),
            ("photo.jpg", "image/jpeg"),
            ("photo.jpeg", "image/jpeg"),
            ("animation.gif", "image/gif"),
            ("image.webp", "image/webp"),
            ("scan.tiff", "image/tiff"),
            ("scan.tif", "image/tiff"),
            ("bitmap.bmp", "image/bmp"),
            ("DOCUMENT.PDF", "application/pdf"),
            ("IMAGE.PNG", "image/png"),
            ("file.unknown-extension", "application/octet-stream"),
        ] {
            assert_eq!(mime_type_for_name(name), expected);
        }
    }

    #[test]
    fn path_documents_are_read_and_named_by_core() {
        let dir = std::env::temp_dir().join(format!("litellm-ocr-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scan.png");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            prepare_document(OcrDocumentInput::Path {
                path: path.clone(),
                mime_type: None,
            })
            .unwrap(),
            OcrDocument::ImageUrl {
                image_url: "data:image/png;base64,YWJj".into(),
                extra_fields: Map::new(),
            }
        );
        assert_eq!(
            prepare_document(OcrDocumentInput::Path {
                path: path.clone(),
                mime_type: Some("application/pdf".into()),
            })
            .unwrap(),
            document("data:application/pdf;base64,YWJj")
        );
        std::fs::write(&path, vec![b'a'; OCR_INLINE_MAX_BYTES + 1]).unwrap();
        assert!(matches!(
            prepare_document(OcrDocumentInput::Path {
                path: path.clone(),
                mime_type: None,
            }),
            Err(OcrRequestError::InlineDocumentTooLarge)
        ));
        std::fs::remove_dir_all(&dir).unwrap();

        let missing = dir.join("missing.pdf");
        let Err(super::super::Error::FileRead { path, source, .. }) =
            prepare_document(OcrDocumentInput::Path {
                path: missing.clone(),
                mime_type: None,
            })
        else {
            panic!("missing paths must surface a file read error");
        };
        assert_eq!(path, missing);
        assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn byte_documents_are_encoded_and_host_readers_must_be_read_first() {
        assert_eq!(
            prepare_document(OcrDocumentInput::Bytes {
                bytes: b"abc".as_slice().into(),
                file_name: Some("scan.pdf".into()),
                mime_type: None,
            })
            .unwrap(),
            document("data:application/pdf;base64,YWJj")
        );
        assert!(prepare_document(OcrDocumentInput::HostReader { mime_type: None }).is_err());
    }

    #[test]
    fn file_encoding_enforces_decoded_size_limit() {
        let bytes = vec![b'a'; OCR_INLINE_MAX_BYTES + 1];
        assert!(matches!(
            encode_file_document(&bytes, None, None),
            Err(OcrRequestError::InlineDocumentTooLarge)
        ));
        let document = encode_file_document(&bytes[..OCR_INLINE_MAX_BYTES], None, None).unwrap();
        let inline = InlineDocument::parse(document.source()).unwrap().unwrap();
        assert_eq!(
            inline.decode(OCR_INLINE_MAX_BYTES).unwrap(),
            bytes[..OCR_INLINE_MAX_BYTES]
        );
    }

    #[test]
    fn file_encoding_rejects_empty_bytes_and_invalid_explicit_mime() {
        assert!(encode_file_document(b"", None, None).is_err());
        for mime in [
            "text/plain;bad",
            "text/plain/extra",
            " text/plain",
            "text/plain\n",
        ] {
            assert!(encode_file_document(b"abc", None, Some(mime)).is_err());
        }
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
            assert!(matches!(
                inline.decode(expected.len() - 1),
                Err(OcrRequestError::InlineDocumentTooLarge)
            ));
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
                extra_fields: Map::from_iter([("detail".into(), Some("high".into()))]),
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
                extra_fields: Map::from_iter([("detail".into(), Some("high".into()))]),
            }
        );
        assert!(!request.to_ascii_lowercase().contains("authorization"));
        assert!(!request.contains("provider-secret"));
    }
}
