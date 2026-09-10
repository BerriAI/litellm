use base64::{Engine, engine::general_purpose::STANDARD};
use data_url::{DataUrl, DataUrlError, forgiving_base64::DecodeError, mime::Mime};
use reqwest::Url;

use super::error::{OcrError, OcrRequestError, OcrResponseError};
use super::types::{OcrConnection, OcrDocument};
use crate::constants::OCR_MAX_FETCH_REDIRECTS;
use crate::error::{MediaError, TransportError};
use crate::media::{DownloadPolicy, MediaFetcher};

pub trait DocumentPreparation: Send + Sync + 'static {
    type Output: Send;

    fn prepare(
        client: &super::OcrClient,
        document: OcrDocument,
        connection: &OcrConnection,
        headers: &[(String, String)],
    ) -> impl std::future::Future<Output = Result<Self::Output, OcrError>> + Send;
}

pub struct PassThrough;

impl DocumentPreparation for PassThrough {
    type Output = OcrDocument;

    async fn prepare(
        _client: &super::OcrClient,
        document: OcrDocument,
        _connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<Self::Output, OcrError> {
        Ok(document)
    }
}

pub struct RequireInline;

impl DocumentPreparation for RequireInline {
    type Output = InlineOcrDocument;

    async fn prepare(
        client: &super::OcrClient,
        document: OcrDocument,
        connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<Self::Output, OcrError> {
        inline_remote_document(client.document_fetcher(), document, connection).await
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

#[derive(Debug)]
pub struct InlineOcrDocument(OcrDocument);

impl TryFrom<OcrDocument> for InlineOcrDocument {
    type Error = OcrRequestError;

    fn try_from(document: OcrDocument) -> Result<Self, Self::Error> {
        let inline =
            InlineDocument::parse(document.source())?.ok_or(OcrRequestError::InvalidDataUri)?;
        inline.decode(crate::constants::OCR_INLINE_MAX_BYTES)?;
        Ok(Self(document))
    }
}

impl From<InlineOcrDocument> for OcrDocument {
    fn from(document: InlineOcrDocument) -> Self {
        document.0
    }
}

pub(crate) async fn inline_remote_document(
    fetcher: &MediaFetcher,
    document: OcrDocument,
    connection: &OcrConnection,
) -> Result<InlineOcrDocument, OcrError> {
    let source = document.source();
    if !source.starts_with("http://") && !source.starts_with("https://") {
        return Ok(InlineOcrDocument::try_from(document)?);
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
    Ok(InlineOcrDocument::try_from(document.with_source(format!(
        "data:{};base64,{}",
        downloaded.content_type,
        STANDARD.encode(downloaded.bytes)
    )))?)
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
    use super::{InlineDocument, inline_remote_document};
    use crate::ocr::OcrClient;
    use crate::ocr::error::OcrRequestError;
    use crate::ocr::types::{OcrConnection, OcrDocument};
    use rstest::rstest;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[rstest]
    #[case("data:application/pdf;base64,YWJj", b"abc")]
    #[case("DATA:application/pdf;BASE64,YWI", b"ab")]
    #[case("data:application/pdf;base64,YQ%3D%3D#fragment", b"a")]
    #[case("data:application/pdf;base64,Y W J j", b"abc")]
    #[case("data:,a%20b%00%FF", b"a b\0\xff")]
    #[case("data:;base64,", b"")]
    fn decodes_data_urls(#[case] source: &str, #[case] expected: &[u8]) {
        let document = InlineDocument::parse(source).unwrap().unwrap();
        assert_eq!(document.decode(expected.len()).unwrap(), expected);
        if !expected.is_empty() {
            assert_eq!(
                document.decode(expected.len() - 1),
                Err(OcrRequestError::InlineDocumentTooLarge)
            );
        }
    }

    #[rstest]
    #[case("https://example.com/document.pdf")]
    #[case("reducto://document")]
    #[case("data:application/pdf;base64,INVALID!")]
    #[case("")]
    fn inline_document_rejects_unprepared_sources(#[case] source: &str) {
        let document = OcrDocument::DocumentUrl {
            document_url: source.into(),
        };
        assert!(super::InlineOcrDocument::try_from(document).is_err());
    }

    #[test]
    fn inline_document_preserves_valid_image_source() {
        let document = OcrDocument::ImageUrl {
            image_url: "data:image/png;base64,YWJj".into(),
        };
        let prepared = super::InlineOcrDocument::try_from(document.clone()).unwrap();
        assert_eq!(OcrDocument::from(prepared), document);
    }

    #[test]
    fn preserves_mime_parameters_and_uses_standard_default() {
        let document = InlineDocument::parse("data:application/pdf;version=1.7;base64,YQ==")
            .unwrap()
            .unwrap();
        assert!(document.mime_type().matches("application", "pdf"));
        assert_eq!(document.mime_type().get_parameter("version"), Some("1.7"));
        let default = InlineDocument::parse("data:,a").unwrap().unwrap();
        assert!(default.mime_type().matches("text", "plain"));
        assert_eq!(
            default.mime_type().get_parameter("charset"),
            Some("US-ASCII")
        );
    }

    #[rstest]
    #[case("data:application/pdf;base64")]
    #[case("data:application/pdf;base64,INVALID!")]
    #[case("data:application/pdf;base64,Y")]
    fn rejects_invalid_input(#[case] source: &str) {
        let result =
            InlineDocument::parse(source).and_then(|document| document.unwrap().decode(100));
        assert_eq!(result, Err(OcrRequestError::InvalidDataUri));
    }

    #[test]
    fn leaves_other_sources_to_the_provider() {
        for source in ["https://example.com/file.pdf", "reducto://file.pdf"] {
            assert!(InlineDocument::parse(source).unwrap().is_none());
        }
    }

    #[test]
    fn limits_total_decoded_bytes_across_chunks() {
        use base64::{Engine, engine::general_purpose::STANDARD};

        let bytes = vec![0xff; 8193];
        let source = format!("data:application/pdf;base64,{}", STANDARD.encode(&bytes));
        let document = InlineDocument::parse(&source).unwrap().unwrap();
        assert_eq!(document.decode(bytes.len()).unwrap(), bytes);
        assert_eq!(
            document.decode(bytes.len() - 1),
            Err(OcrRequestError::InlineDocumentTooLarge)
        );
    }

    #[tokio::test]
    async fn provider_credentials_never_reach_remote_document() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let address = listener.local_addr().expect("listener has address");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts request");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let bytes_read = socket.read(&mut buffer).await.expect("reads request");
                assert!(bytes_read > 0);
                request.extend_from_slice(&buffer[..bytes_read]);
            }
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/pdf\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc",
                )
                .await
                .expect("writes response");
            String::from_utf8(request).expect("request is UTF-8")
        });
        let mut provider_headers = reqwest::header::HeaderMap::new();
        provider_headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_static("Bearer provider-secret"),
        );
        let provider_http = reqwest::Client::builder()
            .default_headers(provider_headers)
            .build()
            .expect("provider client builds");
        let document_http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("document client builds");
        let client = OcrClient::for_test(provider_http, document_http);
        let document = OcrDocument::DocumentUrl {
            document_url: format!("http://{address}/document.pdf"),
        };

        let prepared = inline_remote_document(
            client.document_fetcher(),
            document,
            &OcrConnection::default(),
        )
        .await
        .expect("document is downloaded");
        let request = server.await.expect("server completes");

        assert_eq!(
            OcrDocument::from(prepared).source(),
            "data:application/pdf;base64,YWJj"
        );
        assert!(!request.to_ascii_lowercase().contains("authorization"));
        assert!(!request.contains("provider-secret"));
    }
}
