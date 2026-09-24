use std::{collections::BTreeMap as Map, io::Read, path::Path};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_llms::base_llm::ocr::{
    error::Error,
    transformation::{OCR_INLINE_MAX_BYTES, OcrDocument},
};

use crate::ocr::types::OcrDocumentInput;

pub fn prepare_document(input: OcrDocumentInput) -> Result<OcrDocument, Error> {
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
        OcrDocumentInput::HostReader { .. } => Err(Error::InvalidRequest(
            "OCR file reader was not read by the host".into(),
        )),
    }
}

pub fn read_path_document(path: &Path, mime_type: Option<&str>) -> Result<OcrDocument, Error> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .and_then(|file| {
            file.take(OCR_INLINE_MAX_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
        })
        .map_err(|source| Error::FileRead {
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
) -> Result<OcrDocument, Error> {
    if bytes.is_empty() {
        return Err(Error::EmptyFile);
    }
    if bytes.len() > OCR_INLINE_MAX_BYTES {
        return Err(Error::InlineDocumentTooLarge);
    }
    if let Some(value) = mime_type
        && !valid_mime_type(value)
    {
        return Err(Error::InvalidMimeType(value.into()));
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap as Map;

    use litellm_llms::base_llm::ocr::document::InlineDocument;

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
            Err(Error::InlineDocumentTooLarge)
        ));
        std::fs::remove_dir_all(&dir).unwrap();

        let missing = dir.join("missing.pdf");
        let Err(super::Error::FileRead { path, source, .. }) =
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
            Err(Error::InlineDocumentTooLarge)
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
}

#[cfg(test)]
mod document_tests {
    use litellm_host::event::WireRequest;
    use litellm_llms::base_llm::ocr::error::Error;
    use rstest::rstest;
    use serde_json::{Value, json};

    use crate::ocr::route::LocalOcrHost;
    use crate::ocr::test_support::{
        MockResponse, SERVED_DOCUMENT, document_server, mock_server, perform_ocr_with,
        request_body, wire_request_with_document,
    };

    #[derive(Clone, Copy, Debug)]
    enum Route {
        Mistral,
        AzureAi,
        VertexMistral,
        AzureCohereParse,
        Cohere,
    }

    impl Route {
        fn model(self) -> &'static str {
            match self {
                Self::Mistral => "mistral/model",
                Self::AzureAi => "azure_ai/model",
                Self::VertexMistral => "vertex_ai/mistral-ocr-maas",
                Self::AzureCohereParse => "azure_ai/cohere-parse",
                Self::Cohere => "cohere/model",
            }
        }

        fn document_type(self) -> &'static str {
            match self {
                Self::Mistral | Self::AzureAi | Self::VertexMistral => "document_url",
                Self::AzureCohereParse | Self::Cohere => "image_url",
            }
        }

        fn options(self) -> Value {
            match self {
                Self::Mistral | Self::AzureAi => json!({"pages": [0]}),
                Self::VertexMistral => json!({"pages": [0], "vertex_project": "project-1"}),
                Self::AzureCohereParse | Self::Cohere => json!({"output_format": "markdown"}),
            }
        }
    }

    /// What the host does to the wire request in `before_send`.
    #[derive(Clone, Copy, Debug)]
    enum Host {
        Detached,
        ReplacesDocument,
    }

    const REPLACED_DOCUMENT: &str = "data:image/png;base64,cmVwbGFjZWQ=";

    impl Host {
        fn before_send(self, wire: WireRequest) -> WireRequest {
            let Value::Object(fields) = wire.body else {
                return wire;
            };
            let body = fields
                .into_iter()
                .map(|(name, value)| match self {
                    Self::Detached => (name, value),
                    Self::ReplacesDocument if name == "document" => {
                        let document_type = value["type"].clone();
                        let key = document_type.as_str().unwrap_or_default().to_string();
                        (name, json!({"type": document_type, key: REPLACED_DOCUMENT}))
                    }
                    Self::ReplacesDocument => (name, value),
                })
                .collect();
            WireRequest {
                body: Value::Object(body),
                ..wire
            }
        }
    }

    struct Sent {
        result: Result<(), Error>,
        provider_body: Option<Value>,
    }

    async fn send(route: Route, host: Host, document_base: &str) -> Sent {
        let (base, seen, provider) =
            mock_server(vec![MockResponse::json(json!({"pages": []}))]).await;
        let document_type = route.document_type();
        let document =
            json!({"type": document_type, document_type: format!("{document_base}/scan.png")});
        let request = wire_request_with_document(route.model(), &base, document, route.options());
        let local =
            LocalOcrHost::new(request).with_before_send(move |wire, _| Ok(host.before_send(wire)));
        let result = perform_ocr_with(local).await.map(|_| ());
        match result {
            Ok(()) => provider.await.unwrap(),
            Err(_) => provider.abort(),
        }
        let provider_body = seen
            .lock()
            .unwrap()
            .first()
            .map(|request| request_body(request));
        Sent {
            result,
            provider_body,
        }
    }

    fn served_document_uri() -> String {
        use base64::Engine;
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(SERVED_DOCUMENT)
        )
    }

    #[rstest]
    #[case::azure_ai(Route::AzureAi)]
    #[case::vertex_mistral(Route::VertexMistral)]
    #[case::azure_cohere_parse(Route::AzureCohereParse)]
    #[tokio::test]
    async fn inlining_routes_send_the_downloaded_document(#[case] route: Route) {
        let (document_base, _documents) = document_server().await;
        let sent = send(route, Host::Detached, &document_base).await;
        sent.result.unwrap();
        assert_eq!(
            sent.provider_body.unwrap()["document"][route.document_type()],
            json!(served_document_uri())
        );
    }

    #[rstest]
    #[tokio::test]
    async fn document_replaced_by_the_host_reaches_the_provider(
        #[values(
            Route::Mistral,
            Route::AzureAi,
            Route::VertexMistral,
            Route::AzureCohereParse,
            Route::Cohere
        )]
        route: Route,
    ) {
        let (document_base, _documents) = document_server().await;
        let sent = send(route, Host::ReplacesDocument, &document_base).await;
        sent.result.unwrap();
        assert_eq!(
            sent.provider_body.unwrap()["document"][route.document_type()],
            json!(REPLACED_DOCUMENT)
        );
    }
}
