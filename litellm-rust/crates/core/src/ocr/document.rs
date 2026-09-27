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
    fn byte_documents_are_encoded() {
        assert_eq!(
            prepare_document(OcrDocumentInput::Bytes {
                bytes: b"abc".as_slice().into(),
                file_name: Some("scan.pdf".into()),
                mime_type: None,
            })
            .unwrap(),
            document("data:application/pdf;base64,YWJj")
        );
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
