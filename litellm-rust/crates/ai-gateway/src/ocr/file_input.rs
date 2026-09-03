use std::io::Read;
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use litellm_core::error::Error;
use serde_json::{Value, json};

use crate::constants::{OCR_DEFAULT_MIME_TYPE, OCR_MIME_TYPES_BY_EXTENSION};

pub enum FileInput {
    Path(PathBuf),
    Bytes(Vec<u8>),
    Reader {
        name: Option<String>,
        reader: Box<dyn Read>,
    },
    Str(String),
    Unsupported(String),
}

pub fn get_mime_type(file_path: &str) -> String {
    let extension = Path::new(file_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());
    extension
        .and_then(|ext| {
            OCR_MIME_TYPES_BY_EXTENSION
                .iter()
                .find(|(known, _)| *known == ext)
                .map(|(_, mime)| (*mime).to_string())
        })
        .unwrap_or_else(|| OCR_DEFAULT_MIME_TYPE.to_string())
}

fn is_valid_mime_type(mime_type: &str) -> bool {
    let Some((kind, subtype)) = mime_type.split_once('/') else {
        return false;
    };
    let is_token = |part: &str| {
        !part.is_empty()
            && part
                .chars()
                .all(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | '+' | '-'))
    };
    is_token(kind) && is_token(subtype)
}

fn read_file_input(file: FileInput) -> Result<(Vec<u8>, String), Error> {
    match file {
        FileInput::Str(_) => Err(Error::InvalidRequest(
            "OCR file input does not accept bare str values. Pass bytes, a pathlib.Path, or a \
             file-like object. To OCR a local file from a path, call open(path, 'rb') yourself."
                .to_string(),
        )),
        FileInput::Path(path) => {
            if !path.is_file() {
                return Err(Error::InvalidRequest(format!(
                    "File not found: {}",
                    path.display()
                )));
            }
            let bytes = std::fs::read(&path).map_err(|err| {
                Error::InvalidRequest(format!("File not found: {} ({err})", path.display()))
            })?;
            Ok((bytes, get_mime_type(&path.to_string_lossy())))
        }
        FileInput::Bytes(bytes) => Ok((bytes, OCR_DEFAULT_MIME_TYPE.to_string())),
        FileInput::Reader { name, mut reader } => {
            let mut bytes = Vec::new();
            reader
                .read_to_end(&mut bytes)
                .map_err(|err| Error::InvalidRequest(format!("File could not be read: {err}")))?;
            let mime_type = name
                .filter(|name| !name.is_empty())
                .map(|name| get_mime_type(&name))
                .unwrap_or_else(|| OCR_DEFAULT_MIME_TYPE.to_string());
            Ok((bytes, mime_type))
        }
        FileInput::Unsupported(type_name) => Err(Error::InvalidRequest(format!(
            "Unsupported file input type: {type_name}. Expected pathlib.Path, bytes, or a \
             file-like object."
        ))),
    }
}

fn data_uri_document(mime_type: &str, bytes: &[u8]) -> Value {
    let data_uri = format!("data:{mime_type};base64,{}", BASE64_STANDARD.encode(bytes));
    if mime_type.starts_with("image/") {
        return json!({"type": "image_url", "image_url": data_uri});
    }
    json!({"type": "document_url", "document_url": data_uri})
}

pub fn convert_file_document_to_url_document(
    file: Option<FileInput>,
    mime_type_override: Option<&str>,
) -> Result<Value, Error> {
    let file = file.ok_or_else(|| {
        Error::InvalidRequest(
            "document with type='file' must include a 'file' field containing a pathlib.Path, \
             file-like object, or bytes"
                .to_string(),
        )
    })?;
    let (bytes, detected_mime_type) = read_file_input(file)?;
    if bytes.is_empty() {
        return Err(Error::InvalidRequest(
            "File is empty or could not be read".to_string(),
        ));
    }
    let mime_type = mime_type_override.unwrap_or(&detected_mime_type);
    if !is_valid_mime_type(mime_type) {
        return Err(Error::InvalidRequest(format!(
            "Invalid MIME type: {mime_type}"
        )));
    }
    Ok(data_uri_document(mime_type, &bytes))
}

pub fn build_document_from_upload(
    file_content: Vec<u8>,
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<Value, Error> {
    let header_mime_type = content_type
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != OCR_DEFAULT_MIME_TYPE);
    let mime_type = header_mime_type
        .map(str::to_string)
        .or_else(|| filename.map(get_mime_type))
        .unwrap_or_else(|| OCR_DEFAULT_MIME_TYPE.to_string());
    convert_file_document_to_url_document(Some(FileInput::Bytes(file_content)), Some(&mime_type))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    struct TempFile(PathBuf);

    impl TempFile {
        fn with_suffix(suffix: &str, content: &[u8]) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "litellm-ocr-file-input-{}-{unique}{suffix}",
                std::process::id()
            ));
            std::fs::write(&path, content).unwrap();
            Self(path)
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn data_uri_payload<'a>(document: &'a Value, field: &str) -> &'a str {
        document[field].as_str().unwrap()
    }

    fn decode_base64(data_uri: &str) -> Vec<u8> {
        let (_, encoded) = data_uri.split_once(";base64,").unwrap();
        BASE64_STANDARD.decode(encoded).unwrap()
    }

    fn invalid_request_message(error: Error) -> String {
        match error {
            Error::InvalidRequest(message) => message,
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn mime_type_detects_pdf() {
        assert_eq!(get_mime_type("document.pdf"), "application/pdf");
    }

    #[test]
    fn mime_type_detects_png() {
        assert_eq!(get_mime_type("image.png"), "image/png");
    }

    #[test]
    fn mime_type_detects_jpg() {
        assert_eq!(get_mime_type("photo.jpg"), "image/jpeg");
    }

    #[test]
    fn mime_type_detects_jpeg() {
        assert_eq!(get_mime_type("photo.jpeg"), "image/jpeg");
    }

    #[test]
    fn mime_type_detects_gif() {
        assert_eq!(get_mime_type("animation.gif"), "image/gif");
    }

    #[test]
    fn mime_type_detects_webp() {
        assert_eq!(get_mime_type("image.webp"), "image/webp");
    }

    #[test]
    fn mime_type_detects_tiff() {
        assert_eq!(get_mime_type("scan.tiff"), "image/tiff");
    }

    #[test]
    fn mime_type_detects_tif() {
        assert_eq!(get_mime_type("scan.tif"), "image/tiff");
    }

    #[test]
    fn mime_type_detects_bmp() {
        assert_eq!(get_mime_type("bitmap.bmp"), "image/bmp");
    }

    #[test]
    fn mime_type_is_case_insensitive() {
        assert_eq!(get_mime_type("DOCUMENT.PDF"), "application/pdf");
        assert_eq!(get_mime_type("IMAGE.PNG"), "image/png");
    }

    #[test]
    fn mime_type_falls_back_for_unknown_extension() {
        assert_eq!(get_mime_type("file.xyz123"), "application/octet-stream");
    }

    #[test]
    fn convert_pdf_path_to_document_url() {
        let content = b"%PDF-1.4 test content";
        let file = TempFile::with_suffix(".pdf", content);

        let result =
            convert_file_document_to_url_document(Some(FileInput::Path(file.0.clone())), None)
                .unwrap();

        assert_eq!(result["type"], "document_url");
        let uri = data_uri_payload(&result, "document_url");
        assert!(uri.starts_with("data:application/pdf;base64,"));
        assert_eq!(decode_base64(uri), content);
    }

    #[test]
    fn convert_image_path_to_image_url() {
        let content = b"\x89PNG\r\n\x1a\n fake png content";
        let file = TempFile::with_suffix(".png", content);

        let result =
            convert_file_document_to_url_document(Some(FileInput::Path(file.0.clone())), None)
                .unwrap();

        assert_eq!(result["type"], "image_url");
        let uri = data_uri_payload(&result, "image_url");
        assert!(uri.starts_with("data:image/png;base64,"));
        assert_eq!(decode_base64(uri), content);
    }

    #[test]
    fn convert_rejects_bare_str_path() {
        let error = convert_file_document_to_url_document(
            Some(FileInput::Str("/etc/passwd".to_string())),
            None,
        )
        .unwrap_err();

        assert!(invalid_request_message(error).contains("does not accept bare str values"));
    }

    #[test]
    fn convert_pathbuf_matches_path_behavior() {
        let file = TempFile::with_suffix(".pdf", b"test pdf content");

        let result =
            convert_file_document_to_url_document(Some(FileInput::Path(file.0.clone())), None)
                .unwrap();

        assert_eq!(result["type"], "document_url");
        assert!(
            data_uri_payload(&result, "document_url").starts_with("data:application/pdf;base64,")
        );
    }

    #[test]
    fn convert_raw_bytes_uses_fallback_mime() {
        let content = b"raw bytes content";

        let result =
            convert_file_document_to_url_document(Some(FileInput::Bytes(content.to_vec())), None)
                .unwrap();

        assert_eq!(result["type"], "document_url");
        let uri = data_uri_payload(&result, "document_url");
        assert!(uri.starts_with("data:application/octet-stream;base64,"));
        assert_eq!(decode_base64(uri), content);
    }

    #[test]
    fn convert_raw_bytes_with_explicit_mime_type() {
        let result = convert_file_document_to_url_document(
            Some(FileInput::Bytes(b"raw pdf content".to_vec())),
            Some("application/pdf"),
        )
        .unwrap();

        assert_eq!(result["type"], "document_url");
        assert!(
            data_uri_payload(&result, "document_url").starts_with("data:application/pdf;base64,")
        );
    }

    #[test]
    fn convert_raw_bytes_with_image_mime_type() {
        let result = convert_file_document_to_url_document(
            Some(FileInput::Bytes(b"raw image content".to_vec())),
            Some("image/jpeg"),
        )
        .unwrap();

        assert_eq!(result["type"], "image_url");
        assert!(data_uri_payload(&result, "image_url").starts_with("data:image/jpeg;base64,"));
    }

    #[test]
    fn convert_reader_without_name() {
        let content = b"file-like content";

        let result = convert_file_document_to_url_document(
            Some(FileInput::Reader {
                name: None,
                reader: Box::new(Cursor::new(content.to_vec())),
            }),
            None,
        )
        .unwrap();

        assert_eq!(result["type"], "document_url");
        let uri = data_uri_payload(&result, "document_url");
        assert!(uri.contains("base64,"));
        assert_eq!(decode_base64(uri), content);
    }

    #[test]
    fn convert_reader_with_name_detects_mime() {
        let result = convert_file_document_to_url_document(
            Some(FileInput::Reader {
                name: Some("test_image.png".to_string()),
                reader: Box::new(Cursor::new(b"file-like png content".to_vec())),
            }),
            None,
        )
        .unwrap();

        assert_eq!(result["type"], "image_url");
        assert!(data_uri_payload(&result, "image_url").starts_with("data:image/png;base64,"));
    }

    #[test]
    fn convert_errors_for_missing_file_field() {
        let error = convert_file_document_to_url_document(None, None).unwrap_err();

        assert!(invalid_request_message(error).contains("must include a 'file' field"));
    }

    #[test]
    fn convert_errors_for_nonexistent_path() {
        let error = convert_file_document_to_url_document(
            Some(FileInput::Path(PathBuf::from(
                "/nonexistent/path/to/file.pdf",
            ))),
            None,
        )
        .unwrap_err();

        assert!(invalid_request_message(error).contains("File not found"));
    }

    #[test]
    fn convert_errors_for_empty_file() {
        let file = TempFile::with_suffix(".pdf", b"");

        let error =
            convert_file_document_to_url_document(Some(FileInput::Path(file.0.clone())), None)
                .unwrap_err();

        assert!(invalid_request_message(error).contains("File is empty"));
    }

    #[test]
    fn convert_errors_for_unsupported_type() {
        let error = convert_file_document_to_url_document(
            Some(FileInput::Unsupported("int".to_string())),
            None,
        )
        .unwrap_err();

        assert!(invalid_request_message(error).contains("Unsupported file input type"));
    }

    #[test]
    fn convert_errors_for_invalid_mime_type() {
        let error = convert_file_document_to_url_document(
            Some(FileInput::Bytes(b"some content".to_vec())),
            Some("text/html; charset=utf-8\nX-Injected: true"),
        )
        .unwrap_err();

        assert!(invalid_request_message(error).contains("Invalid MIME type"));
    }

    #[test]
    fn convert_explicit_mime_overrides_path_detection() {
        let file = TempFile::with_suffix(".pdf", b"some content");

        let result = convert_file_document_to_url_document(
            Some(FileInput::Path(file.0.clone())),
            Some("image/png"),
        )
        .unwrap();

        assert_eq!(result["type"], "image_url");
        assert!(data_uri_payload(&result, "image_url").starts_with("data:image/png;base64,"));
    }

    #[test]
    fn upload_builds_document_url_for_pdf() {
        let content = b"%PDF-1.4 test content";

        let result = build_document_from_upload(
            content.to_vec(),
            Some("document.pdf"),
            Some("application/pdf"),
        )
        .unwrap();

        assert_eq!(result["type"], "document_url");
        let uri = data_uri_payload(&result, "document_url");
        assert!(uri.starts_with("data:application/pdf;base64,"));
        assert_eq!(decode_base64(uri), content);
    }

    #[test]
    fn upload_builds_image_url_for_png() {
        let result = build_document_from_upload(
            b"\x89PNG fake png".to_vec(),
            Some("screenshot.png"),
            Some("image/png"),
        )
        .unwrap();

        assert_eq!(result["type"], "image_url");
        assert!(data_uri_payload(&result, "image_url").starts_with("data:image/png;base64,"));
    }

    #[test]
    fn upload_builds_image_url_for_jpeg() {
        let result = build_document_from_upload(
            b"\xff\xd8\xff fake jpeg".to_vec(),
            Some("photo.jpg"),
            Some("image/jpeg"),
        )
        .unwrap();

        assert_eq!(result["type"], "image_url");
        assert!(data_uri_payload(&result, "image_url").starts_with("data:image/jpeg;base64,"));
    }

    #[test]
    fn upload_detects_mime_from_filename_when_content_type_is_octet_stream() {
        let result = build_document_from_upload(
            b"pdf content".to_vec(),
            Some("report.pdf"),
            Some("application/octet-stream"),
        )
        .unwrap();

        assert_eq!(result["type"], "document_url");
        assert!(
            data_uri_payload(&result, "document_url").starts_with("data:application/pdf;base64,")
        );
    }

    #[test]
    fn upload_detects_mime_from_filename_when_content_type_is_none() {
        let result =
            build_document_from_upload(b"png content".to_vec(), Some("image.png"), None).unwrap();

        assert_eq!(result["type"], "image_url");
        assert!(data_uri_payload(&result, "image_url").starts_with("data:image/png;base64,"));
    }

    #[test]
    fn upload_falls_back_to_octet_stream_for_unknown() {
        let result = build_document_from_upload(b"unknown content".to_vec(), None, None).unwrap();

        assert_eq!(result["type"], "document_url");
        assert!(data_uri_payload(&result, "document_url").contains("application/octet-stream"));
    }

    #[test]
    fn upload_preserves_binary_content_through_base64() {
        let content = b"Hello, World! \x00\x01\x02\xff";

        let result =
            build_document_from_upload(content.to_vec(), Some("test.pdf"), Some("application/pdf"))
                .unwrap();

        assert_eq!(
            decode_base64(data_uri_payload(&result, "document_url")),
            content
        );
    }

    #[test]
    fn upload_strips_mime_parameters_from_content_type() {
        let result = build_document_from_upload(
            b"%PDF-1.4 test".to_vec(),
            Some("doc.pdf"),
            Some("application/pdf; charset=utf-8"),
        )
        .unwrap();

        assert_eq!(result["type"], "document_url");
        assert!(
            data_uri_payload(&result, "document_url").starts_with("data:application/pdf;base64,")
        );
    }

    #[test]
    fn upload_strips_multiple_mime_parameters() {
        let result = build_document_from_upload(
            b"image data".to_vec(),
            Some("img.png"),
            Some("image/png; charset=utf-8; boundary=something"),
        )
        .unwrap();

        assert_eq!(result["type"], "image_url");
        assert!(data_uri_payload(&result, "image_url").starts_with("data:image/png;base64,"));
    }
}
