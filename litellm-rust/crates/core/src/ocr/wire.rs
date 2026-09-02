use std::collections::BTreeMap;
use std::io::{self, Write};
use std::str::Utf8Error;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use mime::Mime;
use serde_json::Value;
use thiserror::Error;

const BASE64_INPUT_CHUNK_SIZE: usize = 48 * 1024;

#[derive(Debug, Error)]
pub enum OcrWireError {
    #[error("failed to encode OCR JSON body: {0}")]
    Json(#[from] serde_json::Error),
    #[error("failed to write OCR body: {0}")]
    Io(#[from] io::Error),
    #[error("encoded OCR data URI is not UTF-8: {0}")]
    InvalidDataUri(#[from] Utf8Error),
}

#[derive(Clone, PartialEq)]
pub enum OcrJsonValue {
    Value(Value),
    Array(Vec<Self>),
    Object(BTreeMap<String, Self>),
    InlineDataUri { media_type: Mime, bytes: Bytes },
    EncodedDataUri(Bytes),
}

impl OcrJsonValue {
    fn write_to(&self, writer: &mut impl Write) -> Result<(), OcrWireError> {
        match self {
            Self::Value(value) => serde_json::to_writer(writer, value).map_err(Into::into),
            Self::Array(values) => {
                writer.write_all(b"[")?;
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        writer.write_all(b",")?;
                    }
                    value.write_to(writer)?;
                }
                writer.write_all(b"]")?;
                Ok(())
            }
            Self::Object(fields) => {
                writer.write_all(b"{")?;
                for (index, (key, value)) in fields.iter().enumerate() {
                    if index != 0 {
                        writer.write_all(b",")?;
                    }
                    serde_json::to_writer(&mut *writer, key)?;
                    writer.write_all(b":")?;
                    value.write_to(writer)?;
                }
                writer.write_all(b"}")?;
                Ok(())
            }
            Self::InlineDataUri { media_type, bytes } => {
                writer.write_all(b"\"data:")?;
                writer.write_all(media_type.as_ref().as_bytes())?;
                writer.write_all(b";base64,")?;
                for chunk in bytes.chunks(BASE64_INPUT_CHUNK_SIZE) {
                    let encoded = STANDARD.encode(chunk);
                    writer.write_all(encoded.as_bytes())?;
                }
                writer.write_all(b"\"")?;
                Ok(())
            }
            Self::EncodedDataUri(data_uri) => {
                let data_uri = std::str::from_utf8(data_uri)?;
                serde_json::to_writer(writer, data_uri).map_err(Into::into)
            }
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum MultipartPart {
    Text {
        name: String,
        value: String,
    },
    Json {
        name: String,
        value: Value,
    },
    File {
        name: String,
        file_name: String,
        media_type: Mime,
        bytes: Bytes,
    },
}

#[derive(Clone, PartialEq)]
pub struct MultipartBodyPlan {
    boundary: String,
    parts: Vec<MultipartPart>,
}

impl MultipartBodyPlan {
    pub fn new(boundary: impl Into<String>, parts: Vec<MultipartPart>) -> Self {
        Self {
            boundary: boundary.into(),
            parts,
        }
    }

    pub fn boundary(&self) -> &str {
        &self.boundary
    }

    pub fn parts(&self) -> &[MultipartPart] {
        &self.parts
    }

    fn write_to(&self, writer: &mut impl Write) -> Result<(), OcrWireError> {
        for part in &self.parts {
            write!(writer, "--{}\r\n", self.boundary)?;
            match part {
                MultipartPart::Text { name, value } => {
                    write!(
                        writer,
                        "Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                    )?;
                }
                MultipartPart::Json { name, value } => {
                    write!(
                        writer,
                        "Content-Disposition: form-data; name=\"{name}\"\r\nContent-Type: application/json\r\n\r\n"
                    )?;
                    serde_json::to_writer(&mut *writer, value)?;
                    writer.write_all(b"\r\n")?;
                }
                MultipartPart::File {
                    name,
                    file_name,
                    media_type,
                    bytes,
                } => {
                    write!(
                        writer,
                        "Content-Disposition: form-data; name=\"{name}\"; filename=\"{file_name}\"\r\nContent-Type: {media_type}\r\n\r\n"
                    )?;
                    writer.write_all(bytes)?;
                    writer.write_all(b"\r\n")?;
                }
            }
        }
        write!(writer, "--{}--\r\n", self.boundary)?;
        Ok(())
    }
}

#[derive(Clone, PartialEq)]
pub enum OcrWireBody {
    Json(Value),
    JsonWithMedia(OcrJsonValue),
    Multipart(MultipartBodyPlan),
}

impl OcrWireBody {
    pub fn content_type(&self) -> String {
        match self {
            Self::Json(_) | Self::JsonWithMedia(_) => "application/json".to_string(),
            Self::Multipart(plan) => {
                format!("multipart/form-data; boundary={}", plan.boundary())
            }
        }
    }

    pub fn write_to(&self, mut writer: impl Write) -> Result<(), OcrWireError> {
        match self {
            Self::Json(value) => serde_json::to_writer(writer, value).map_err(Into::into),
            Self::JsonWithMedia(value) => value.write_to(&mut writer),
            Self::Multipart(plan) => plan.write_to(&mut writer),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::Arc;

    use serde_json::json;

    use super::*;

    #[test]
    fn ordinary_json_body_writes_without_a_media_plan() {
        let body = OcrWireBody::Json(json!({"model": "ocr-model", "pages": [0, 2]}));
        let mut encoded = Vec::new();

        body.write_to(&mut encoded).expect("body writes");

        assert_eq!(body.content_type(), "application/json");
        assert_eq!(
            serde_json::from_slice::<Value>(&encoded).expect("valid JSON"),
            json!({"model": "ocr-model", "pages": [0, 2]})
        );
    }

    #[test]
    fn json_with_media_streams_raw_bytes_as_a_data_uri() {
        let owner: Arc<[u8]> = vec![b'x'; BASE64_INPUT_CHUNK_SIZE + 1].into();
        let bytes = Bytes::from_owner(Arc::clone(&owner));
        let source_pointer = bytes.as_ptr();
        let body = OcrWireBody::JsonWithMedia(OcrJsonValue::Object(BTreeMap::from([
            (
                "document".to_string(),
                OcrJsonValue::InlineDataUri {
                    media_type: "application/pdf".parse().expect("valid MIME type"),
                    bytes,
                },
            ),
            ("model".to_string(), OcrJsonValue::Value(json!("ocr-model"))),
        ])));

        let OcrWireBody::JsonWithMedia(OcrJsonValue::Object(fields)) = &body else {
            panic!("media JSON body must retain its typed representation");
        };
        let OcrJsonValue::InlineDataUri { bytes, .. } = &fields["document"] else {
            panic!("document must remain shared binary media");
        };
        assert_eq!(bytes.as_ptr(), source_pointer);
        assert_eq!(Arc::strong_count(&owner), 2);

        let mut encoded = Vec::new();
        body.write_to(&mut encoded).expect("body writes");
        let expected_data_uri = format!(
            "data:application/pdf;base64,{}",
            STANDARD.encode(owner.as_ref())
        );
        let expected = json!({"document": expected_data_uri, "model": "ocr-model"});
        assert_eq!(
            serde_json::from_slice::<Value>(&encoded).expect("valid JSON"),
            expected
        );
    }

    #[test]
    fn raw_media_encoding_uses_bounded_writes() {
        let bytes = Bytes::from(vec![b'x'; BASE64_INPUT_CHUNK_SIZE * 3 + 1]);
        let body = OcrWireBody::JsonWithMedia(OcrJsonValue::InlineDataUri {
            media_type: "application/pdf".parse().expect("valid MIME type"),
            bytes,
        });
        let mut sink = BoundedSink {
            maximum_write: BASE64_INPUT_CHUNK_SIZE * 4 / 3,
            written: 0,
        };

        body.write_to(&mut sink).expect("writes remain bounded");

        assert!(sink.written > BASE64_INPUT_CHUNK_SIZE * 4);
    }

    #[test]
    fn encoded_data_uri_is_retained_without_decoding_or_copying() {
        let data_uri = Bytes::from_static(b"data:image/png;base64,aGVsbG8=");
        let source_pointer = data_uri.as_ptr();
        let body = OcrWireBody::JsonWithMedia(OcrJsonValue::EncodedDataUri(data_uri));

        let OcrWireBody::JsonWithMedia(OcrJsonValue::EncodedDataUri(retained)) = &body else {
            panic!("encoded data URI must remain bytes");
        };
        assert_eq!(retained.as_ptr(), source_pointer);

        let mut encoded = Vec::new();
        body.write_to(&mut encoded).expect("body writes");
        assert_eq!(encoded, br#""data:image/png;base64,aGVsbG8=""#);
    }

    #[test]
    fn encoded_data_uri_is_json_escaped_without_changing_its_allocation() {
        let data_uri = Bytes::from_static(b"data:text/plain,quoted%20\"value\"");
        let source_pointer = data_uri.as_ptr();
        let body = OcrWireBody::JsonWithMedia(OcrJsonValue::EncodedDataUri(data_uri));

        let mut encoded = Vec::new();
        body.write_to(&mut encoded).expect("body writes");

        let OcrWireBody::JsonWithMedia(OcrJsonValue::EncodedDataUri(retained)) = &body else {
            panic!("encoded data URI expected");
        };
        assert_eq!(retained.as_ptr(), source_pointer);
        assert_eq!(
            serde_json::from_slice::<String>(&encoded).expect("valid JSON string"),
            "data:text/plain,quoted%20\"value\""
        );
    }

    #[test]
    fn multipart_file_is_replayable_and_retains_shared_bytes() {
        let file = Bytes::from_static(b"large-pdf-payload");
        let source_pointer = file.as_ptr();
        let body = OcrWireBody::Multipart(MultipartBodyPlan::new(
            "ocr-boundary",
            vec![MultipartPart::File {
                name: "file".to_string(),
                file_name: "document.pdf".to_string(),
                media_type: "application/pdf".parse().expect("valid MIME type"),
                bytes: file,
            }],
        ));

        let OcrWireBody::Multipart(plan) = &body else {
            panic!("multipart body expected");
        };
        let MultipartPart::File { bytes, .. } = &plan.parts()[0] else {
            panic!("file part expected");
        };
        assert_eq!(bytes.as_ptr(), source_pointer);

        let mut first = Vec::new();
        let mut retry = Vec::new();
        body.write_to(&mut first).expect("first write succeeds");
        body.write_to(&mut retry).expect("retry write succeeds");
        assert_eq!(first, retry);
        assert!(
            first
                .windows(file_name_marker().len())
                .any(|window| window == file_name_marker())
        );
        assert!(
            first
                .windows(bytes.len())
                .any(|window| window == bytes.as_ref())
        );
    }

    fn file_name_marker() -> &'static [u8] {
        b"filename=\"document.pdf\""
    }

    struct BoundedSink {
        maximum_write: usize,
        written: usize,
    }

    impl Write for BoundedSink {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            if buffer.len() > self.maximum_write {
                return Err(io::Error::other("write exceeded bound"));
            }
            self.written += buffer.len();
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}
