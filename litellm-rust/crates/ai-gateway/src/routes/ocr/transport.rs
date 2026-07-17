use std::time::Duration;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use litellm_core::error::CoreError;
use litellm_core::ocr::mime::sniff_mime;
use litellm_core::CoreResult;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::constants::{
    DEFAULT_UPLOAD_MIME_TYPE, GENERIC_UPLOAD_MIME_TYPES, OCR_MULTIPART_UNIQUE_FIELDS,
    OCR_RESERVED_PARAM_KEYS, OCR_UPLOAD_MIME_BY_EXTENSION,
};

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OcrDocument {
    DocumentUrl { document_url: String },
    ImageUrl { image_url: String },
    File {},
}

impl OcrDocument {
    fn into_value(self) -> CoreResult<Value> {
        let (field, url) = match self {
            OcrDocument::DocumentUrl { document_url } => ("document_url", document_url),
            OcrDocument::ImageUrl { image_url } => ("image_url", image_url),
            OcrDocument::File {} => {
                return Err(CoreError::InvalidRequest(
                    "document type 'file' is not supported through the JSON API; upload the \
                     file via multipart/form-data with a 'file' field, or use a 'document_url' \
                     or 'image_url' document type"
                        .to_string(),
                ))
            }
        };
        Ok(json!({ "type": field, field: url }))
    }
}

#[derive(Debug, Deserialize)]
pub struct OcrJsonRequest {
    pub model: String,
    pub document: OcrDocument,
    #[serde(default)]
    pub timeout: Option<f64>,
    #[serde(flatten)]
    pub optional_params: Map<String, Value>,
}

#[derive(Debug)]
pub struct OcrCall {
    pub model: String,
    pub document: Value,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
}

fn parse_timeout(seconds: Option<f64>) -> CoreResult<Option<Duration>> {
    match seconds {
        None => Ok(None),
        Some(value) if value.is_finite() && value > 0.0 => Ok(Some(Duration::from_secs_f64(value))),
        Some(_) => Err(CoreError::InvalidRequest(
            "'timeout' must be a positive, finite number of seconds".to_string(),
        )),
    }
}

fn reject_reserved_params<'a>(names: impl IntoIterator<Item = &'a str>) -> CoreResult<()> {
    let reserved = names.into_iter().find_map(|name| {
        OCR_RESERVED_PARAM_KEYS
            .iter()
            .copied()
            .find(|candidate| *candidate == name)
    });
    match reserved {
        None => Ok(()),
        Some(reserved) => Err(CoreError::InvalidRequest(format!(
            "the '{reserved}' parameter is not accepted on an OCR request; deployment \
             credentials, routing, and headers are server-controlled"
        ))),
    }
}

pub fn parse_json_body(body: &[u8]) -> CoreResult<OcrCall> {
    if body.is_empty() {
        return Err(CoreError::InvalidRequest(
            "empty request body; send a JSON body with 'model' and 'document', or use \
             multipart/form-data for file uploads"
                .to_string(),
        ));
    }
    let request: OcrJsonRequest = serde_json::from_slice(body)
        .map_err(|err| CoreError::InvalidRequest(format!("invalid OCR request body: {err}")))?;
    reject_reserved_params(request.optional_params.keys().map(String::as_str))?;
    Ok(OcrCall {
        model: request.model,
        document: request.document.into_value()?,
        optional_params: request.optional_params,
        timeout: parse_timeout(request.timeout)?,
    })
}

fn mime_from_filename(filename: &str) -> Option<&'static str> {
    let extension = filename
        .rsplit_once('.')
        .map(|(_, ext)| ext.to_ascii_lowercase())?;
    OCR_UPLOAD_MIME_BY_EXTENSION
        .iter()
        .find(|(candidate, _)| *candidate == extension)
        .map(|(_, mime)| *mime)
}

fn is_generic_upload_mime(value: &str) -> bool {
    GENERIC_UPLOAD_MIME_TYPES
        .iter()
        .any(|generic| value.eq_ignore_ascii_case(generic))
}

fn resolve_upload_mime(bytes: &[u8], content_type: Option<&str>, filename: Option<&str>) -> String {
    let declared = content_type
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty() && !is_generic_upload_mime(value));
    if let Some(declared) = declared {
        return declared.to_string();
    }
    if let Some(sniffed) = sniff_mime(bytes) {
        return sniffed.to_string();
    }
    filename
        .and_then(mime_from_filename)
        .unwrap_or(DEFAULT_UPLOAD_MIME_TYPE)
        .to_string()
}

pub fn build_upload_document(
    bytes: Vec<u8>,
    filename: Option<&str>,
    content_type: Option<&str>,
) -> CoreResult<Value> {
    if bytes.is_empty() {
        return Err(CoreError::InvalidRequest(
            "uploaded file is empty".to_string(),
        ));
    }
    let mime = resolve_upload_mime(&bytes, content_type, filename);
    let data_uri = format!("data:{mime};base64,{}", BASE64_STANDARD.encode(&bytes));
    let field = if mime.starts_with("image/") {
        "image_url"
    } else {
        "document_url"
    };
    Ok(json!({ "type": field, field: data_uri }))
}

fn coerce_form_field(value: &str) -> Value {
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()))
}

fn reject_duplicate_unique_fields(text_fields: &[(String, String)]) -> CoreResult<()> {
    let duplicate = OCR_MULTIPART_UNIQUE_FIELDS
        .iter()
        .copied()
        .find(|field| text_fields.iter().filter(|(name, _)| name == field).count() > 1);
    match duplicate {
        None => Ok(()),
        Some(field) => Err(CoreError::InvalidRequest(format!(
            "the '{field}' field must appear at most once in a multipart OCR request"
        ))),
    }
}

pub fn assemble_multipart_call(
    document: Value,
    text_fields: &[(String, String)],
) -> CoreResult<OcrCall> {
    reject_reserved_params(text_fields.iter().map(|(name, _)| name.as_str()))?;
    reject_duplicate_unique_fields(text_fields)?;

    let model = text_fields
        .iter()
        .find(|(name, _)| name == "model")
        .map(|(_, value)| value.clone())
        .ok_or_else(|| {
            CoreError::InvalidRequest(
                "multipart OCR request must include a 'model' form field".to_string(),
            )
        })?;

    let timeout = match text_fields.iter().find(|(name, _)| name == "timeout") {
        Some((_, raw)) => {
            let seconds = raw.parse::<f64>().map_err(|_| {
                CoreError::InvalidRequest(
                    "'timeout' form field must be a number of seconds".to_string(),
                )
            })?;
            parse_timeout(Some(seconds))?
        }
        None => None,
    };

    let optional_params: Map<String, Value> = text_fields
        .iter()
        .filter(|(name, _)| !matches!(name.as_str(), "model" | "timeout" | "file" | "document"))
        .map(|(name, value)| (name.clone(), coerce_form_field(value)))
        .collect();

    Ok(OcrCall {
        model,
        document,
        optional_params,
        timeout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_document_url_and_flattens_provider_params() {
        let call = parse_json_body(
            br#"{"model":"rust-ocr","document":{"type":"document_url","document_url":"https://x/doc.pdf"},"include_image_base64":true}"#,
        )
        .expect("valid body parses");

        assert_eq!(call.model, "rust-ocr");
        assert_eq!(call.document["type"], "document_url");
        assert_eq!(call.document["document_url"], "https://x/doc.pdf");
        assert_eq!(
            call.optional_params["include_image_base64"],
            Value::Bool(true)
        );
        assert!(call.timeout.is_none());
    }

    #[test]
    fn parses_image_url_document() {
        let call = parse_json_body(
            br#"{"model":"m","document":{"type":"image_url","image_url":"https://x/i.png"}}"#,
        )
        .expect("valid body parses");
        assert_eq!(call.document["type"], "image_url");
        assert_eq!(call.document["image_url"], "https://x/i.png");
    }

    #[test]
    fn extracts_timeout_and_keeps_it_out_of_provider_params() {
        let call = parse_json_body(
            br#"{"model":"m","document":{"type":"document_url","document_url":"https://x"},"timeout":12.5}"#,
        )
        .expect("valid body parses");
        assert_eq!(call.timeout, Some(Duration::from_secs_f64(12.5)));
        assert!(!call.optional_params.contains_key("timeout"));
    }

    #[test]
    fn rejects_file_document_over_json() {
        let err =
            parse_json_body(br#"{"model":"m","document":{"type":"file","file":"/etc/passwd"}}"#)
                .expect_err("file type rejected");
        match err {
            CoreError::InvalidRequest(message) => assert!(message.contains("multipart/form-data")),
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn preserves_reducto_file_id_over_json() {
        let call = parse_json_body(
            br#"{"model":"m","document":{"type":"document_url","document_url":"reducto://abc123"}}"#,
        )
        .expect("reducto id preserved");
        assert_eq!(call.document["type"], "document_url");
        assert_eq!(call.document["document_url"], "reducto://abc123");
    }

    #[test]
    fn empty_body_is_rejected() {
        assert!(matches!(
            parse_json_body(b""),
            Err(CoreError::InvalidRequest(_))
        ));
    }

    #[test]
    fn upload_prefers_declared_content_type() {
        let document = build_upload_document(
            b"%PDF-1.4".to_vec(),
            Some("scan.bin"),
            Some("application/pdf"),
        )
        .expect("builds document");
        assert_eq!(document["type"], "document_url");
        assert!(document["document_url"]
            .as_str()
            .expect("data uri")
            .starts_with("data:application/pdf;base64,"));
    }

    #[test]
    fn upload_sniffs_pdf_from_bytes_when_unnamed_octet_stream() {
        let document = build_upload_document(
            b"%PDF-1.7 minimal".to_vec(),
            None,
            Some("application/octet-stream"),
        )
        .expect("builds document");
        assert_eq!(document["type"], "document_url");
        assert!(document["document_url"]
            .as_str()
            .expect("data uri")
            .starts_with("data:application/pdf;base64,"));
    }

    #[test]
    fn upload_sniffs_png_from_bytes_when_unnamed_and_no_content_type() {
        let document = build_upload_document(
            vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00],
            None,
            None,
        )
        .expect("builds document");
        assert_eq!(document["type"], "image_url");
        assert!(document["image_url"]
            .as_str()
            .expect("data uri")
            .starts_with("data:image/png;base64,"));
    }

    #[test]
    fn upload_infers_mime_from_filename_when_octet_stream() {
        let document = build_upload_document(
            vec![0x89, b'P', b'N', b'G'],
            Some("photo.PNG"),
            Some("application/octet-stream"),
        )
        .expect("builds document");
        assert_eq!(document["type"], "image_url");
        assert!(document["image_url"]
            .as_str()
            .expect("data uri")
            .starts_with("data:image/png;base64,"));
    }

    #[test]
    fn upload_falls_back_to_octet_stream() {
        let document = build_upload_document(vec![1, 2, 3], Some("data.unknown"), None)
            .expect("builds document");
        assert_eq!(document["type"], "document_url");
        assert!(document["document_url"]
            .as_str()
            .expect("data uri")
            .starts_with("data:application/octet-stream;base64,"));
    }

    #[test]
    fn empty_upload_is_rejected() {
        assert!(matches!(
            build_upload_document(Vec::new(), Some("a.pdf"), Some("application/pdf")),
            Err(CoreError::InvalidRequest(_))
        ));
    }

    #[test]
    fn multipart_extracts_model_timeout_and_json_params() {
        let document =
            json!({"type": "document_url", "document_url": "data:application/pdf;base64,AA=="});
        let fields = vec![
            ("model".to_string(), "rust-ocr".to_string()),
            ("timeout".to_string(), "30".to_string()),
            ("pages".to_string(), "[0,1,2]".to_string()),
            ("id".to_string(), "abc".to_string()),
        ];
        let call = assemble_multipart_call(document, &fields).expect("assembles call");

        assert_eq!(call.model, "rust-ocr");
        assert_eq!(call.timeout, Some(Duration::from_secs(30)));
        assert_eq!(call.optional_params["pages"], json!([0, 1, 2]));
        assert_eq!(call.optional_params["id"], Value::String("abc".to_string()));
        assert!(!call.optional_params.contains_key("timeout"));
        assert!(!call.optional_params.contains_key("model"));
    }

    #[test]
    fn multipart_requires_model() {
        let document = json!({"type": "document_url", "document_url": "data:x"});
        let err = assemble_multipart_call(document, &[]).expect_err("model required");
        assert!(matches!(err, CoreError::InvalidRequest(_)));
    }

    #[test]
    fn json_rejects_reserved_control_params() {
        for reserved in [
            "api_key",
            "api_base",
            "custom_llm_provider",
            "extra_headers",
            "vertex_credentials",
            "vertex_ai_credentials",
            "vertex_project",
            "vertex_ai_project",
            "vertex_location",
            "vertex_ai_location",
        ] {
            let body = format!(
                r#"{{"model":"m","document":{{"type":"document_url","document_url":"https://x"}},"{reserved}":"attacker"}}"#
            );
            let err = parse_json_body(body.as_bytes())
                .expect_err("reserved control param must be rejected");
            match err {
                CoreError::InvalidRequest(message) => {
                    assert!(
                        message.contains(reserved),
                        "names the rejected key: {message}"
                    );
                    assert!(
                        !message.contains("attacker"),
                        "must not echo the attacker value: {message}"
                    );
                }
                other => panic!("expected InvalidRequest, got {other:?}"),
            }
        }
    }

    #[test]
    fn multipart_rejects_reserved_control_params() {
        let document = json!({"type": "document_url", "document_url": "data:x"});
        let fields = vec![
            ("model".to_string(), "m".to_string()),
            (
                "vertex_credentials".to_string(),
                "/etc/gcp/service-account.json".to_string(),
            ),
        ];
        let err = assemble_multipart_call(document, &fields).expect_err("reserved param rejected");
        match err {
            CoreError::InvalidRequest(message) => {
                assert!(message.contains("vertex_credentials"));
                assert!(
                    !message.contains("service-account"),
                    "must not echo the attacker value: {message}"
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn json_rejects_non_positive_timeout() {
        for timeout in ["0", "-1", "-0.5"] {
            let body = format!(
                r#"{{"model":"m","document":{{"type":"document_url","document_url":"https://x"}},"timeout":{timeout}}}"#
            );
            assert!(
                matches!(
                    parse_json_body(body.as_bytes()),
                    Err(CoreError::InvalidRequest(_))
                ),
                "timeout {timeout} must be rejected"
            );
        }
    }

    #[test]
    fn multipart_rejects_non_positive_and_non_finite_timeout() {
        let document = json!({"type": "document_url", "document_url": "data:x"});
        for timeout in ["0", "-3", "inf", "-inf", "NaN"] {
            let fields = vec![
                ("model".to_string(), "m".to_string()),
                ("timeout".to_string(), timeout.to_string()),
            ];
            assert!(
                matches!(
                    assemble_multipart_call(document.clone(), &fields),
                    Err(CoreError::InvalidRequest(_))
                ),
                "timeout {timeout} must be rejected"
            );
        }
    }

    #[test]
    fn multipart_rejects_non_numeric_timeout_without_echoing_value() {
        let document = json!({"type": "document_url", "document_url": "data:x"});
        let fields = vec![
            ("model".to_string(), "m".to_string()),
            ("timeout".to_string(), "not-a-number".to_string()),
        ];
        let err =
            assemble_multipart_call(document, &fields).expect_err("non-numeric timeout rejected");
        match err {
            CoreError::InvalidRequest(message) => assert!(
                !message.contains("not-a-number"),
                "must not echo the attacker value: {message}"
            ),
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn multipart_rejects_duplicate_model_field() {
        let document = json!({"type": "document_url", "document_url": "data:x"});
        let fields = vec![
            ("model".to_string(), "first".to_string()),
            ("model".to_string(), "second".to_string()),
        ];
        let err = assemble_multipart_call(document, &fields).expect_err("duplicate model rejected");
        assert!(matches!(err, CoreError::InvalidRequest(_)));
    }

    #[test]
    fn upload_treats_binary_octet_stream_as_generic() {
        let document = build_upload_document(
            b"%PDF-1.7 minimal".to_vec(),
            None,
            Some("Binary/Octet-Stream"),
        )
        .expect("builds document");
        assert!(document["document_url"]
            .as_str()
            .expect("data uri")
            .starts_with("data:application/pdf;base64,"));
    }
}
