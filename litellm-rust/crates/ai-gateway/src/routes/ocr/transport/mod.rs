use std::time::Duration;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use litellm_core::error::CoreError;
use litellm_core::ocr::mime::sniff_mime;
use litellm_core::CoreResult;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::constants::{
    DEFAULT_UPLOAD_MIME_TYPE, GENERIC_UPLOAD_MIME_TYPES, OCR_MULTIPART_RESERVED_TEXT_FIELDS,
    OCR_MULTIPART_UNIQUE_FIELDS, OCR_RESERVED_PARAM_KEYS, OCR_UPLOAD_MIME_BY_EXTENSION,
};

#[cfg(test)]
mod tests;

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
    let Some(value) = seconds else {
        return Ok(None);
    };
    if !value.is_finite() || value <= 0.0 {
        return Err(CoreError::InvalidRequest(
            "'timeout' must be a positive, finite number of seconds".to_string(),
        ));
    }
    Duration::try_from_secs_f64(value).map(Some).map_err(|_| {
        CoreError::InvalidRequest("'timeout' is larger than the supported range".to_string())
    })
}

fn validate_model(model: &str) -> CoreResult<()> {
    if model.trim().is_empty() {
        return Err(CoreError::InvalidRequest(
            "'model' must be a non-empty string".to_string(),
        ));
    }
    Ok(())
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
    let request: OcrJsonRequest = serde_json::from_slice(body).map_err(|_| {
        CoreError::InvalidRequest(
            "request body is not a valid OCR request; expected JSON with 'model' and 'document'"
                .to_string(),
        )
    })?;
    validate_model(&request.model)?;
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
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .filter(|value| !is_generic_upload_mime(value));
    if let Some(declared) = declared {
        return declared;
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

fn reject_reserved_text_fields(text_fields: &[(String, String)]) -> CoreResult<()> {
    let reserved = OCR_MULTIPART_RESERVED_TEXT_FIELDS
        .iter()
        .copied()
        .find(|field| text_fields.iter().any(|(name, _)| name == field));
    match reserved {
        None => Ok(()),
        Some(field) => Err(CoreError::InvalidRequest(format!(
            "the '{field}' field is not accepted as a form field; upload the document through the \
             'file' field"
        ))),
    }
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
    reject_reserved_text_fields(text_fields)?;
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
    validate_model(&model)?;

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
