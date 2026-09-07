pub mod prepare;
pub mod transformation;
pub mod types;

use serde_json::Value;

use crate::Error;
use crate::error::json_type_name;
use crate::http_utils::{buffered_post, has_header};

use types::{OcrDocument, OcrDocumentProjection};
pub use types::{OcrRequest, OcrResponseData, PreparedOcr};

pub fn terminal_callbacks(asynchronous: bool, success: bool) -> &'static [&'static str] {
    match (asynchronous, success) {
        (false, true) => &["sync_success"],
        (true, true) => &["async_success", "sync_success_if_needed"],
        (false, false) => &["sync_failure"],
        (true, false) => &["sync_failure", "async_failure"],
    }
}

pub async fn ocr(
    prepared: PreparedOcr,
    headers: Vec<(String, String)>,
    body: Value,
) -> Result<OcrResponseData, Error> {
    let config = prepare::provider_config(&prepared.custom_llm_provider, &prepared.model)?;
    prepare::validate_capabilities(config)?;
    let object = body.as_object().ok_or_else(|| Error::InvalidType {
        expected: "object",
        actual: json_type_name(&body),
    })?;
    if config.document_projection() != OcrDocumentProjection::Transformed {
        let document = object
            .get("document")
            .ok_or(Error::MissingField("document"))?;
        let document: OcrDocument = serde_json::from_value(document.clone())
            .map_err(|_| Error::InvalidRequest("invalid OCR document".into()))?;
        document.validate(config.requires_data_uri_document())?;
    }
    if object.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err(Error::Unsupported("OCR streaming response handling"));
    }
    if headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("content-encoding") && !value.eq_ignore_ascii_case("identity")
    }) {
        return Err(Error::Unsupported("compressed OCR request"));
    }
    let headers = headers
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .chain(
            [
                ("Content-Type", "application/json"),
                ("Accept-Encoding", "identity"),
            ]
            .into_iter()
            .filter(|(name, _)| !has_header(&headers, name)),
        )
        .map(|(name, value)| (name.as_bytes().to_vec(), value.as_bytes().to_vec()))
        .collect();
    let body = serde_json::to_vec(&body)
        .map_err(|_| Error::InvalidRequest("could not encode OCR request".into()))?;
    let response = buffered_post::send(buffered_post::Request {
        url: prepared.url,
        headers,
        body,
        timeout_seconds: prepared.timeout_seconds,
    })
    .await?;
    if !(200..300).contains(&response.status) {
        return Err(Error::Http {
            status: response.status,
            body: "OCR provider request failed".into(),
        });
    }
    if response.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case(b"content-encoding")
            && value
                .split(|byte| *byte == b',')
                .any(|encoding| !encoding.trim_ascii().eq_ignore_ascii_case(b"identity"))
    }) {
        return Err(Error::Unsupported("compressed OCR response"));
    }
    let response_json = serde_json::from_slice(&response.content)
        .map_err(|_| Error::InvalidResponse("invalid OCR JSON response".into()))?;
    config.transform_ocr_response(&prepared.model, response_json)
}
