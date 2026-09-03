use litellm_core::error::Error;
use serde_json::{Map, Value};

use super::file_input::build_document_from_upload;

pub fn parse_ocr_json_body(body: &[u8]) -> Result<Map<String, Value>, Error> {
    let data: Value = serde_json::from_slice(body).map_err(|err| {
        Error::InvalidRequest(format!(
            "Invalid JSON in request body: {err}. Ensure the request body is valid JSON with \
             Content-Type: application/json, or use multipart/form-data for file uploads."
        ))
    })?;
    let Value::Object(data) = data else {
        return Err(Error::InvalidRequest(
            "OCR request body must be a JSON object".to_string(),
        ));
    };
    let document_type = data
        .get("document")
        .and_then(Value::as_object)
        .and_then(|document| document.get("type"))
        .and_then(Value::as_str);
    if document_type == Some("file") {
        return Err(Error::InvalidRequest(
            "document type 'file' is not supported through the JSON API. To upload a local file, \
             use multipart/form-data with a 'file' field. For JSON requests, use 'document_url' \
             or 'image_url' document types."
                .to_string(),
        ));
    }
    Ok(data)
}

pub fn parse_ocr_multipart_form(
    file_content: Vec<u8>,
    filename: Option<&str>,
    content_type: Option<&str>,
    fields: &[(String, String)],
) -> Result<Map<String, Value>, Error> {
    if file_content.is_empty() {
        return Err(Error::InvalidRequest("Uploaded file is empty".to_string()));
    }
    let document = build_document_from_upload(file_content, filename, content_type)?;
    let mut data = Map::new();
    data.insert("document".to_string(), document);
    for (name, value) in fields {
        if name == "file" || name == "document" {
            continue;
        }
        let parsed =
            serde_json::from_str::<Value>(value).unwrap_or_else(|_| Value::String(value.clone()));
        data.insert(name.clone(), parsed);
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invalid_request_message(error: Error) -> String {
        match error {
            Error::InvalidRequest(message) => message,
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn json_body_rejects_file_type_document() {
        let body = br#"{"model": "mistral/mistral-ocr-latest", "document": {"type": "file", "file": "/etc/passwd"}}"#;

        let error = parse_ocr_json_body(body).unwrap_err();

        assert!(invalid_request_message(error).contains("not supported through the JSON API"));
    }

    #[test]
    fn json_body_accepts_document_url_type() {
        let body = br#"{"model": "mistral/mistral-ocr-latest", "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"}}"#;

        let data = parse_ocr_json_body(body).unwrap();

        assert_eq!(data["document"]["type"], "document_url");
        assert_eq!(data["model"], "mistral/mistral-ocr-latest");
    }

    #[test]
    fn json_body_rejects_invalid_json() {
        let error = parse_ocr_json_body(b"not valid json{{{").unwrap_err();

        assert!(invalid_request_message(error).contains("Invalid JSON in request body"));
    }

    #[test]
    fn multipart_ignores_document_form_field_injection() {
        let fields = vec![
            (
                "model".to_string(),
                "mistral/mistral-ocr-latest".to_string(),
            ),
            (
                "document".to_string(),
                r#"{"type": "file", "file": "/etc/passwd"}"#.to_string(),
            ),
            ("pages".to_string(), "[0,1,2]".to_string()),
        ];

        let data = parse_ocr_multipart_form(
            b"%PDF-1.4 legit content".to_vec(),
            Some("legit.pdf"),
            Some("application/pdf"),
            &fields,
        )
        .unwrap();

        assert_eq!(data["document"]["type"], "document_url");
        assert!(
            data["document"]["document_url"]
                .as_str()
                .unwrap()
                .starts_with("data:application/pdf;base64,")
        );
        assert_eq!(data["model"], "mistral/mistral-ocr-latest");
        assert_eq!(data["pages"], serde_json::json!([0, 1, 2]));
    }

    #[test]
    fn multipart_rejects_empty_upload() {
        let error = parse_ocr_multipart_form(Vec::new(), Some("empty.pdf"), None, &[]).unwrap_err();

        assert!(invalid_request_message(error).contains("Uploaded file is empty"));
    }
}
