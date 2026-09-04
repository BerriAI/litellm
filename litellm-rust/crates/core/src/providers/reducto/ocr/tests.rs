use rstest::{fixture, rstest};
use serde_json::{Value, json};

use super::transformation::*;
use crate::ocr::transformation::OcrProviderConfig;

#[fixture]
fn parse_response() -> Value {
    json!({
        "job_id": "job_123",
        "usage": {"num_pages": 3, "credits": 3},
        "result": {
            "chunks": [
                {
                    "content": "Page 1 block A",
                    "blocks": [{
                        "content": "Page 1 block A",
                        "bbox": {"page": 1},
                        "kind": "text",
                    }],
                },
                {
                    "content": "Page 2 block A",
                    "blocks": [{
                        "content": "Page 2 block A",
                        "bbox": {"page": 2},
                        "kind": "table",
                    }],
                },
                {
                    "content": "Page 1 block B",
                    "blocks": [{
                        "content": "Page 1 block B",
                        "bbox": {"page": 1},
                        "kind": "text",
                    }],
                },
                {
                    "content": "Page 3 block A",
                    "blocks": [{
                        "content": "Page 3 block A",
                        "bbox": {"page": 3},
                        "kind": "figure",
                    }],
                },
            ],
        },
    })
}

#[rstest]
fn test_parse_v3_file_upload_and_response_mapping(parse_response: Value) {
    let source = classify_document_source("data:application/pdf;base64,JVBERi0xLjQ=")
        .expect("PDF data URI should be valid");
    let upload = build_upload_request(
        source,
        "Bearer test-key",
        Some("https://platform.reducto.ai"),
    )
    .expect("data URI should require upload");
    assert_eq!(upload.url, "https://platform.reducto.ai/upload");
    assert_eq!(upload.authorization, "Bearer test-key");
    assert_eq!(upload.file_name, "document");
    assert_eq!(upload.mime_type, "application/pdf");
    assert_eq!(upload.bytes, b"%PDF-1.4");

    let optional_params = json!({
        "formatting": {"table_output_format": "html"},
        "retrieval": {"chunk_mode": "section"},
        "settings": {"ocr_system": "standard"},
    })
    .as_object()
    .expect("params should be an object")
    .clone();
    let request = build_parse_v3_request("reducto://uploaded.pdf", optional_params);
    assert_eq!(
        request.data,
        json!({
            "input": "reducto://uploaded.pdf",
            "formatting": {"table_output_format": "html"},
            "retrieval": {"chunk_mode": "section"},
            "settings": {"ocr_system": "standard"},
        })
    );

    let transformed = transform_reducto_response("parse-v3", parse_response.clone())
        .expect("response should transform");
    assert_eq!(
        transformed.usage_info,
        Some(json!({"pages_processed": 3, "credits": 3}))
    );
    assert_eq!(transformed.pages.len(), 3);
    assert_eq!(
        transformed.pages[0],
        json!({
            "index": 0,
            "markdown": "Page 1 block A\n\nPage 1 block B",
            "blocks": [
                {"content": "Page 1 block A", "bbox": {"page": 1}, "kind": "text"},
                {"content": "Page 1 block B", "bbox": {"page": 1}, "kind": "text"},
            ],
        })
    );
    assert_eq!(transformed.pages[1]["markdown"], "Page 2 block A");
    assert_eq!(transformed.pages[2]["markdown"], "Page 3 block A");
    assert_eq!(transformed.provider_native_response, Some(parse_response));
}

#[rstest]
fn test_parse_v3_reducto_id_passthrough_skips_upload(parse_response: Value) {
    let document = json!({
        "type": "document_url",
        "document_url": "reducto://already-uploaded.pdf",
    });
    let source = extract_document_source(&document).expect("Reducto ID should be valid");
    assert!(build_upload_request(source.clone(), "Bearer test-key", None).is_none());
    assert_eq!(
        source,
        ReductoDocumentSource::FileId("reducto://already-uploaded.pdf".to_string())
    );

    let request = REDUCTO_PARSE_V3_CONFIG
        .transform_ocr_request(
            "parse-v3",
            document,
            json!({"retrieval": {"chunk_mode": "section"}})
                .as_object()
                .expect("params should be object")
                .clone(),
        )
        .expect("direct ID should transform");
    assert_eq!(request.data["input"], "reducto://already-uploaded.pdf");
    assert_eq!(request.data["retrieval"]["chunk_mode"], "section");

    let response = REDUCTO_PARSE_V3_CONFIG
        .transform_ocr_response("parse-v3", parse_response)
        .expect("response should transform");
    assert!(
        response.pages[0]["markdown"]
            .as_str()
            .expect("markdown should be string")
            .starts_with("Page 1 block A")
    );
}

#[rstest]
fn test_parse_legacy_wraps_enhance_under_options() {
    let request = build_parse_legacy_request(
        "reducto://legacy.pdf",
        json!({"enhance": {"agentic": [{"type": "table"}]}})
            .as_object()
            .expect("params should be object"),
    );
    assert_eq!(
        request.data,
        json!({
            "document_url": "reducto://legacy.pdf",
            "options": {"enhance": {"agentic": [{"type": "table"}]}},
        })
    );
}

#[rstest]
fn test_parse_v3_image_data_uri_upload_uses_image_mime() {
    let source = classify_document_source("data:image/png;base64,iVBORw0KGgo=")
        .expect("PNG data URI should be valid");
    let upload = build_upload_request(
        source,
        "Bearer programmatic-key",
        Some("https://custom.reducto.test/"),
    )
    .expect("data URI should require upload");
    assert_eq!(upload.url, "https://custom.reducto.test/upload");
    assert_eq!(upload.authorization, "Bearer programmatic-key");
    assert_eq!(upload.mime_type, "image/png");
    assert_eq!(upload.bytes, b"\x89PNG\r\n\x1a\n");
}

#[rstest]
#[case::http("http://example.com/document.pdf")]
#[case::https("https://example.com/document.pdf")]
fn test_parse_v3_rejects_plain_http_urls(#[case] source: &str) {
    let error = classify_document_source(source).expect_err("plain URL should be rejected");
    assert!(error.to_string().contains("upload the file first"));
}

#[rstest]
fn test_parse_v3_uses_programmatic_api_key_over_env() {
    let key = resolve_api_key(Some("passed-key"), &|_| Some("env-reducto-key".to_string()))
        .expect("explicit key should resolve");
    assert_eq!(key, "passed-key");

    let headers = REDUCTO_PARSE_V3_CONFIG
        .validate_environment(Vec::new(), Some("passed-key"), &|_| {
            Some("env-reducto-key".to_string())
        })
        .expect("headers should validate");
    assert_eq!(
        headers,
        vec![("Authorization".to_string(), "Bearer passed-key".to_string())]
    );
}
