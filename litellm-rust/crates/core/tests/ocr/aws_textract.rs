use std::{collections::BTreeMap, time::SystemTime};

use litellm_auth_aws::{Credentials, aws_signature_headers, sign_post};
use litellm_llms::base_llm::ocr::error::Error;
use serde_json::{Value, json};
use time::{PrimitiveDateTime, format_description};

use crate::ocr::{
    route::LocalOcrHost,
    test_support::{
        MockResponse, header, mock_server, perform_ocr_with, request_body,
        wire_request_with_document,
    },
    types::LiteLLMOcrRequest,
};

const ACCESS_KEY_ID: &str = "AKIDEXAMPLE";
const SECRET_ACCESS_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";

fn textract_request(base: &str) -> LiteLLMOcrRequest {
    textract_request_for("aws_textract/detect-document-text", base)
}

fn textract_request_for(model: &str, base: &str) -> LiteLLMOcrRequest {
    wire_request_with_document(
        model,
        &format!("{base}/"),
        json!({"type": "image_url", "image_url": "data:image/png;base64,b3JpZ2luYWw="}),
        json!({
            "aws_access_key_id": ACCESS_KEY_ID,
            "aws_secret_access_key": SECRET_ACCESS_KEY,
            "aws_region_name": "eu-west-1"
        }),
    )
}

fn textract_response() -> MockResponse {
    MockResponse::json(json!({
        "DocumentMetadata": {"Pages": 1},
        "Blocks": [{"BlockType": "PAGE"}, {"BlockType": "LINE", "Text": "Invoice 12345"}]
    }))
}

/// Recomputes SigV4 over the bytes the server received, at the time the client claimed.
fn expected_authorization(url: &str, raw_request: &str) -> String {
    let format =
        format_description::parse_borrowed::<2>("[year][month][day]T[hour][minute][second]Z")
            .unwrap();
    let signed_at: SystemTime =
        PrimitiveDateTime::parse(header(raw_request, "x-amz-date").unwrap(), &format)
            .unwrap()
            .assume_utc()
            .into();
    let headers: BTreeMap<String, String> = ["content-type", "x-amz-target"]
        .into_iter()
        .map(|name| {
            (
                name.to_string(),
                header(raw_request, name).unwrap().to_string(),
            )
        })
        .collect();
    let body = raw_request.split_once("\r\n\r\n").unwrap().1;
    sign_post(
        url,
        body.as_bytes(),
        &aws_signature_headers(&headers),
        "eu-west-1",
        "textract",
        &Credentials::new(ACCESS_KEY_ID, SECRET_ACCESS_KEY, None, None, "test"),
        signed_at,
    )
    .unwrap()["Authorization"]
        .clone()
}

#[tokio::test]
async fn the_request_is_signed_for_textract_and_lines_become_the_page() {
    let (base, seen, server) = mock_server(vec![textract_response()]).await;

    let response = perform_ocr_with(LocalOcrHost::new(textract_request(&base)))
        .await
        .unwrap();
    server.await.unwrap();

    let raw = seen.lock().unwrap()[0].clone();
    assert_eq!(
        header(&raw, "x-amz-target"),
        Some("Textract.DetectDocumentText")
    );
    assert_eq!(
        header(&raw, "content-type"),
        Some("application/x-amz-json-1.1")
    );
    assert_eq!(
        request_body(&raw),
        json!({"Document": {"Bytes": "b3JpZ2luYWw="}})
    );
    assert_eq!(
        header(&raw, "authorization"),
        Some(expected_authorization(&format!("{base}/"), &raw).as_str())
    );
    assert_eq!(response.pages[0].markdown, "Invoice 12345");
    assert_eq!(response.usage_info.unwrap().pages_processed, Some(1));
}

#[tokio::test]
async fn a_body_rewritten_by_before_send_is_what_gets_signed_and_sent() {
    let (base, seen, server) = mock_server(vec![textract_response()]).await;
    let host = LocalOcrHost::new(textract_request(&base)).with_before_send(|mut wire, _| {
        assert!(
            !wire
                .headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("authorization")),
            "the hook ran after signing"
        );
        wire.body["Document"]["Bytes"] = Value::from("cmVkYWN0ZWQ=");
        Ok(wire)
    });

    perform_ocr_with(host).await.unwrap();
    server.await.unwrap();

    let raw = seen.lock().unwrap()[0].clone();
    assert_eq!(
        request_body(&raw),
        json!({"Document": {"Bytes": "cmVkYWN0ZWQ="}})
    );
    assert_eq!(
        header(&raw, "authorization"),
        Some(expected_authorization(&format!("{base}/"), &raw).as_str())
    );
}

#[tokio::test]
async fn a_multi_page_rejection_reaches_the_caller_with_the_single_page_limit() {
    let (base, _, server) = mock_server(vec![MockResponse {
        status: 400,
        headers: vec![],
        body: json!({
            "__type": "UnsupportedDocumentException",
            "Message": "Request has unsupported document format"
        }),
    }])
    .await;

    let error = perform_ocr_with(LocalOcrHost::new(textract_request(&base)))
        .await
        .unwrap_err();
    server.await.unwrap();

    let Error::Provider { status, body, .. } = error else {
        panic!("expected a provider error, got {error:?}");
    };
    assert_eq!(status, 400);
    assert!(
        body.contains("multi-page documents are not supported"),
        "{body}"
    );
}

#[tokio::test]
async fn analyze_document_asks_for_layout_and_tables_and_returns_markdown() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "DocumentMetadata": {"Pages": 1},
        "Blocks": [
            {"Id": "l1", "BlockType": "LINE", "Text": "Quarterly Report"},
            {"Id": "t", "BlockType": "LAYOUT_TITLE",
                "Relationships": [{"Type": "CHILD", "Ids": ["l1"]}]}
        ]
    }))])
    .await;
    let request = textract_request_for("aws_textract/analyze-document", &base);

    let response = perform_ocr_with(LocalOcrHost::new(request)).await.unwrap();
    server.await.unwrap();

    let raw = seen.lock().unwrap()[0].clone();
    assert_eq!(
        header(&raw, "x-amz-target"),
        Some("Textract.AnalyzeDocument")
    );
    assert_eq!(
        request_body(&raw)["FeatureTypes"],
        json!(["LAYOUT", "TABLES"])
    );
    assert_eq!(
        header(&raw, "authorization"),
        Some(expected_authorization(&format!("{base}/"), &raw).as_str())
    );
    assert_eq!(response.pages[0].markdown, "# Quarterly Report");
}
