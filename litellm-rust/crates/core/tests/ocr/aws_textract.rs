use std::{collections::BTreeMap, time::SystemTime};

use litellm_auth_aws::{Credentials, aws_signature_headers, sign_post};
use rstest::rstest;
use time::{PrimitiveDateTime, format_description};
use wiremock::Request;

use super::*;

const ACCESS_KEY_ID: &str = "AKIDEXAMPLE";
const SECRET_ACCESS_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
const DETECT: &str = "aws_textract/detect-document-text";
const ANALYZE: &str = "aws_textract/analyze-document";

fn textract_request(model: &str, base: &str) -> LiteLLMOcrRequest {
    ocr_request_with_document(
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

fn textract_response() -> ResponseTemplate {
    json_response(json!({
        "DocumentMetadata": {"Pages": 1},
        "Blocks": [{"BlockType": "PAGE"}, {"BlockType": "LINE", "Text": "Invoice 12345"}]
    }))
}

/// Recomputes SigV4 over the request the upstream received, at the time the client claimed.
fn expected_authorization(url: &str, sent: &Request) -> String {
    let format =
        format_description::parse_borrowed::<2>("[year][month][day]T[hour][minute][second]Z")
            .unwrap();
    let signed_at: SystemTime =
        PrimitiveDateTime::parse(sent.header("x-amz-date").unwrap(), &format)
            .unwrap()
            .assume_utc()
            .into();
    let headers: BTreeMap<String, String> = ["content-type", "x-amz-target"]
        .into_iter()
        .map(|name| (name.to_string(), sent.header(name).unwrap().to_string()))
        .collect();
    sign_post(
        url,
        &sent.body,
        &aws_signature_headers(&headers),
        "eu-west-1",
        "textract",
        &Credentials::new(ACCESS_KEY_ID, SECRET_ACCESS_KEY, None, None, "test"),
        signed_at,
    )
    .unwrap()["Authorization"]
        .clone()
}

/// The recorded URL names wiremock's host, not the address the client signed for.
fn assert_signed(upstream: &MockServer, sent: &Request) {
    let url = format!("{}/", upstream.uri());
    assert_eq!(
        sent.header("authorization"),
        Some(expected_authorization(&url, sent).as_str())
    );
}

#[tokio::test]
async fn detect_document_text_is_signed_and_lines_become_the_page() {
    let upstream = upstream([textract_response()]).await;

    let response = perform_with(LocalOcrHost::new(textract_request(DETECT, &upstream.uri())))
        .await
        .unwrap();

    let sent = only_request(&upstream).await;
    assert_eq!(
        sent.header("x-amz-target"),
        Some("Textract.DetectDocumentText")
    );
    assert_eq!(
        sent.header("content-type"),
        Some("application/x-amz-json-1.1")
    );
    assert_eq!(sent.json(), json!({"Document": {"Bytes": "b3JpZ2luYWw="}}));
    assert_signed(&upstream, &sent);
    assert_eq!(response.pages[0].markdown, "Invoice 12345");
    assert_eq!(response.usage_info.unwrap().pages_processed, Some(1));
}

#[tokio::test]
async fn a_body_rewritten_by_before_send_is_what_gets_signed_and_sent() {
    let upstream = upstream([textract_response()]).await;
    let host = LocalOcrHost::new(textract_request(DETECT, &upstream.uri())).with_before_send(
        |mut wire, _| {
            assert!(
                !wire
                    .headers
                    .iter()
                    .any(|(name, _)| name.eq_ignore_ascii_case("authorization")),
                "the hook ran after signing"
            );
            wire.body["Document"]["Bytes"] = Value::from("cmVkYWN0ZWQ=");
            Ok(wire)
        },
    );

    perform_with(host).await.unwrap();

    let sent = only_request(&upstream).await;
    assert_eq!(sent.json(), json!({"Document": {"Bytes": "cmVkYWN0ZWQ="}}));
    assert_signed(&upstream, &sent);
}

#[tokio::test]
async fn analyze_document_asks_for_layout_and_tables_and_returns_markdown() {
    let upstream = upstream([json_response(json!({
        "DocumentMetadata": {"Pages": 1},
        "Blocks": [
            {"Id": "l1", "BlockType": "LINE", "Text": "Quarterly Report"},
            {"Id": "t", "BlockType": "LAYOUT_TITLE",
                "Relationships": [{"Type": "CHILD", "Ids": ["l1"]}]}
        ]
    }))])
    .await;

    let response = perform_with(LocalOcrHost::new(textract_request(
        ANALYZE,
        &upstream.uri(),
    )))
    .await
    .unwrap();

    let sent = only_request(&upstream).await;
    assert_eq!(
        sent.header("x-amz-target"),
        Some("Textract.AnalyzeDocument")
    );
    assert_eq!(sent.json()["FeatureTypes"], json!(["LAYOUT", "TABLES"]));
    assert_signed(&upstream, &sent);
    assert_eq!(response.pages[0].markdown, "# Quarterly Report");
}

#[rstest]
#[case::detect(DETECT)]
#[case::analyze(ANALYZE)]
#[tokio::test]
async fn a_multi_page_rejection_reaches_the_caller_with_the_single_page_limit(#[case] model: &str) {
    let upstream = upstream([status_response(
        400,
        json!({
            "__type": "UnsupportedDocumentException",
            "Message": "Request has unsupported document format"
        }),
    )])
    .await;

    let error = perform_with(LocalOcrHost::new(textract_request(model, &upstream.uri())))
        .await
        .unwrap_err();

    let Error::Provider { status, body, .. } = error else {
        panic!("expected a provider error, got {error:?}");
    };
    assert_eq!(status, 400);
    assert!(
        body.contains("multi-page documents are not supported"),
        "{body}"
    );
}
