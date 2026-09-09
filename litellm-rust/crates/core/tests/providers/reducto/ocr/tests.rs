use super::transformation::{REDUCTO_PARSE_LEGACY_CONFIG as LEGACY, REDUCTO_PARSE_V3_CONFIG as V3};
use crate::ocr::perform_ocr;
use crate::ocr::tests::{MockResponse, body, mock_server, transform, wire_request};
use rstest::rstest;
use serde_json::json;

#[tokio::test]
async fn test_parse_v3_reducto_id_passthrough_skips_upload() {
    let result=body(&V3,"parse-v3",json!({"type":"document_url","document_url":"reducto://already.pdf"}),json!({"formatting":{"table_output_format":"html"},"retrieval":{"chunk_mode":"section"},"settings":{"ocr_system":"standard"}})).await.unwrap();
    assert_eq!(result["input"], "reducto://already.pdf");
    assert_eq!(result["formatting"]["table_output_format"], "html");
}
#[tokio::test]
async fn test_parse_legacy_wraps_enhance_under_options() {
    let doc = json!({"type":"document_url","document_url":"reducto://legacy.pdf"});
    let result = body(
        &LEGACY,
        "parse-legacy",
        doc.clone(),
        json!({"enhance":{"agentic":[{"type":"table"}]}}),
    )
    .await
    .unwrap();
    assert_eq!(
        result,
        json!({"document_url":"reducto://legacy.pdf","options":{"enhance":{"agentic":[{"type":"table"}]}}})
    );
    assert!(
        body(&LEGACY, "parse-legacy", doc, json!({}))
            .await
            .unwrap()
            .get("options")
            .is_none()
    );
}
#[rstest]
#[case("http://example.com/a.pdf")]
#[case("https://example.com/a.pdf")]
#[case("data:application/pdf,YWJj")]
#[case("data:application/pdf;base64,INVALID!")]
#[tokio::test]
async fn test_parse_v3_rejects_plain_http_urls(#[case] source: &str) {
    assert!(
        body(
            &V3,
            "parse-v3",
            json!({"type":"document_url","document_url":source}),
            json!({})
        )
        .await
        .is_err()
    );
}
#[test]
fn reducto_groups_blocks_and_preserves_native_response() {
    let raw = json!({"job_id":"job-1","usage":{"num_pages":2,"credits":3},"result":{"chunks":[
        {"blocks":[{"content":"B","bbox":{"page":2},"kind":"table"}]},
        {"blocks":[{"content":"A","bbox":{"page":1},"kind":"text"},{"content":"C","bbox":{"page":1},"kind":"text"}]}]}});
    let response = transform(&V3, "parse-v3", raw.clone(), json!({})).unwrap();
    assert_eq!(response["pages"][0]["markdown"], "A\n\nC");
    assert_eq!(response["pages"][1]["markdown"], "B");
    assert_eq!(response["pages"][1]["blocks"][0]["kind"], "table");
    assert_eq!(response["provider_native_response"], raw);
    assert_eq!(response["usage_info"]["credits"], 3.0);
}
#[test]
fn reducto_missing_result_and_null_result_are_distinct() {
    let flat = transform(
        &V3,
        "parse-v3",
        json!({"chunks":[{"content":"text"}]}),
        json!({}),
    )
    .unwrap();
    assert_eq!(flat["pages"][0]["markdown"], "text");
    let null = transform(
        &V3,
        "parse-v3",
        json!({"result":null,"chunks":[{"content":"ignored"}]}),
        json!({}),
    )
    .unwrap();
    assert_eq!(null["pages"], json!([]));
}
#[rstest]
#[case("parse-v3")]
#[case("parse-legacy")]
#[tokio::test]
async fn test_parse_v3_file_upload_and_response_mapping(#[case] model: &str) {
    let (base, seen, server) = mock_server(vec![
        MockResponse::json(json!({"file_id":"reducto://uploaded.pdf"})),
        MockResponse::json(
            json!({"result":{"chunks":[{"content":"hello"}]},"usage":{"num_pages":1}}),
        ),
    ])
    .await;
    let response = perform_ocr(wire_request(&format!("reducto/{model}"), &base, json!({})))
        .await
        .unwrap();
    server.await.unwrap();
    assert_eq!(response.pages[0].markdown, "hello");
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert!(seen[0].starts_with("POST /upload "));
    assert!(seen[0].contains("application/pdf"));
    assert!(seen[0].contains("abc"));
    assert!(seen[1].starts_with("POST /parse "));
    assert!(seen[1].contains("reducto://uploaded.pdf"));
}
