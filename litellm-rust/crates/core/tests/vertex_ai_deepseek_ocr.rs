use serde_json::{Value, json};

use super::test_support::{MockResponse, mock_server, perform_ocr, wire_request};
use crate::auth::InputSource;

fn request_body(request: &str) -> Value {
    serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
}

#[tokio::test]
async fn facade_executes_vertex_deepseek_at_the_openai_endpoint() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "choices":[{"message":{"content":"recognized"}}],
        "usage":{"prompt_tokens":1}
    }))])
    .await;
    let mut request = wire_request(
        "vertex_ai/deepseek-ocr-maas",
        &base,
        json!({
            "vertex_project":"project-1",
            "vertex_location":"europe-west4",
            "temperature":0.1,
            "future_ocr_option":true,
            "extra_body":{"provider_option":"value"}
        }),
    );
    request.document = request
        .document
        .with_source("gs://bucket/document.pdf".into());

    let response = perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(response.pages[0]["markdown"], "recognized");
    assert_eq!(response.usage_info.unwrap()["prompt_tokens"], 1);
    let requests = seen.lock().unwrap();
    assert!(requests[0].starts_with(
        "POST /v1/projects/project-1/locations/europe-west4/endpoints/openapi/chat/completions "
    ));
    assert!(
        requests[0]
            .to_ascii_lowercase()
            .contains("authorization: bearer test-key")
    );
    let body = request_body(&requests[0]);
    assert_eq!(body["model"], "deepseek-ai/deepseek-ocr-maas");
    assert_eq!(body["temperature"], 0.1);
    assert!(body.get("future_ocr_option").is_none());
    assert!(body.get("extra_body").is_none());
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({"type":"image_url","image_url":"gs://bucket/document.pdf"})
    );
}

#[test]
fn host_registration_selects_deepseek_without_affecting_mistral() {
    assert!(crate::ocr::wire::is_supported_request(
        "deepseek-ocr-maas",
        Some("vertex_ai")
    ));
    assert!(crate::ocr::wire::is_supported_request(
        "mistral-ocr-maas",
        Some("vertex_ai")
    ));
}

#[tokio::test]
async fn request_controlled_api_base_is_rejected_before_vertex_auth() {
    let mut request = wire_request(
        "vertex_ai/deepseek-ocr-maas",
        "https://caller.example",
        json!({"vertex_project":"project-1"}),
    );
    request.connection.api_base_source = InputSource::Request;

    let error = perform_ocr(request).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("request-controlled Vertex AI endpoint")
    );
}
