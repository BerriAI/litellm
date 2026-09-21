use serde_json::{Value, json};

use super::test_support::{MockResponse, mock_server, perform_ocr, wire_request};
use crate::auth::InputSource;

fn request_body(request: &str) -> Value {
    serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
}

#[tokio::test]
async fn facade_executes_vertex_mistral_with_resolved_project_and_location() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "pages":[{"index":0,"markdown":"hello"}],
        "usage_info":{"pages_processed":1}
    }))])
    .await;
    let request = wire_request(
        "vertex_ai/mistral-ocr-maas",
        &base,
        json!({
            "vertex_project":"project-1",
            "vertex_location":"europe-west4",
            "extract_footer":true
        }),
    );

    let response = perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(response.pages[0]["markdown"], "hello");
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].starts_with(
        "POST /v1/projects/project-1/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict "
    ));
    assert!(
        requests[0]
            .to_ascii_lowercase()
            .contains("authorization: bearer test-key")
    );
    assert_eq!(
        request_body(&requests[0]),
        json!({
            "model":"mistral-ocr-maas",
            "document":{"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
            "extract_footer":true
        })
    );
}

#[tokio::test]
async fn supplied_authorization_is_forwarded_without_a_static_token() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let mut request = wire_request(
        "vertex_ai/model",
        &base,
        json!({"vertex_project":"project-1"}),
    );
    request.connection.api_key = None;
    request.connection.extra_headers = vec![("authorization".into(), "Bearer supplied".into())];

    perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert!(
        seen.lock().unwrap()[0]
            .to_ascii_lowercase()
            .contains("authorization: bearer supplied")
    );
}

#[tokio::test]
async fn invalid_credentials_fail_before_provider_http() {
    let request = wire_request(
        "vertex_ai/model",
        "http://127.0.0.1:1",
        json!({"vertex_credentials": true}),
    );
    let error = perform_ocr(request).await.unwrap_err();
    assert!(error.to_string().contains("vertex_credentials"));
}

#[tokio::test]
async fn request_controlled_api_base_is_rejected_before_vertex_auth() {
    let mut request = wire_request(
        "vertex_ai/mistral-ocr-maas",
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

#[tokio::test]
async fn adapters_build_complete_requests_and_share_mistral_normalization() {
    use std::time::Duration;

    use crate::ocr::adapters::{MistralAdapter, OcrAdapter, VertexMistralAdapter};
    use crate::ocr::test_support::ocr_client;

    let client = ocr_client();
    let options = json!({
        "pages": [0, 2],
        "include_image_base64": true,
        "vertex_project": "project-1",
        "vertex_location": "us-central1",
        "unknown": "ignored"
    });
    let direct = wire_request(
        "mistral/mistral-ocr-maas",
        "https://mistral.test",
        options.clone(),
    );
    let vertex = wire_request("vertex_ai/mistral-ocr-maas", "https://vertex.test", options);
    let direct_http = MistralAdapter
        .prepare_request(&direct, &client)
        .await
        .unwrap();
    let vertex_http = VertexMistralAdapter
        .prepare_request(&vertex, &client)
        .await
        .unwrap();
    assert_eq!(direct_http.url().as_str(), "https://mistral.test/v1/ocr");
    assert_eq!(
        vertex_http.url().as_str(),
        "https://vertex.test/v1/projects/project-1/locations/us-central1/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
    );
    for http in [&direct_http, &vertex_http] {
        assert_eq!(http.method(), reqwest::Method::POST);
        assert_eq!(http.headers()["authorization"], "Bearer test-key");
        assert_eq!(http.headers()["content-type"], "application/json");
        assert_eq!(http.timeout(), Some(&Duration::from_secs(2)));
        let body: Value = serde_json::from_slice(http.body().unwrap().as_bytes().unwrap()).unwrap();
        assert_eq!(
            body,
            json!({
                "model": "mistral-ocr-maas",
                "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                "pages": [0, 2],
                "include_image_base64": true
            })
        );
    }
    let payload = json!({"pages": [{"index": 0, "markdown": "hello"}], "extra": "preserved"});
    let direct_response = MistralAdapter
        .transform_ocr_response(&direct, serde_json::from_value(payload.clone()).unwrap())
        .unwrap()
        .into_json();
    let vertex_response = VertexMistralAdapter
        .transform_ocr_response(&vertex, serde_json::from_value(payload).unwrap())
        .unwrap()
        .into_json();
    assert_eq!(direct_response, vertex_response);
    assert_eq!(direct_response["model"], "mistral-ocr-maas");
    assert_eq!(direct_response["object"], "ocr");
    assert_eq!(direct_response["extra"], "preserved");
}
