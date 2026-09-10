use serde_json::{Value, json};

use super::test_support::{MockResponse, mock_server, perform_ocr, wire_request};

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

#[test]
fn invalid_credentials_fail_during_request_decoding() {
    assert!(
        crate::ocr::wire::decode_request(crate::ocr::wire::OcrWireRequest {
            model: "vertex_ai/model".into(),
            document: json!({"type":"document_url","document_url":"data:application/pdf;base64,YWJj"}),
            api_key: None,
            api_base: Some("http://127.0.0.1:1".into()),
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: json!({"vertex_credentials":true}).as_object().unwrap().clone(),
            timeout_seconds: Some(1.0),
        })
        .is_err()
    );
}
