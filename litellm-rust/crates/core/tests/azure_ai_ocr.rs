use std::sync::Arc;

use serde_json::{Value, json};

use super::hooks::{OcrDuringCallRequest, OcrHookFuture, OcrHooks};
use super::test_support::{MockResponse, mock_server, perform_ocr, wire_request};

#[tokio::test]
async fn facade_executes_azure_mistral_with_prepared_auth() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "pages":[{"index":0,"markdown":"hello"}],
        "usage_info":{"pages_processed":1}
    }))])
    .await;
    let mut request = wire_request(
        "azure_ai/model",
        &base,
        json!({"include_image_base64":true}),
    );
    request.connection.api_key = None;
    request.connection.extra_headers = vec![(
        "Authorization".into(),
        "Bearer python-prepared-token".into(),
    )];

    let result = perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(result.pages[0]["markdown"], "hello");
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].starts_with("POST /providers/mistral/azure/ocr "));
    assert!(
        requests[0]
            .to_ascii_lowercase()
            .contains("authorization: bearer python-prepared-token\r\n")
    );
    let body: Value = serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        body,
        json!({
            "model":"model",
            "document":{"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
            "include_image_base64":true
        })
    );
}

struct ReplaceBodyDocument;

impl OcrHooks for ReplaceBodyDocument {
    fn has_guardrails(&self) -> bool {
        true
    }

    fn during_call(
        &self,
        mut request: OcrDuringCallRequest,
    ) -> OcrHookFuture<'_, OcrDuringCallRequest> {
        Box::pin(async move {
            request.body["document"] = json!({
                "type":"document_url",
                "document_url":"https://example.com/not-inline.pdf"
            });
            Ok(request)
        })
    }
}

#[tokio::test]
async fn rejects_non_inline_body_after_guardrails() {
    let mut request = wire_request("azure_ai/model", "http://127.0.0.1:1", json!({}));
    request.hooks = Arc::new(ReplaceBodyDocument);
    let error = perform_ocr(request).await.unwrap_err();
    assert!(error.to_string().contains("data URI"));
}
