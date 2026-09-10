use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::OcrClient;
use super::test_support::{MockResponse, mock_server, perform_ocr, wire_request};
use super::wire::{OcrWireRequest, decode_request, decode_response};

#[test]
fn provider_options_are_decoded_only_for_the_selected_backend() {
    let mistral = decode_request(OcrWireRequest {
        model: "mistral/model".into(),
        document: json!({"type":"document_url","document_url":"data:application/pdf;base64,YWJj"}),
        api_key: Some("key".into()),
        api_base: None,
        custom_llm_provider: None,
        extra_headers: None,
        optional_params: json!({"vertex_project":42}).as_object().unwrap().clone(),
        timeout_seconds: None,
    });
    assert!(mistral.is_ok());
}

#[rstest::rstest]
#[case("mistral/model", "/v1/ocr")]
#[case("azure_ai/model", "/providers/mistral/azure/ocr")]
#[tokio::test]
async fn performs_mistral_ocr_across_backends(#[case] model: &str, #[case] path: &str) {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[{"index":0,"markdown":"hello","custom":"preserved"}],"usage_info":{"pages_processed":1}}))]).await;
    let result = perform_ocr(wire_request(
        model,
        &base,
        json!({"extract_header":true,"unknown":"ignored"}),
    ))
    .await
    .unwrap();
    server.await.unwrap();
    assert_eq!(result.pages[0].markdown, "hello");
    assert_eq!(result.pages[0].extra_fields["custom"], "preserved");
    let request = seen.lock().unwrap();
    assert!(request[0].starts_with(&format!("POST {path} ")));
    assert!(
        request[0]
            .to_ascii_lowercase()
            .contains("authorization: bearer test-key\r\n")
    );
    let body: Value = serde_json::from_str(request[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "model",
            "document": {"type":"document_url", "document_url":"data:application/pdf;base64,YWJj"},
            "extract_header": true
        })
    );
    assert_eq!(result.model, "model");
    assert_eq!(result.usage_info.unwrap().pages_processed, Some(1));
}

#[tokio::test]
async fn uses_the_host_injected_http_client() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(
        json!({"pages":[],"usage_info":{"pages_processed":0}}),
    )])
    .await;
    let mut default_headers = reqwest::header::HeaderMap::new();
    default_headers.insert(
        "x-transport-owner",
        reqwest::header::HeaderValue::from_static("python-sdk"),
    );
    let provider_http = reqwest::Client::builder()
        .default_headers(default_headers)
        .build()
        .expect("test HTTP client builds");
    let client = OcrClient::new(provider_http).expect("test OCR client builds");

    client
        .perform(wire_request("mistral/model", &base, json!({})))
        .await
        .expect("OCR request succeeds");

    server.await.expect("mock server completes");
    assert!(seen.lock().unwrap()[0].contains("x-transport-owner: python-sdk"));
}

#[tokio::test]
async fn invalid_pages_fail_before_network_and_invoke_failure_hook() {
    use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleTiming};
    use crate::ocr::hooks::{OcrHooks, OcrLogFuture};
    struct Failures(Arc<Mutex<Vec<String>>>);
    impl OcrHooks for Failures {
        fn failure<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            error: &'a crate::Error,
            _timing: &'a CallLifecycleTiming,
        ) -> OcrLogFuture<'a> {
            Box::pin(async move {
                self.0.lock().unwrap().push(error.to_string());
            })
        }
    }
    let failures = Arc::new(Mutex::new(Vec::new()));
    let mut request = wire_request(
        "azure_ai/doc-intelligence/prebuilt-read",
        "http://127.0.0.1:1",
        json!({"pages":[-1]}),
    );
    request.hooks = Arc::new(Failures(failures.clone()));
    assert_eq!(
        perform_ocr(request).await.unwrap_err().kind(),
        crate::error::ErrorKind::InvalidRequest
    );
    assert_eq!(failures.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn pre_call_hook_params_use_the_same_validation_as_wire_params() {
    use crate::ocr::hooks::{OcrHookFuture, OcrHooks, OcrPreCallRequest};

    struct InvalidPages;
    impl OcrHooks for InvalidPages {
        fn has_guardrails(&self) -> bool {
            true
        }

        fn pre_call(&self, request: OcrPreCallRequest) -> OcrHookFuture<'_, OcrPreCallRequest> {
            Box::pin(async move {
                Ok(OcrPreCallRequest {
                    optional_params: json!({"pages":[true]}),
                    ..request
                })
            })
        }
    }

    let mut request = wire_request(
        "azure_ai/doc-intelligence/prebuilt-read",
        "http://127.0.0.1:1",
        json!({}),
    );
    request.hooks = Arc::new(InvalidPages);
    let error = perform_ocr(request).await.unwrap_err();
    assert_eq!(error.kind(), crate::error::ErrorKind::InvalidRequest);
    assert!(error.to_string().contains("integers, not booleans"));
}

#[test]
fn typed_response_rejects_malformed_pages_with_field_path() {
    let error = decode_response::<crate::ocr::formats::mistral::types::MistralOcrResponse>(
        br#"{"pages":[{"index":0,"markdown":42}]}"#,
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("pages[0].markdown"));
    assert!(!error.to_string().contains("42"));
}

#[rstest::rstest]
#[case("mistral/model", json!({"model":"model","document":42}), "guardrail.body.document")]
#[case("azure_ai/model", json!({"model":"model","document":{"type":"document_url","document_url":"https://example.com/doc.pdf"}}), "data URI")]
#[tokio::test]
async fn rejects_invalid_guardrail_body_before_sending(
    #[case] model: &str,
    #[case] body: Value,
    #[case] expected: &str,
) {
    use crate::ocr::hooks::{OcrDuringCallRequest, OcrHookFuture, OcrHooks};

    struct Rewrite(Value);
    impl OcrHooks for Rewrite {
        fn has_guardrails(&self) -> bool {
            true
        }

        fn during_call(
            &self,
            request: OcrDuringCallRequest,
        ) -> OcrHookFuture<'_, OcrDuringCallRequest> {
            Box::pin(async move {
                Ok(OcrDuringCallRequest {
                    body: self.0.clone(),
                    ..request
                })
            })
        }
    }

    let mut request = wire_request(model, "http://127.0.0.1:1", json!({}));
    request.hooks = Arc::new(Rewrite(body));
    let error = perform_ocr(request).await.unwrap_err();
    assert_eq!(error.kind(), crate::error::ErrorKind::InvalidRequest);
    assert!(error.to_string().contains(expected), "{error}");
}
