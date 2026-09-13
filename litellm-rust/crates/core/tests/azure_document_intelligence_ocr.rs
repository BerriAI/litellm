use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

use super::test_support::{MockResponse, mock_server, perform_ocr, wire_request};
use super::wire::{OcrWireRequest, decode_request};

fn query_value(url: &str, key: &str) -> Option<String> {
    url::Url::parse(url)
        .unwrap()
        .query_pairs()
        .find_map(|(name, value)| (name == key).then(|| value.into_owned()))
}

#[tokio::test]
async fn facade_maps_pages_features_and_url_document() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "status":"succeeded",
        "analyzeResult":{"pages":[]}
    }))])
    .await;
    let mut request = wire_request(
        "azure_ai/doc-intelligence/prebuilt-read",
        &base,
        json!({"pages":[2,0,0,1],"features":["keyValuePairs","languages"]}),
    );
    request.document = serde_json::from_value(json!({
        "type":"document_url",
        "document_url":"https://example.com/document.pdf"
    }))
    .unwrap();

    perform_ocr(request).await.unwrap();
    server.await.unwrap();
    let request = &seen.lock().unwrap()[0];
    let target = request.split_whitespace().nth(1).unwrap();
    let url = format!("{base}{target}");
    assert_eq!(query_value(&url, "pages").as_deref(), Some("1,2,3"));
    assert_eq!(
        query_value(&url, "features").as_deref(),
        Some("keyValuePairs,languages")
    );
    let body: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        body,
        json!({"urlSource":"https://example.com/document.pdf"})
    );
}

#[tokio::test]
async fn rejects_invalid_pages_features_and_format() {
    for options in [
        json!({"pages":[true]}),
        json!({"pages":[1,"2"]}),
        json!({"pages":[-1]}),
        json!({"pages":"1&&features=bad"}),
        json!({"features":"languages&pages=1"}),
        json!({"req_format":"azure"}),
    ] {
        let result = decode_request(OcrWireRequest {
            model: "azure_ai/doc-intelligence/prebuilt-read".into(),
            document: json!({"type":"document_url","document_url":"https://example.com/a.pdf"}),
            api_key: Some("key".into()),
            api_base: Some("http://127.0.0.1:1".into()),
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: options.as_object().unwrap().clone(),
            input_sources: Default::default(),
            timeout_seconds: None,
        });
        let rejected = match result {
            Ok(request) => perform_ocr(request).await.is_err(),
            Err(_) => true,
        };
        assert!(rejected, "accepted {options}");
    }
}

#[tokio::test]
async fn inline_document_decodes_to_base64_source() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "status":"succeeded"
    }))])
    .await;
    let request = wire_request("azure_ai/doc-intelligence/prebuilt-read", &base, json!({}));

    perform_ocr(request).await.unwrap();
    server.await.unwrap();
    let request = &seen.lock().unwrap()[0];
    let body: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body, json!({"base64Source":"YWJj"}));
}

#[tokio::test]
async fn immediate_response_normalizes_pages_and_preserves_native() {
    let operation = json!({
        "status":"succeeded",
        "operationExtension":42,
        "analyzeResult":{
            "content":"A\n\nB",
            "tables":[{"cells":[]}],
            "keyValuePairs":[{"key":{"content":"A"}}],
            "pages":[{
                "pageNumber":"2",
                "width":"8.5",
                "height":11,
                "unit":"inch",
                "lines":[{"content":"A"},{"content":null},{"content":"B"}]
            }]
        }
    });
    let (base, _, server) = mock_server(vec![MockResponse::json(operation.clone())]).await;
    let result = perform_ocr(wire_request(
        "azure_ai/doc-intelligence/prebuilt-read",
        &base,
        json!({"req_format":"native"}),
    ))
    .await
    .unwrap();
    server.await.unwrap();

    assert_eq!(result.pages[0]["index"], 1);
    assert_eq!(result.pages[0]["markdown"], "A\n\nB");
    assert_eq!(
        result.pages[0]["dimensions"],
        json!({"width":816,"height":1056,"dpi":96})
    );
    assert_eq!(result.usage_info, Some(json!({"pages_processed":1})));
    let serialized = result.clone().into_json();
    assert_eq!(serialized["content"], "A\n\nB");
    assert_eq!(serialized["tables"], json!([{"cells":[]}]));
    assert_eq!(
        serialized["keyValuePairs"],
        json!([{"key":{"content":"A"}}])
    );
    assert!(serialized.get("key_value_pairs").is_none());
    assert_eq!(result.provider_native_response, Some(operation));
}

#[tokio::test]
async fn accepted_response_polls_to_success_with_only_credentials() {
    let operation = json!({"status":"succeeded","analyzeResult":{"pages":[]}});
    let (base, seen, server) = mock_server(vec![
        MockResponse {
            status: 202,
            headers: vec![("Operation-Location", "{base}/operation".into())],
            body: json!({}),
        },
        MockResponse {
            status: 200,
            headers: vec![("Retry-After", "0".into())],
            body: json!({"status":"running"}),
        },
        MockResponse::json(operation.clone()),
    ])
    .await;
    let mut request = wire_request(
        "azure_ai/doc-intelligence/prebuilt-read",
        &base,
        json!({"req_format":"native"}),
    );
    request
        .connection
        .extra_headers
        .push(("X-Trace".into(), "initial-only".into()));

    let result = perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(result.provider_native_response, Some(operation));
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(requests[0].to_ascii_lowercase().contains("x-trace:"));
    for poll in &requests[1..] {
        assert!(!poll.to_ascii_lowercase().contains("x-trace:"));
        assert!(
            poll.to_ascii_lowercase()
                .contains("ocp-apim-subscription-key: test-key")
        );
    }
}

struct SubmissionBoundary {
    request_count: Arc<Mutex<Vec<String>>>,
}

impl super::hooks::OcrHooks for SubmissionBoundary {
    fn post_call(
        &self,
        request: super::hooks::OcrPostCallRequest,
    ) -> super::hooks::OcrHookFuture<'_, super::hooks::OcrPostCallRequest> {
        Box::pin(async move {
            match self.request_count.lock().unwrap().len() {
                1 => assert_eq!(request.original_response, json!(r#"{"submitted":true}"#)),
                2 => assert!(
                    request
                        .original_response
                        .as_str()
                        .unwrap()
                        .contains("succeeded")
                ),
                count => panic!("unexpected callback after {count} requests"),
            }
            Ok(request)
        })
    }
}

#[tokio::test]
async fn accepted_response_runs_post_call_before_polling() {
    let (base, seen, server) = mock_server(vec![
        MockResponse {
            status: 202,
            headers: vec![("Operation-Location", "{base}/operation".into())],
            body: json!({"submitted": true}),
        },
        MockResponse::json(json!({"status":"succeeded"})),
    ])
    .await;
    let request = super::LiteLLMOcrRequest {
        hooks: Arc::new(SubmissionBoundary {
            request_count: seen.clone(),
        }),
        ..wire_request("azure_ai/doc-intelligence/prebuilt-read", &base, json!({}))
    };

    perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn polling_forwards_bearer_credentials() {
    let (base, seen, server) = mock_server(vec![
        MockResponse {
            status: 202,
            headers: vec![("Operation-Location", "{base}/operation".into())],
            body: json!({}),
        },
        MockResponse::json(json!({"status":"succeeded"})),
    ])
    .await;
    let mut request = wire_request("azure_ai/doc-intelligence/prebuilt-read", &base, json!({}));
    request.connection.api_key = None;
    request.connection.extra_headers = vec![("Authorization".into(), "Bearer token".into())];

    perform_ocr(request).await.unwrap();
    server.await.unwrap();
    let requests = seen.lock().unwrap();
    assert!(
        requests[1]
            .to_ascii_lowercase()
            .contains("authorization: bearer token")
    );
}

#[tokio::test]
async fn polling_does_not_follow_redirects() {
    let (base, seen, server) = mock_server(vec![
        MockResponse {
            status: 202,
            headers: vec![("Operation-Location", "{base}/operation".into())],
            body: json!({}),
        },
        MockResponse {
            status: 302,
            headers: vec![("Location", "{base}/redirected".into())],
            body: json!({}),
        },
        MockResponse::json(json!({"status":"succeeded"})),
    ])
    .await;

    let error = perform_ocr(wire_request(
        "azure_ai/doc-intelligence/prebuilt-read",
        &base,
        json!({}),
    ))
    .await
    .unwrap_err();

    assert!(error.to_string().contains("status 302"), "{error}");
    assert_eq!(seen.lock().unwrap().len(), 2);
    server.abort();
}

#[tokio::test]
async fn polling_rejects_terminal_failure() {
    let (base, _, server) = mock_server(vec![
        MockResponse {
            status: 202,
            headers: vec![("Operation-Location", "{base}/operation".into())],
            body: json!({}),
        },
        MockResponse::json(json!({"status":"failed"})),
    ])
    .await;

    let error = perform_ocr(wire_request(
        "azure_ai/doc-intelligence/prebuilt-read",
        &base,
        json!({}),
    ))
    .await
    .unwrap_err();
    server.await.unwrap();
    assert!(error.to_string().contains("status failed"));
}

#[tokio::test]
async fn malformed_provider_pages_report_response_paths() {
    for (analysis, path) in [
        (json!({"pages":null}), "pages"),
        (json!({"pages":[null]}), "pages[0]"),
        (json!({"pages":[{"lines":null}]}), "lines"),
        (json!({"pages":[{"width":"bad"}]}), "width"),
    ] {
        let (base, _, server) = mock_server(vec![MockResponse::json(json!({
            "status":"succeeded",
            "analyzeResult":analysis
        }))])
        .await;
        let error = perform_ocr(wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({}),
        ))
        .await
        .unwrap_err();
        server.await.unwrap();
        assert!(error.to_string().contains(path), "{error}");
    }
}

#[tokio::test]
async fn rejects_missing_invalid_and_cross_origin_operation_locations() {
    for headers in [
        Vec::new(),
        vec![("Operation-Location", "/relative".into())],
        vec![("Operation-Location", "http://example.com/operation".into())],
        vec![(
            "Operation-Location",
            "http://user:password@127.0.0.1/operation".into(),
        )],
    ] {
        let (base, _, server) = mock_server(vec![MockResponse {
            status: 202,
            headers,
            body: json!({}),
        }])
        .await;
        let error = perform_ocr(wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({}),
        ))
        .await
        .unwrap_err();
        server.await.unwrap();
        assert!(error.to_string().contains("operation-location"));
    }
}

#[tokio::test]
async fn polling_deadline_bounds_retry_delay() {
    let (base, _, server) = mock_server(vec![
        MockResponse {
            status: 202,
            headers: vec![("Operation-Location", "{base}/operation".into())],
            body: json!({}),
        },
        MockResponse {
            status: 200,
            headers: vec![("Retry-After", "9999".into())],
            body: json!({"status":"notStarted"}),
        },
    ])
    .await;
    let mut request = wire_request("azure_ai/doc-intelligence/prebuilt-read", &base, json!({}));
    request.connection.poll_timeout = std::time::Duration::from_millis(100);

    let error = tokio::time::timeout(std::time::Duration::from_secs(1), perform_ocr(request))
        .await
        .unwrap()
        .unwrap_err();
    server.await.unwrap();
    assert!(error.to_string().contains("timed out"));
}

#[tokio::test]
async fn model_id_is_encoded_and_dot_segments_are_rejected() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "status":"succeeded"
    }))])
    .await;
    perform_ocr(wire_request(
        "azure_ai/doc-intelligence/a ?#é",
        &base,
        json!({}),
    ))
    .await
    .unwrap();
    server.await.unwrap();
    assert!(seen.lock().unwrap()[0].contains("a%20%3F%23%C3%A9:analyze"));

    for model in [
        "azure_ai/doc-intelligence/.",
        "azure_ai/doc-intelligence/..",
    ] {
        let error = perform_ocr(wire_request(model, "http://127.0.0.1:1", json!({})))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("dot segment"));
    }
}

#[tokio::test]
async fn pre_call_guardrail_receives_caller_pages_before_mapping() {
    use crate::ocr::hooks::{OcrHookFuture, OcrHooks, OcrPreCallRequest};
    use std::sync::Arc;

    struct RewritePages;
    impl OcrHooks for RewritePages {
        fn intercepts_requests(&self) -> bool {
            true
        }

        fn pre_call(&self, request: OcrPreCallRequest) -> OcrHookFuture<'_, OcrPreCallRequest> {
            Box::pin(async move {
                assert_eq!(request.optional_params["pages"], json!([0, 2]));
                Ok(OcrPreCallRequest {
                    optional_params: json!({"pages": [1]}),
                    ..request
                })
            })
        }
    }
    let (base, seen, server) =
        mock_server(vec![MockResponse::json(json!({"status": "succeeded"}))]).await;
    let request = wire_request(
        "azure_ai/doc-intelligence/prebuilt-read",
        &base,
        json!({"pages": [0, 2]}),
    )
    .with_host_hooks(Arc::new(RewritePages), None);
    perform_ocr(request).await.unwrap();
    server.await.unwrap();
    let requests = seen.lock().unwrap();
    let target = requests[0].split_whitespace().nth(1).unwrap();
    assert_eq!(
        query_value(&format!("{base}{target}"), "pages").as_deref(),
        Some("2")
    );
    assert_eq!(requests.len(), 1);
}
