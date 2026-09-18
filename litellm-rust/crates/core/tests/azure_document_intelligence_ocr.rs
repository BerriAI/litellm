use litellm_callbacks::event::CallEvent;
use litellm_llms::base_llm::ocr::error::Error;
use rstest::rstest;
use serde_json::{Value, json};

use super::{
    test_support::{MockResponse, mock_server, perform_ocr, perform_ocr_with, wire_request},
    wire::{OcrWireRequest, decode_request},
};
use crate::ocr::route::LocalOcrHost;

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
    request.document =
        serde_json::from_value::<litellm_llms::base_llm::ocr::transformation::OcrDocument>(json!({
            "type":"document_url",
            "document_url":"https://example.com/document.pdf"
        }))
        .unwrap()
        .into();

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

#[rstest]
#[case(json!({"pages":[true]}), Error::Pages("expected only integers or only strings".into()))]
#[case(json!({"pages":[1,"2"]}), Error::Pages("expected only integers or only strings".into()))]
#[case(json!({"pages":[-1]}), Error::Pages("negative page index".into()))]
#[case(json!({"pages":"1&&features=bad"}), Error::Pages("invalid native page range".into()))]
#[case(json!({"features":"languages&pages=1"}), Error::Features)]
#[case(json!({"req_format":"azure"}), Error::RequestFormat)]
#[tokio::test]
async fn rejects_invalid_pages_features_and_format(
    #[case] options: Value,
    #[case] expected: Error,
) {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({}))]).await;
    let result = decode_request(OcrWireRequest {
        model: "azure_ai/doc-intelligence/prebuilt-read".into(),
        document: json!({"type":"document_url","document_url":"https://example.com/a.pdf"}),
        api_key: Some("key".into()),
        api_base: Some(base),
        custom_llm_provider: None,
        extra_headers: None,
        optional_params: options.as_object().unwrap().clone(),
        input_sources: Default::default(),
        timeout_seconds: Some(2.0),
    });
    let result = match result {
        Ok(request) => perform_ocr(request).await,
        Err(error) => Err(error),
    };
    server.abort();
    let _ = server.await;
    assert!(
        seen.lock().unwrap().is_empty(),
        "sent invalid options: {options}"
    );
    let error = result.unwrap_err();
    assert_eq!(
        std::mem::discriminant(&error),
        std::mem::discriminant(&expected)
    );
    assert_eq!(error.http_status_code(), Some(400));
    assert_eq!(error.to_string(), expected.to_string());
}

#[rstest]
#[case(json!({}))]
#[case(json!({"req_format":"litellm"}))]
#[tokio::test]
async fn missing_native_fields_keep_page_text_without_retaining_raw_response(
    #[case] options: Value,
) {
    let operation = json!({
        "status":"succeeded",
        "analyzeResult":{"pages":[{"pageNumber":1,"lines":[{"content":"hello"}]}]}
    });
    let (base, seen, server) = mock_server(vec![MockResponse::json(operation)]).await;
    let response = perform_ocr(wire_request(
        "azure_ai/doc-intelligence/prebuilt-read",
        &base,
        options,
    ))
    .await
    .unwrap();
    server.await.unwrap();

    assert_eq!(response.pages.len(), 1);
    assert_eq!(response.pages[0].index, 0);
    assert_eq!(response.pages[0].markdown, "hello");
    assert_eq!(response.provider_native_response, None);
    let serialized = response.into_json();
    assert_eq!(serialized.get("content"), Some(&Value::Null));
    assert_eq!(serialized.get("tables"), Some(&Value::Null));
    assert_eq!(serialized.get("keyValuePairs"), Some(&Value::Null));
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let target = requests[0].split_whitespace().nth(1).unwrap();
    let url = format!("{base}{target}");
    for field in ["pages", "features", "req_format"] {
        assert_eq!(query_value(&url, field), None);
    }
    let body: Value = serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body, json!({"base64Source":"YWJj"}));
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

    assert_eq!(result.pages[0].index, 1);
    assert_eq!(result.pages[0].markdown, "A\n\nB");
    assert_eq!(
        serde_json::to_value(&result.pages[0].dimensions).unwrap(),
        json!({"width":816,"height":1056,"dpi":96})
    );
    assert_eq!(result.usage_info.as_ref().unwrap().pages_processed, Some(1));
    let serialized = result.clone().into_json();
    assert_eq!(serialized["content"], "A\n\nB");
    assert_eq!(serialized["tables"], json!([{"cells":[]}]));
    assert_eq!(
        serialized["keyValuePairs"],
        json!([{"key":{"content":"A"}}])
    );
    assert!(serialized.get("key_value_pairs").is_none());
    assert_eq!(
        result.provider_native_response.map(Value::Object),
        Some(operation)
    );
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
        .transport
        .extra_headers
        .push(("X-Trace".into(), "initial-only".into()));

    let result = perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(
        result.provider_native_response.map(Value::Object),
        Some(operation)
    );
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

#[tokio::test]
async fn accepted_response_emits_response_received_before_polling() {
    let (base, seen, server) = mock_server(vec![
        MockResponse {
            status: 202,
            headers: vec![("Operation-Location", "{base}/operation".into())],
            body: json!({"submitted": true}),
        },
        MockResponse::json(json!({"status":"succeeded"})),
    ])
    .await;
    let request_count = seen.clone();
    let host = LocalOcrHost::new(wire_request(
        "azure_ai/doc-intelligence/prebuilt-read",
        &base,
        json!({}),
    ))
    .with_observer(move |event| {
        let CallEvent::ResponseReceived { raw } = event else {
            return;
        };
        match request_count.lock().unwrap().len() {
            1 => assert_eq!(raw.body, r#"{"submitted":true}"#),
            2 => assert!(raw.body.contains("succeeded")),
            count => panic!("unexpected callback after {count} requests"),
        }
    });

    perform_ocr_with(host).await.unwrap();
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
    request.credentials.api_key = None;
    request.transport.extra_headers = vec![("Authorization".into(), "Bearer token".into())];

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
    request.transport.poll_timeout = std::time::Duration::from_millis(100);

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

mod transformation {
    use std::sync::{Arc, Mutex};

    use litellm_callbacks::event::CallEvent;
    use litellm_llms::base_llm::ocr::transformation::OcrDocument;
    use serde_json::{Value, json};

    use super::*;
    use crate::ocr::{
        route::LocalOcrHost,
        test_support::{MockResponse, mock_server, perform_ocr, perform_ocr_with, wire_request},
    };

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
            json!({"pages":[2,0,0,1],"features":["keyValuePairs","languages"], "future_option": {"nested":null}, "extra_body":{"provider_option":false}}),
        );
        request.document = serde_json::from_value::<OcrDocument>(json!({
            "type":"document_url",
            "document_url":"https://example.com/document.pdf"
        }))
        .unwrap()
        .into();

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
            json!({"urlSource":"https://example.com/document.pdf", "future_option":{"nested":null}, "provider_option":false})
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
            let request = wire_request(
                "azure_ai/doc-intelligence/prebuilt-read",
                "http://127.0.0.1:1",
                options.clone(),
            );
            let rejected = perform_ocr(request).await.is_err();
            assert!(rejected, "accepted {options}");
        }
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

        assert_eq!(result.pages[0].index, 1);
        assert_eq!(result.pages[0].markdown, "A\n\nB");
        assert_eq!(
            serde_json::to_value(&result.pages[0].dimensions).unwrap(),
            json!({"width":816,"height":1056,"dpi":96})
        );
        assert_eq!(result.usage_info.as_ref().unwrap().pages_processed, Some(1));
        let serialized = result.clone().into_json();
        assert_eq!(serialized["content"], "A\n\nB");
        assert_eq!(serialized["tables"], json!([{"cells":[]}]));
        assert_eq!(
            serialized["keyValuePairs"],
            json!([{"key":{"content":"A"}}])
        );
        assert!(serialized.get("key_value_pairs").is_none());
        assert_eq!(
            result.provider_native_response.as_ref(),
            operation.as_object()
        );
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
            .transport
            .extra_headers
            .push(("X-Trace".into(), "initial-only".into()));

        let result = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(
            result.provider_native_response.as_ref(),
            operation.as_object()
        );
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

    #[tokio::test]
    async fn accepted_response_emits_response_received_for_submission_and_completed_poll() {
        let (base, seen, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({"submitted": true}),
            },
            MockResponse::json(json!({"status":"succeeded"})),
        ])
        .await;
        let responses_received = Arc::new(Mutex::new(Vec::new()));
        let request_count = seen.clone();
        let observed = responses_received.clone();
        let host = LocalOcrHost::new(wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({}),
        ))
        .with_observer(move |event| {
            if let CallEvent::ResponseReceived { raw } = event {
                observed
                    .lock()
                    .unwrap()
                    .push((request_count.lock().unwrap().len(), raw.body.clone()));
            }
        });

        perform_ocr_with(host).await.unwrap();
        server.await.unwrap();
        assert_eq!(seen.lock().unwrap().len(), 2);
        assert_eq!(
            *responses_received.lock().unwrap(),
            [
                (1, r#"{"submitted":true}"#.to_string()),
                (2, r#"{"status":"succeeded"}"#.to_string()),
            ]
        );
    }
}
