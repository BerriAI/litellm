use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_host::event::{CallEvent, MachineEvent};
use litellm_llms::base_llm::ocr::settings::OcrSettings;
use rstest::rstest;

use super::*;

const MODEL: &str = "azure_ai/doc-intelligence/prebuilt-read";

fn read_request(base: &str, options: Value) -> LiteLLMOcrRequest {
    ocr_request(MODEL, base, options)
}

#[tokio::test]
async fn pages_features_and_extra_options_map_to_the_analyze_call() {
    let upstream = upstream([json_response(json!({
        "status": "succeeded",
        "analyzeResult": {"pages": []}
    }))])
    .await;
    let request = read_request(
        &upstream.uri(),
        json!({
            "pages": [2, 0, 0, 1],
            "features": ["keyValuePairs", "languages"],
            "future_option": {"nested": null},
            "extra_body": {"provider_option": false}
        }),
    )
    .with_document(
        document(
            json!({"type": "document_url", "document_url": "https://example.com/document.pdf"}),
        )
        .into(),
    );

    perform(request).await.unwrap();

    let sent = only_request(&upstream).await;
    assert!(
        sent.url.path().ends_with("/prebuilt-read:analyze"),
        "{}",
        sent.url
    );
    assert_eq!(sent.query("pages").as_deref(), Some("1,2,3"));
    assert_eq!(
        sent.query("features").as_deref(),
        Some("keyValuePairs,languages")
    );
    assert_eq!(
        sent.json(),
        json!({
            "urlSource": "https://example.com/document.pdf",
            "future_option": {"nested": null},
            "provider_option": false
        })
    );
}

#[rstest]
#[case(json!({"pages": [true]}), Error::Pages("expected only integers or only strings".into()))]
#[case(json!({"pages": [1, "2"]}), Error::Pages("expected only integers or only strings".into()))]
#[case(json!({"pages": [-1]}), Error::Pages("negative page index".into()))]
#[case(json!({"pages": "1&&features=bad"}), Error::Pages("invalid native page range".into()))]
#[case(json!({"features": "languages&pages=1"}), Error::Features)]
#[case(json!({"req_format": "azure"}), Error::RequestFormat)]
#[tokio::test]
async fn invalid_pages_features_and_format_are_rejected_before_sending(
    #[case] options: Value,
    #[case] expected: Error,
) {
    let upstream = upstream([json_response(json!({}))]).await;

    let result = match decode_request(wire(
        MODEL,
        &upstream.uri(),
        json!({"type": "document_url", "document_url": "https://example.com/a.pdf"}),
        options.clone(),
    )) {
        Ok(request) => perform(request).await,
        Err(error) => Err(error),
    };

    assert!(
        received(&upstream).await.is_empty(),
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
#[case::no_options(json!({}))]
#[case::litellm_format(json!({"req_format": "litellm"}))]
#[tokio::test]
async fn an_inline_document_is_sent_as_base64_and_only_page_text_is_kept(#[case] options: Value) {
    let upstream = upstream([json_response(json!({
        "status": "succeeded",
        "analyzeResult": {"pages": [{"pageNumber": 1, "lines": [{"content": "hello"}]}]}
    }))])
    .await;

    let response = perform(read_request(&upstream.uri(), options))
        .await
        .unwrap();

    assert_eq!(response.pages.len(), 1);
    assert_eq!(response.pages[0].index, 0);
    assert_eq!(response.pages[0].markdown, "hello");
    assert_eq!(response.provider_native_response, None);
    let serialized = response.into_json();
    for field in ["content", "tables", "keyValuePairs"] {
        assert_eq!(serialized.get(field), Some(&Value::Null), "{field}");
    }
    let sent = only_request(&upstream).await;
    for field in ["pages", "features", "req_format"] {
        assert_eq!(sent.query(field), None, "{field}");
    }
    assert_eq!(sent.json(), json!({"base64Source": "YWJj"}));
}

#[tokio::test]
async fn native_format_normalizes_pages_and_keeps_the_provider_response() {
    let operation = json!({
        "status": "succeeded",
        "operationExtension": 42,
        "analyzeResult": {
            "content": "A\n\nB",
            "tables": [{"cells": []}],
            "keyValuePairs": [{"key": {"content": "A"}}],
            "pages": [{
                "pageNumber": "2",
                "width": "8.5",
                "height": 11,
                "unit": "inch",
                "lines": [{"content": "A"}, {"content": null}, {"content": "B"}]
            }]
        }
    });
    let upstream = upstream([json_response(operation.clone())]).await;

    let result = perform(read_request(
        &upstream.uri(),
        json!({"req_format": "native"}),
    ))
    .await
    .unwrap();

    assert_eq!(result.pages[0].index, 1);
    assert_eq!(result.pages[0].markdown, "A\n\nB");
    assert_eq!(
        serde_json::to_value(&result.pages[0].dimensions).unwrap(),
        json!({"width": 816, "height": 1056, "dpi": 96})
    );
    assert_eq!(result.usage_info.as_ref().unwrap().pages_processed, Some(1));
    let serialized = result.clone().into_json();
    assert_eq!(serialized["content"], "A\n\nB");
    assert_eq!(serialized["tables"], json!([{"cells": []}]));
    assert_eq!(
        serialized["keyValuePairs"],
        json!([{"key": {"content": "A"}}])
    );
    assert!(serialized.get("key_value_pairs").is_none());
    assert_eq!(
        result.provider_native_response.map(Value::Object),
        Some(operation)
    );
}

#[tokio::test]
async fn client_settings_choose_the_api_version_and_the_inch_to_pixel_dpi() {
    let upstream = upstream([json_response(json!({
        "status": "succeeded",
        "analyzeResult": {"pages": [{"pageNumber": 1, "width": 8.5, "height": 11, "unit": "inch"}]}
    }))])
    .await;
    let client = ocr_client().with_settings(OcrSettings {
        document_intelligence_api_version: "2099-01-01".into(),
        document_intelligence_dpi: 72,
        ..OcrSettings::default()
    });

    let result =
        litellm_core::ocr::client::perform(&client, read_request(&upstream.uri(), json!({})))
            .await
            .unwrap();

    assert_eq!(
        only_request(&upstream)
            .await
            .query("api-version")
            .as_deref(),
        Some("2099-01-01")
    );
    assert_eq!(
        serde_json::to_value(&result.pages[0].dimensions).unwrap(),
        json!({"width": 612, "height": 792, "dpi": 72})
    );
}

#[tokio::test]
async fn an_accepted_response_polls_to_success_with_only_credentials() {
    let operation = json!({"status": "succeeded", "analyzeResult": {"pages": []}});
    let upstream = MockServer::start().await;
    respond_in_order(
        &upstream,
        [
            accepted(&upstream, json!({})),
            json_response(json!({"status": "running"})).insert_header("Retry-After", "0"),
            json_response(operation.clone()),
        ],
    )
    .await;
    let request = with_headers(
        read_request(&upstream.uri(), json!({"req_format": "native"})),
        &[("X-Trace", "initial-only")],
    );

    let result = perform(request).await.unwrap();

    assert_eq!(
        result.provider_native_response.map(Value::Object),
        Some(operation)
    );
    let requests = received(&upstream).await;
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].header("x-trace"), Some("initial-only"));
    for poll in &requests[1..] {
        assert_eq!(poll.method.as_str(), "GET");
        assert_eq!(poll.url.path(), "/operation");
        assert_eq!(poll.header("x-trace"), None);
        assert_eq!(poll.header("ocp-apim-subscription-key"), Some("test-key"));
    }
}

#[tokio::test]
async fn polling_forwards_bearer_credentials() {
    let upstream = MockServer::start().await;
    respond_in_order(
        &upstream,
        [
            accepted(&upstream, json!({})),
            json_response(json!({"status": "succeeded"})),
        ],
    )
    .await;
    let request = with_headers(
        without_api_key(read_request(&upstream.uri(), json!({}))),
        &[("Authorization", "Bearer token")],
    );

    perform(request).await.unwrap();

    assert_eq!(
        received(&upstream).await[1].header("authorization"),
        Some("Bearer token")
    );
}

#[tokio::test]
async fn response_received_fires_for_the_submission_and_the_completed_poll() {
    let upstream = MockServer::start().await;
    respond_in_order(
        &upstream,
        [
            accepted(&upstream, json!({"submitted": true})),
            json_response(json!({"status": "succeeded"})),
        ],
    )
    .await;
    let observed = Arc::new(Mutex::new(Vec::new()));
    let recorder = observed.clone();
    let host =
        LocalOcrHost::new(read_request(&upstream.uri(), json!({}))).with_observer(move |event| {
            if let CallEvent::Machine(MachineEvent::ResponseReceived { raw }) = event {
                recorder.lock().unwrap().push(raw.body.clone());
            }
        });

    perform_with(host).await.unwrap();

    assert_eq!(received(&upstream).await.len(), 2);
    assert_eq!(
        *observed.lock().unwrap(),
        [r#"{"submitted":true}"#, r#"{"status":"succeeded"}"#]
    );
}

#[tokio::test]
async fn polling_does_not_follow_redirects() {
    let upstream = MockServer::start().await;
    respond_in_order(
        &upstream,
        [
            accepted(&upstream, json!({})),
            ResponseTemplate::new(302)
                .insert_header("Location", format!("{}/redirected", upstream.uri())),
            json_response(json!({"status": "succeeded"})),
        ],
    )
    .await;

    let error = perform(read_request(&upstream.uri(), json!({})))
        .await
        .unwrap_err();

    assert!(error.to_string().contains("status 302"), "{error}");
    assert_eq!(received(&upstream).await.len(), 2);
}

#[tokio::test]
async fn a_failed_operation_is_an_error() {
    let upstream = MockServer::start().await;
    respond_in_order(
        &upstream,
        [
            accepted(&upstream, json!({})),
            json_response(json!({"status": "failed"})),
        ],
    )
    .await;

    let error = perform(read_request(&upstream.uri(), json!({})))
        .await
        .unwrap_err();

    assert!(error.to_string().contains("status failed"), "{error}");
}

#[tokio::test]
async fn the_polling_deadline_bounds_the_retry_delay() {
    let upstream = MockServer::start().await;
    respond_in_order(
        &upstream,
        [
            accepted(&upstream, json!({})),
            json_response(json!({"status": "notStarted"})).insert_header("Retry-After", "9999"),
        ],
    )
    .await;
    let client = ocr_client().with_settings(OcrSettings {
        poll_timeout: Duration::from_millis(100),
        ..OcrSettings::default()
    });

    let error = tokio::time::timeout(
        Duration::from_secs(1),
        litellm_core::ocr::client::perform(&client, read_request(&upstream.uri(), json!({}))),
    )
    .await
    .expect("the deadline cuts the retry delay short")
    .unwrap_err();

    assert!(error.to_string().contains("timed out"), "{error}");
}

#[rstest]
#[case::null_pages(json!({"pages": null}), "pages")]
#[case::null_page(json!({"pages": [null]}), "pages[0]")]
#[case::null_lines(json!({"pages": [{"lines": null}]}), "lines")]
#[case::bad_width(json!({"pages": [{"width": "bad"}]}), "width")]
#[tokio::test]
async fn malformed_provider_pages_report_the_response_path(
    #[case] analysis: Value,
    #[case] path: &str,
) {
    let upstream = upstream([json_response(json!({
        "status": "succeeded",
        "analyzeResult": analysis
    }))])
    .await;

    let error = perform(read_request(&upstream.uri(), json!({})))
        .await
        .unwrap_err();

    assert!(error.to_string().contains(path), "{error}");
}

#[rstest]
#[case::missing(None)]
#[case::relative(Some("/relative"))]
#[case::cross_origin(Some("http://example.com/operation"))]
#[case::with_userinfo(Some("http://user:password@127.0.0.1/operation"))]
#[tokio::test]
async fn an_unusable_operation_location_is_rejected(#[case] location: Option<&str>) {
    let response = location
        .into_iter()
        .fold(ResponseTemplate::new(202), |response, location| {
            response.insert_header("Operation-Location", location)
        });
    let upstream = upstream([response]).await;

    let error = perform(read_request(&upstream.uri(), json!({})))
        .await
        .unwrap_err();

    assert!(error.to_string().contains("operation-location"), "{error}");
    assert_eq!(received(&upstream).await.len(), 1);
}

#[tokio::test]
async fn the_model_id_is_percent_encoded() {
    let upstream = upstream([json_response(json!({"status": "succeeded"}))]).await;

    perform(ocr_request(
        "azure_ai/doc-intelligence/a ?#é",
        &upstream.uri(),
        json!({}),
    ))
    .await
    .unwrap();

    let sent = only_request(&upstream).await;
    assert!(
        sent.url.path().ends_with("/a%20%3F%23%C3%A9:analyze"),
        "{}",
        sent.url
    );
}

#[rstest]
#[case::dot("azure_ai/doc-intelligence/.")]
#[case::dot_dot("azure_ai/doc-intelligence/..")]
#[tokio::test]
async fn dot_segment_model_ids_are_rejected(#[case] model: &str) {
    let error = perform(ocr_request(model, UNREACHABLE_BASE, json!({})))
        .await
        .unwrap_err();

    assert!(error.to_string().contains("dot segment"), "{error}");
}
