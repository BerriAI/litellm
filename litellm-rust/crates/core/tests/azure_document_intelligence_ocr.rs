use super::transformation::AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG as CONFIG;
use crate::ocr::perform_ocr;
use crate::ocr::prepare::OcrProviderKind;
use crate::ocr::tests::{MockResponse, body, mock_server, params, transform, wire_request};
use crate::ocr::transformation::OcrProviderConfig;
use crate::ocr::types::OcrConnection;
use crate::ocr::wire::decode_params;
use rstest::rstest;
use serde_json::{Value, json};

fn connection() -> OcrConnection {
    OcrConnection {
        api_base: Some("https://example.com".into()),
        api_key: Some("key".into()),
        ..Default::default()
    }
}
fn operation() -> Value {
    json!({"status":"succeeded","operationExtension":42,"analyzeResult":{
        "content":"A\n\nB","tables":[{"cells":[]}],"keyValuePairs":[{"key":{"content":"A"}}],
        "pages":[{"pageNumber":1,"width":8.5,"height":11,"unit":"inch","lines":[{"content":"A"},{"content":null},{"content":"B"}],"words":[{"content":"A","confidence":0.99}]}]}})
}

#[rstest]
#[case(json!([2,0,0,1]), "1,2,3")]
#[case(json!("1-3, 5"), "1-3,5")]
#[case(json!(["1","3-5"]), "1,3-5")]
#[case(json!("0,3-1"), "0,3-1")]
fn document_intelligence_url_normalizes_zero_based_pages(
    #[case] pages: Value,
    #[case] expected: &str,
) {
    let mapped = params(&CONFIG, json!({"pages":pages}));
    let url = CONFIG
        .complete_url(&connection(), "prebuilt-read", &mapped)
        .unwrap();
    assert!(url.ends_with(&format!("&pages={expected}")));
}
#[rstest]
#[case(json!([true]))]
#[case(json!([1,"2"]))]
#[case(json!([18446744073709551615u64]))]
fn invalid_page_types_return_typed_errors(#[case] pages: Value) {
    assert!(
        decode_params(
            OcrProviderKind::AzureDocumentIntelligence,
            json!({"pages":pages}).as_object().unwrap().clone()
        )
        .is_err()
    );
}
#[rstest]
#[case(json!([-1]))]
#[case(json!([i64::MAX]))]
#[case(json!("1&&features=bad"))]
#[case(json!(""))]
#[case(json!(["1-2-3"]))]
fn invalid_page_values_never_panic(#[case] pages: Value) {
    let input = serde_json::from_value(json!({"pages":pages})).unwrap();
    assert!(CONFIG.map_ocr_params(input).is_err());
}
#[test]
fn document_intelligence_page_mapping_omits_empty_list() {
    let mapped = params(&CONFIG, json!({"pages":[],"features":[]}));
    assert!(
        !CONFIG
            .complete_url(&connection(), "prebuilt-read", &mapped)
            .unwrap()
            .contains("&pages")
    );
}
#[rstest]
#[case(json!(["keyValuePairs","languages"]), "keyValuePairs,languages")]
#[case(json!("keyValuePairs, languages"), "keyValuePairs,languages")]
fn document_intelligence_maps_features(#[case] features: Value, #[case] expected: &str) {
    let mapped = params(&CONFIG, json!({"features":features}));
    assert!(
        CONFIG
            .complete_url(&connection(), "prebuilt-read", &mapped)
            .unwrap()
            .ends_with(&format!("&features={expected}"))
    );
}
#[rstest]
#[case(json!(["languages,ocr"]))]
#[case(json!("languages&pages=1"))]
#[case(json!(""))]
fn document_intelligence_rejects_invalid_features(#[case] features: Value) {
    assert!(
        CONFIG
            .map_ocr_params(serde_json::from_value(json!({"features":features})).unwrap())
            .is_err()
    );
}
#[tokio::test]
async fn document_intelligence_mistral_pages_flow_to_query_only() {
    let result = body(
        &CONFIG,
        "model",
        json!({"type":"document_url","document_url":"https://example.com/doc.pdf"}),
        json!({"pages":[0,1],"features":"languages"}),
    )
    .await
    .unwrap();
    assert_eq!(result, json!({"urlSource":"https://example.com/doc.pdf"}));
}
#[rstest]
#[case("data:application/pdf;base64,YWJj", "YWJj")]
#[case("data:application/pdf,YWJj", "YWJj")]
#[case("data:application/pdf;base64,", "data:application/pdf;base64,")]
#[case("data:application/pdf;other,YWJj", "data:application/pdf;other,YWJj")]
#[case("data:application/pdf;base64", "data:application/pdf;base64")]
#[tokio::test]
async fn document_intelligence_request_uses_base64_source_for_data_uri(
    #[case] source: &str,
    #[case] expected: &str,
) {
    assert_eq!(
        body(
            &CONFIG,
            "model",
            json!({"type":"image_url","image_url":source}),
            json!({})
        )
        .await
        .unwrap(),
        json!({"base64Source":expected})
    );
}
#[test]
fn azure_document_intelligence_model_id_is_encoded() {
    let mapped = params(&CONFIG, json!({}));
    assert!(
        CONFIG
            .complete_url(&connection(), "azure_ai/doc-intelligence/a ?#é", &mapped)
            .unwrap()
            .contains("a%20%3F%23%C3%A9:analyze")
    );
}
#[test]
fn azure_document_intelligence_dot_segment_model_id_is_rejected() {
    let mapped = params(&CONFIG, json!({}));
    for model in [".", "..", "azure_ai/doc-intelligence/.."] {
        assert!(CONFIG.complete_url(&connection(), model, &mapped).is_err());
    }
}
#[test]
fn document_intelligence_response_normalizes_pages() {
    let result = transform(&CONFIG, "model", operation(), json!({})).unwrap();
    assert_eq!(result["pages"][0]["markdown"], "A\n\nB");
    assert_eq!(
        result["pages"][0]["dimensions"],
        json!({"width":816,"height":1056,"dpi":96})
    );
    assert_eq!(result["usage_info"]["pages_processed"], 1);
    assert_eq!(result["tables"], operation()["analyzeResult"]["tables"]);
}
#[test]
fn document_intelligence_response_tolerates_missing_native_fields() {
    for value in [
        json!({"status":"succeeded"}),
        json!({"status":"succeeded","analyzeResult":null}),
    ] {
        let result = transform(&CONFIG, "model", value, json!({})).unwrap();
        assert_eq!(result["pages"], json!([]));
        assert_eq!(result["tables"], Value::Null);
    }
    let result = transform(
        &CONFIG,
        "model",
        json!({"status":"succeeded","analyzeResult":{"pages":[{}]}}),
        json!({}),
    )
    .unwrap();
    assert_eq!(result["pages"][0]["dimensions"]["width"], 816);
}
#[rstest]
#[case(json!({"pages":null}), "pages")]
#[case(json!({"pages":[null]}), "pages[0]")]
#[case(json!({"pages":[{"lines":null}]}), "lines")]
#[case(json!({"pages":[{"lines":[{"content":42}]}]}), "content")]
#[case(json!({"pages":[{"width":"bad"}]}), "width")]
fn malformed_pages_are_rejected_with_paths(#[case] analysis: Value, #[case] path: &str) {
    let error = transform(
        &CONFIG,
        "model",
        json!({"status":"succeeded","analyzeResult":analysis}),
        json!({}),
    )
    .unwrap_err();
    assert!(error.to_string().contains(path), "{error}");
}
#[test]
fn page_coercions_and_overflow_are_explicit() {
    let result = transform(&CONFIG,"model",json!({"status":"succeeded","analyzeResult":{"pages":[{"pageNumber":"2","width":"8.5","height":11.0}]}}),json!({})).unwrap();
    assert_eq!(result["pages"][0]["index"], 1);
    for page in [json!({"pageNumber":i64::MIN}), json!({"width":1e100})] {
        let error = transform(
            &CONFIG,
            "model",
            json!({"status":"succeeded","analyzeResult":{"pages":[page]}}),
            json!({}),
        )
        .expect_err("provider page overflows");
        let error = crate::Error::from(error);
        assert_eq!(error.kind(), crate::error::ErrorKind::InvalidResponse);
        assert!(matches!(
            error,
            crate::Error::OcrResponse(crate::ocr::error::OcrResponseError::NumericRange(_))
        ));
    }
}
#[test]
fn document_intelligence_non_succeeded_status_is_rejected() {
    for status in [
        json!("failed"),
        json!("running"),
        json!("unknown"),
        Value::Null,
    ] {
        assert!(transform(&CONFIG, "model", json!({"status":status}), json!({})).is_err());
    }
}
#[test]
fn document_intelligence_native_format_carries_raw_operation() {
    let response = transform(
        &CONFIG,
        "model",
        operation(),
        json!({"req_format":"native"}),
    )
    .unwrap();
    assert_eq!(response["provider_native_response"], operation());
    assert!(
        transform(
            &CONFIG,
            "model",
            operation(),
            json!({"req_format":"litellm"})
        )
        .unwrap()
        .get("provider_native_response")
        .is_none()
    );
}
#[test]
fn document_intelligence_url_omits_req_format() {
    let mapped = params(&CONFIG, json!({"req_format":"native"}));
    assert!(
        !CONFIG
            .complete_url(&connection(), "model", &mapped)
            .unwrap()
            .contains("req_format")
    );
}
#[test]
fn document_intelligence_rejects_unknown_req_format() {
    assert!(
        decode_params(
            OcrProviderKind::AzureDocumentIntelligence,
            json!({"req_format":"azure"}).as_object().unwrap().clone()
        )
        .is_err()
    );
}
#[rstest]
#[case(false)]
#[case(true)]
#[tokio::test]
async fn polling_forwards_subscription_or_bearer_and_preserves_native(#[case] bearer: bool) {
    let (base, requests, server) = mock_server(vec![
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
        MockResponse::json(operation()),
    ])
    .await;
    let mut request = wire_request(
        "azure_ai/doc-intelligence/prebuilt-read",
        &base,
        json!({"req_format":"native","pages":[0,2]}),
    );
    if bearer {
        request.connection.api_key = None;
        request.connection.extra_headers = vec![("Authorization".into(), "Bearer token".into())];
    }
    let result = perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(
        result.provider_native_response,
        Some(operation().as_object().unwrap().clone())
    );
    let seen = requests.lock().unwrap();
    assert!(seen[0].contains("&pages=1,3"));
    for poll in &seen[1..] {
        assert!(poll.to_ascii_lowercase().contains(if bearer {
            "authorization: bearer token"
        } else {
            "ocp-apim-subscription-key: test-key"
        }));
    }
}
#[tokio::test]
async fn polling_rejects_cross_origin_location() {
    let (base, requests, server) = mock_server(vec![MockResponse {
        status: 202,
        headers: vec![("Operation-Location", "http://example.com/operation".into())],
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
    assert!(error.to_string().contains("cross-origin"));
    assert_eq!(requests.lock().unwrap().len(), 1);
}
#[tokio::test]
async fn polling_deadline_bounds_retry_after() {
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
