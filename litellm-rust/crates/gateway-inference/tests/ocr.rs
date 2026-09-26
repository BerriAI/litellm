mod support;

use axum::{body::Body, http::Request};
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

const DOCUMENT: &str = "data:application/pdf;base64,YWJj";

#[rstest]
#[case("/ocr", false)]
#[case("/v1/ocr", true)]
#[tokio::test]
async fn json_and_multipart_reach_ocr_with_the_deployment(
    #[case] route: &str,
    #[case] multipart: bool,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST")).and(path("/v1/ocr"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({"model": "test-ocr", "document": {"type": "document_url", "document_url": DOCUMENT}, "pages": [0]})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"pages": [{"index": 0, "markdown": "recognized text"}]})))
        .expect(1).mount(&upstream).await;
    let app = support::app("mistral/test-ocr", &upstream.uri());
    let response = if multipart {
        let body = "--boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\npublic/model\r\n--boundary\r\nContent-Disposition: form-data; name=\"pages\"\r\n\r\n[0]\r\n--boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.pdf\"\r\nContent-Type: application/pdf\r\n\r\nabc\r\n--boundary--\r\n";
        app.oneshot(
            Request::post(route)
                .header("content-type", "multipart/form-data; boundary=boundary")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap()
    } else {
        support::post(app, route, json!({"model": "public/model", "document": {"type": "document_url", "document_url": DOCUMENT}, "pages": [0]})).await
    };
    assert_eq!(response.status(), 200);
    let body = support::json(response).await;
    assert_eq!(body["pages"][0]["markdown"], "recognized text");
    assert_eq!(body["model"], "test-ocr");
}

#[rstest]
#[case(None, true)]
#[case(Some("litellm"), false)]
#[tokio::test]
async fn native_format_header_is_used_unless_the_body_overrides_it(
    #[case] format: Option<&str>,
    #[case] native: bool,
) {
    let upstream = MockServer::start().await;
    let payload = json!({"pages": [{"index": 0, "markdown": "text"}], "provider_only": true});
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(payload.clone()))
        .expect(1)
        .mount(&upstream)
        .await;
    let body = json!({"model": "public/model", "document": {"type": "document_url", "document_url": DOCUMENT}, "req_format": format});
    let response = support::app("mistral/test-ocr", &upstream.uri())
        .oneshot(
            Request::post("/ocr")
                .header("x-req-format", " Native ")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = support::json(response).await;
    if native {
        assert_eq!(body, payload);
    } else {
        assert_eq!(body["object"], "ocr");
        assert_eq!(
            body["pages"][0]["markdown"],
            payload["pages"][0]["markdown"]
        );
    }
}

#[tokio::test]
async fn ocr_keeps_upstream_status_in_an_openai_error_envelope() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({"message": "busy"})))
        .expect(1)
        .mount(&upstream)
        .await;
    let response = support::post(support::app("mistral/test-ocr", &upstream.uri()), "/ocr",
        json!({"model": "public/model", "document": {"type": "document_url", "document_url": DOCUMENT}})).await;
    assert_eq!(response.status(), 429);
    let body = support::json(response).await;
    assert_eq!(body["error"]["code"], 429);
    assert!(body["error"]["message"].as_str().unwrap().contains("busy"));
}

#[rstest]
#[case(json!({"model": "public/model"}))]
#[case(json!({"model": "public/model", "document": "/etc/passwd"}))]
#[case(json!({"model": "missing", "document": {"type": "document_url", "document_url": DOCUMENT}}))]
#[tokio::test]
async fn invalid_ocr_requests_do_not_call_the_provider(#[case] body: Value) {
    let upstream = MockServer::start().await;
    let response = support::post(
        support::app("mistral/test-ocr", &upstream.uri()),
        "/ocr",
        body,
    )
    .await;
    assert_eq!(response.status(), 400);
    assert!(support::json(response).await["error"]["message"].is_string());
    assert!(upstream.received_requests().await.unwrap().is_empty());
}
