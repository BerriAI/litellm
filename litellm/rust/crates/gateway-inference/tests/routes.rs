mod support;

use axum::{
    body::{Body, Bytes},
    http::Request,
};
use rstest::rstest;
use serde_json::json;
use tower::ServiceExt;
use wiremock::MockServer;

#[tokio::test]
async fn a_missing_ocr_document_uses_the_same_validation_as_null() {
    let (path, field) = ("/v1/ocr", "document");
    let upstream = MockServer::start().await;
    let app = support::app("mistral/test-ocr", &upstream.uri());
    let missing = support::post(app.clone(), path, json!({"model": "public/model"})).await;
    let null = support::post(app, path, json!({"model": "public/model", field: null})).await;
    assert_eq!(missing.status(), 400);
    assert_eq!(missing.status(), null.status());
    assert_eq!(support::json(missing).await, support::json(null).await);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::messages("/v1/messages", true)]
#[case::ocr("/v1/ocr", false)]
#[tokio::test]
async fn json_extraction_rejections_use_the_endpoint_error_envelope(
    #[case] path: &str,
    #[case] anthropic: bool,
    #[values("syntax", "array", "oversized", "read_failure")] failure: &str,
) {
    let upstream = MockServer::start().await;
    let payload = match failure {
        "syntax" => Body::from("{"),
        "array" => Body::from("[]"),
        "oversized" => Body::from_stream(futures_util::stream::iter(
            std::iter::repeat_n(Bytes::from(vec![b' '; 1024 * 1024]), 52)
                .map(Ok::<_, std::io::Error>),
        )),
        "read_failure" => Body::from_stream(futures_util::stream::once(async {
            Err::<Bytes, _>(std::io::Error::other("body read failed"))
        })),
        _ => unreachable!(),
    };
    let request = Request::post(path)
        .header("x-request-id", "extractor-request")
        .body(payload)
        .unwrap();
    let response = support::app("anthropic/test-model", &upstream.uri())
        .oneshot(request)
        .await
        .unwrap();
    let status = if failure == "oversized" { 413 } else { 400 };
    assert_eq!(response.status(), status);
    let body = support::json(response).await;
    assert_eq!(
        body["error"]["type"],
        if status == 413 {
            "request_too_large"
        } else {
            "invalid_request_error"
        }
    );
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| !message.is_empty())
    );
    if anthropic {
        assert_eq!(body["type"], "error");
        assert_eq!(body["request_id"], "extractor-request");
    } else {
        assert_eq!(body["error"]["code"], status);
        assert!(body["error"]["param"].is_null());
    }
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::messages("/v1/messages", "anthropic/test-model")]
#[case::ocr("/v1/ocr", "mistral/test-ocr")]
#[tokio::test]
async fn a_body_that_sets_the_deployments_endpoint_is_refused_before_any_call(
    #[case] path: &str,
    #[case] model: &str,
) {
    let upstream = MockServer::start().await;
    let app = support::app(model, &upstream.uri());

    let response = support::post(
        app,
        path,
        json!({"model": "public/model", "api_base": "https://attacker.example", "max_tokens": 8, "messages": []}),
    )
    .await;

    assert_eq!(response.status(), 401);
    let message = support::json(response).await["error"]["message"].clone();
    assert!(
        message.as_str().is_some_and(|message| message.starts_with(
            "Authentication Error, Rejected Request: api_base is not allowed in request body."
        )),
        "{message}"
    );
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn a_deployment_that_opts_in_lets_a_caller_set_its_endpoint() {
    let upstream = MockServer::start().await;
    let app = support::app_with_litellm_params(
        "anthropic/test-model",
        &upstream.uri(),
        json!({"configurable_clientside_auth_params": ["api_base"]}),
    );

    let response = support::post(
        app,
        "/v1/messages",
        json!({"model": "public/model", "api_base": upstream.uri(), "max_tokens": 8, "messages": []}),
    )
    .await;

    assert_ne!(response.status(), 401);
    assert_eq!(upstream.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn a_secret_reference_in_a_parameter_no_list_names_is_refused_before_any_call() {
    let upstream = MockServer::start().await;
    let app = support::app("anthropic/test-model", &upstream.uri());

    let response = support::post(
        app,
        "/v1/messages",
        json!({"model": "public/model", "some_provider_param": "os.environ/LITELLM_MASTER_KEY", "max_tokens": 8, "messages": []}),
    )
    .await;

    assert_eq!(response.status(), 401);
    let message = support::json(response).await["error"]["message"].clone();
    assert!(
        message.as_str().is_some_and(|message| message.starts_with(
            "Authentication Error, Rejected Request: some_provider_param holds a secret reference"
        )),
        "{message}"
    );
    assert!(upstream.received_requests().await.unwrap().is_empty());
}
