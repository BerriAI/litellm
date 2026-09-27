mod support;

use axum::{
    body::{Body, Bytes},
    http::Request,
};
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, method},
};

#[rstest]
#[case::chat("/chat/completions", Some("public/model"))]
#[case::versioned_chat("/v1/chat/completions", Some("public/model"))]
#[case::engine("/engines/public/model/chat/completions", None)]
#[case::deployment("/openai/deployments/public/model/chat/completions", None)]
#[case::body_model_wins("/openai/deployments/unused/chat/completions", Some("public/model"))]
#[tokio::test]
async fn chat_aliases_call_core_and_use_the_body_model_before_the_path(
    #[case] route: &str,
    #[case] model: Option<&str>,
    #[values(None, Some("application/json"), Some("text/plain"))] content_type: Option<&str>,
) {
    let upstream = MockServer::start().await;
    let messages = json!([{"role": "user", "content": "hi"}]);
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"model": "test-model", "max_tokens": 16}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test", "model": "test-model", "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .expect(1)
        .mount(&upstream)
        .await;
    let request = Request::post(route);
    let request = match content_type {
        Some(content_type) => request.header("content-type", content_type),
        None => request,
    };
    let response = support::app("anthropic/test-model", &upstream.uri())
        .oneshot(
            request
                .body(Body::from(
                    json!({"model": model, "messages": messages, "max_tokens": 16}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(
        support::json(response).await["choices"][0]["message"]["content"],
        "hello"
    );
}

#[rstest]
#[case("/responses")]
#[case("/v1/responses")]
#[case("/embeddings")]
#[case("/v1/embeddings")]
#[case("/completions")]
#[case("/v1/completions")]
#[case("/engines/public/model/embeddings")]
#[case("/openai/deployments/public/model/completions")]
#[tokio::test]
async fn unimplemented_routes_return_an_explicit_error(#[case] path: &str) {
    let response = support::post(
        support::app("anthropic/test-model", "http://127.0.0.1:1"),
        path,
        json!({}),
    )
    .await;
    assert_eq!(response.status(), 501);
    assert!(
        support::json(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not implemented")
    );
}

#[rstest]
#[case("/audio/transcriptions")]
#[case("/v1/audio/transcriptions")]
#[tokio::test]
async fn transcription_aliases_reach_core_validation(#[case] path: &str) {
    let response = support::post(
        support::app("bedrock/test-model", "http://127.0.0.1:1"),
        path,
        json!({"model": "public/model", "audio": {"data": "YWJj", "format": "invalid"}}),
    )
    .await;
    assert_eq!(response.status(), 400);
    let body: Value = support::json(response).await;
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("audio.format")
    );
}

#[rstest]
#[case::chat("/v1/chat/completions", false)]
#[case::deployment("/openai/deployments/public/model/chat/completions", false)]
#[case::messages("/v1/messages", true)]
#[case::ocr("/v1/ocr", false)]
#[case::transcription("/v1/audio/transcriptions", false)]
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
#[case::unsupported("/engines/public/model/embeddings", 501)]
#[case::unknown("/engines/public/model/unknown", 404)]
#[tokio::test]
async fn deployment_path_errors_take_precedence_over_invalid_json(
    #[case] path: &str,
    #[case] status: u16,
) {
    let response = support::app("anthropic/test-model", "http://127.0.0.1:1")
        .oneshot(Request::post(path).body(Body::from("{")).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), status);
}
