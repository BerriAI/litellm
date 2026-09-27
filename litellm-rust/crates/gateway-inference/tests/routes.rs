mod support;

use std::sync::Arc;

use axum::{
    body::{Body, Bytes},
    http::Request,
};
use litellm_core::{
    chat_completions::{chat_completions, types::ChatCompletionsRequest},
    resources::CoreResources,
};
use litellm_http::{HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver};
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
#[case::engine("/engines/public%2Fmodel/chat/completions", None)]
#[case::deployment("/openai/deployments/public%2Fmodel/chat/completions", None)]
#[case::body_model_wins("/openai/deployments/unused/chat/completions", Some("public/model"))]
#[tokio::test]
async fn chat_aliases_call_core_and_use_the_body_model_before_the_path(
    #[case] route: &str,
    #[case] model: Option<&str>,
    #[values(None, Some("application/json"), Some("text/plain"))] content_type: Option<&str>,
    #[values(None, Some(false))] stream: Option<bool>,
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
                    json!({"model": model, "messages": messages, "max_tokens": 16, "stream": stream}).to_string(),
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
#[case::streaming(json!({"messages": [{"role": "user", "content": "hi"}], "stream": true}), 501)]
#[case::missing_messages(json!({}), 400)]
#[case::malformed_messages(json!({"messages": "hi"}), 400)]
#[case::invalid_streaming_request(json!({"messages": [], "stream": true}), 400)]
#[tokio::test]
async fn chat_errors_come_from_core(
    #[case] fields: Value,
    #[case] status: u16,
    #[values("/v1/chat/completions", "/engines/public%2Fmodel/chat/completions")] path: &str,
) {
    let upstream = MockServer::start().await;
    let base = upstream.uri();
    let resources = CoreResources::new(Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver))));
    let http = Resolution::from(&HttpSettings::default()).config;
    let fields = fields.as_object().unwrap();
    let error = chat_completions(
        &resources,
        &http,
        ChatCompletionsRequest {
            model: "anthropic/test-model",
            messages: fields.get("messages").cloned().unwrap_or_default(),
            optional_params: fields
                .iter()
                .filter(|(name, _)| name.as_str() != "messages")
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
            api_key: Some("test-key"),
            api_base: Some(&base),
            custom_llm_provider: None,
            extra_headers: None,
            timeout: None,
        },
    )
    .await
    .unwrap_err();
    let body = fields
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .chain([("model".into(), json!("public/model"))])
        .collect();
    let response = support::post(
        support::app("anthropic/test-model", &base),
        path,
        Value::Object(body),
    )
    .await;
    assert_eq!(response.status(), status);
    let body = support::json(response).await;
    assert_eq!(body["error"]["message"], error.to_string());
    assert_eq!(body["error"]["code"], status);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::audio("/v1/audio/transcriptions", "bedrock/test-model", "audio")]
#[case::document("/v1/ocr", "mistral/test-ocr", "document")]
#[tokio::test]
async fn missing_inference_fields_use_the_same_validation_as_null(
    #[case] path: &str,
    #[case] model: &str,
    #[case] field: &str,
) {
    let upstream = MockServer::start().await;
    let app = support::app(model, &upstream.uri());
    let missing = support::post(app.clone(), path, json!({"model": "public/model"})).await;
    let null = support::post(app, path, json!({"model": "public/model", field: null})).await;
    assert_eq!(missing.status(), 400);
    assert_eq!(missing.status(), null.status());
    assert_eq!(support::json(missing).await, support::json(null).await);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case("/embeddings")]
#[case("/v1/embeddings")]
#[case("/completions")]
#[case("/v1/completions")]
#[case("/engines/public%2Fmodel/embeddings")]
#[case("/openai/deployments/public%2Fmodel/completions")]
#[tokio::test]
async fn unimplemented_routes_return_an_explicit_error(#[case] path: &str) {
    let response = support::post(
        support::app("anthropic/test-model", "http://127.0.0.1:1"),
        path,
        json!({}),
    )
    .await;
    assert_eq!(response.status(), 501);
    let body = support::json(response).await;
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("not implemented"));
    assert!(message.contains(path));
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
#[case::no_extension("test")]
#[case::unsupported_extension("test.invalid")]
#[tokio::test]
async fn upload_audio_format_validation_matches_core(#[case] filename: &str) {
    let upstream = MockServer::start().await;
    let app = support::app("bedrock/test-model", &upstream.uri());
    let path = "/v1/audio/transcriptions";
    let expected = support::post(
        app.clone(),
        path,
        json!({"model": "public/model", "audio": {"data": "YWJj", "format": null}}),
    )
    .await;
    let payload = format!(
        "--test\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\npublic/model\r\n\
         --test\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n\r\nabc\r\n--test--\r\n"
    );
    let response = app
        .oneshot(
            Request::post(path)
                .header("content-type", "multipart/form-data; boundary=test")
                .body(Body::from(payload))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    assert_eq!(support::json(response).await, support::json(expected).await);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::chat("/v1/chat/completions", false)]
#[case::responses("/v1/responses", false)]
#[case::deployment("/openai/deployments/public%2Fmodel/chat/completions", false)]
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
#[case::unsupported("/engines/public%2Fmodel/embeddings", 501)]
#[case::unknown("/engines/public%2Fmodel/unknown", 404)]
#[case::unescaped_model("/engines/public/model/chat/completions", 404)]
#[case::extra_segment("/engines/public%2Fmodel/extra/chat/completions", 404)]
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
