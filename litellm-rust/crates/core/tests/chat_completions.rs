use std::time::Duration;

use litellm_core::chat_completions::{
    Error, chat_completions, chat_completions_decline_reason, types::ChatCompletionsRequest,
};
use litellm_http::transport::Error as TransportError;
use litellm_types::utils::ChatCompletionsResponse;
use rstest::{fixture, rstest};
use serde_json::{Map, Value, json};
use wiremock::ResponseTemplate;

mod support;
use support::*;

const ANTHROPIC_MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20260101","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":4}}"#;

async fn complete(request: ChatCompletionsRequest<'_>) -> Result<ChatCompletionsResponse, Error> {
    chat_completions(&http_pool(), &http_config(), request).await
}

fn object(value: Value) -> Map<String, Value> {
    let Value::Object(map) = value else {
        panic!("expected a json object, got {value}");
    };
    map
}

fn anthropic_response(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body, "application/json")
}

fn hi() -> Value {
    json!([{"role": "user", "content": "hi"}])
}

#[fixture]
fn request() -> ChatCompletionsRequest<'static> {
    ChatCompletionsRequest {
        model: "anthropic/claude-sonnet-4-5",
        messages: hi(),
        optional_params: object(json!({"max_tokens": 16})),
        api_key: Some("sk-test"),
        api_base: None,
        custom_llm_provider: None,
        extra_headers: None,
        timeout: Some(Duration::from_secs(10)),
    }
}

#[rstest]
#[tokio::test]
async fn anthropic_round_trip_translates_the_conversation_and_normalizes_the_response(
    request: ChatCompletionsRequest<'static>,
) {
    let upstream = upstream([anthropic_response(ANTHROPIC_MESSAGE)]).await;
    let base = upstream.uri();

    let response = complete(ChatCompletionsRequest {
        messages: json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"}
        ]),
        api_base: Some(&base),
        ..request
    })
    .await
    .expect("call succeeds");

    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/v1/messages");
    assert_eq!(sent.header_values("x-api-key"), ["sk-test"]);
    let body = sent.json();
    assert_eq!(body["model"], "claude-sonnet-4-5");
    assert_eq!(
        body["messages"],
        json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    );
    assert_eq!(
        body["system"],
        json!([{"type": "text", "text": "be terse"}])
    );
    assert_eq!(body["max_tokens"], 16);
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello")
    );
    assert_eq!(response.usage.total_tokens, 15);
}

#[rstest]
#[tokio::test]
async fn the_deployment_key_replaces_a_caller_supplied_x_api_key(
    request: ChatCompletionsRequest<'static>,
) {
    let upstream = upstream([anthropic_response(ANTHROPIC_MESSAGE)]).await;
    let base = upstream.uri();

    complete(ChatCompletionsRequest {
        api_base: Some(&base),
        extra_headers: Some(object(
            json!({"x-api-key": "caller-key", "x-trace": "kept"}),
        )),
        ..request
    })
    .await
    .expect("call succeeds");

    let sent = only_request(&upstream).await;
    assert_eq!(sent.header_values("x-api-key"), ["sk-test"]);
    assert_eq!(sent.header("x-trace"), Some("kept"));
}

#[rstest]
#[tokio::test]
async fn bedrock_round_trip_is_signed_and_normalized(request: ChatCompletionsRequest<'static>) {
    let upstream = upstream([json_response(json!({
        "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}
    }))])
    .await;
    let base = upstream.uri();

    let response = complete(ChatCompletionsRequest {
        model: "bedrock/anthropic.claude-sonnet-4-5",
        optional_params: object(json!({
            "aws_access_key_id": "access-key",
            "aws_secret_access_key": "secret-key",
            "aws_region_name": "eu-west-1"
        })),
        api_key: None,
        api_base: Some(&base),
        ..request
    })
    .await
    .expect("call succeeds");

    let sent = only_request(&upstream).await;
    assert_eq!(
        sent.url.path(),
        "/model/anthropic.claude-sonnet-4-5/converse"
    );
    let authorization = sent.header("authorization").expect("request is signed");
    assert!(
        authorization.contains("/eu-west-1/bedrock/aws4_request"),
        "{authorization}"
    );
    assert_eq!(
        sent.json()["messages"],
        json!([{"role": "user", "content": [{"text": "hi"}]}])
    );
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello")
    );
    assert_eq!(response.usage.total_tokens, 15);
}

/// The provider already answered and billed these, so the host must not retry them on
/// its own path: they surface as `InvalidResponse`, never as a pre-send decline.
#[rstest]
#[case::missing_usage(
    r#"{"model":"m","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn"}"#
)]
#[case::tool_use_block(r#"{"model":"m","content":[{"type":"tool_use","id":"t","name":"f","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}"#)]
#[case::not_json("not json")]
#[tokio::test]
async fn a_response_it_cannot_normalize_is_reported_as_already_sent(
    request: ChatCompletionsRequest<'static>,
    #[case] body: &str,
) {
    let upstream = upstream([anthropic_response(body)]).await;
    let base = upstream.uri();

    let error = complete(ChatCompletionsRequest {
        api_base: Some(&base),
        ..request
    })
    .await
    .expect_err("response cannot be normalized");

    assert!(matches!(error, Error::InvalidResponse(_)), "{error:?}");
}

#[rstest]
#[case::rate_limited(429)]
#[case::server_error(500)]
#[tokio::test]
async fn an_upstream_error_status_keeps_its_code_and_body(
    request: ChatCompletionsRequest<'static>,
    #[case] status: u16,
) {
    let upstream = upstream([ResponseTemplate::new(status).set_body_string("slow down")]).await;
    let base = upstream.uri();

    let error = complete(ChatCompletionsRequest {
        api_base: Some(&base),
        ..request
    })
    .await
    .expect_err("upstream rejects");

    assert_eq!(
        error,
        Error::Transport(TransportError::Http {
            status,
            body: "slow down".into()
        })
    );
}

/// Nothing was sent, so nothing was billed and the host can still serve the request.
#[rstest]
#[tokio::test]
async fn a_connection_that_is_never_established_declines_instead_of_failing(
    request: ChatCompletionsRequest<'static>,
) {
    let error = complete(ChatCompletionsRequest {
        api_base: Some(UNREACHABLE_BASE),
        ..request
    })
    .await
    .expect_err("nothing is listening");

    assert!(
        matches!(error, Error::Transport(TransportError::Connect(_))),
        "{error:?}"
    );
}

#[rstest]
#[tokio::test]
async fn a_timeout_after_sending_is_not_a_pre_send_decline(
    request: ChatCompletionsRequest<'static>,
) {
    let upstream =
        upstream([anthropic_response(ANTHROPIC_MESSAGE).set_delay(Duration::from_secs(5))]).await;
    let base = upstream.uri();

    let error = complete(ChatCompletionsRequest {
        api_base: Some(&base),
        timeout: Some(Duration::from_millis(100)),
        ..request
    })
    .await
    .expect_err("the call times out");

    assert!(
        matches!(error, Error::Transport(TransportError::Network(_))),
        "{error:?}"
    );
}

#[rstest]
#[case::accepted("anthropic/claude-sonnet-4-5", None, hi(), json!({"max_tokens": 16}), None)]
#[case::accepted_bedrock("bedrock/anthropic.claude-sonnet-4-5", None, hi(), json!({}), None)]
#[case::unknown_provider(
    "gpt-4o",
    Some("openai"),
    hi(),
    json!({}),
    Some("provider is not on the rust chat completions path")
)]
#[case::unreadable_messages(
    "anthropic/claude-sonnet-4-5",
    None,
    json!("hi"),
    json!({}),
    Some("unreadable message list")
)]
#[case::empty_messages("anthropic/claude-sonnet-4-5", None, json!([]), json!({}), Some("empty message list"))]
#[case::streaming(
    "anthropic/claude-sonnet-4-5",
    None,
    hi(),
    json!({"stream": true}),
    Some("streaming")
)]
#[case::unrecognized_param(
    "anthropic/claude-sonnet-4-5",
    None,
    hi(),
    json!({"not_a_param": 1}),
    Some("unrecognized request parameter")
)]
#[case::opens_on_assistant_turn(
    "anthropic/claude-sonnet-4-5",
    None,
    json!([{"role": "assistant", "content": "hi"}]),
    json!({}),
    Some("conversation does not open on a user turn")
)]
fn decline_reason_names_why_the_core_would_not_serve_the_request(
    #[case] model: &str,
    #[case] provider: Option<&str>,
    #[case] messages: Value,
    #[case] params: Value,
    #[case] reason: Option<&str>,
) {
    assert_eq!(
        chat_completions_decline_reason(model, provider, messages, &object(params)),
        reason
    );
}

/// A request the decline check accepts must not be declined by the call itself.
#[rstest]
#[tokio::test]
async fn a_declined_request_fails_the_call_before_sending(
    request: ChatCompletionsRequest<'static>,
) {
    let upstream = upstream([anthropic_response(ANTHROPIC_MESSAGE)]).await;
    let base = upstream.uri();

    let error = complete(ChatCompletionsRequest {
        optional_params: object(json!({"stream": true})),
        api_base: Some(&base),
        ..request
    })
    .await
    .expect_err("streaming is declined");

    assert_eq!(error, Error::Unsupported("streaming"));
    assert!(received(&upstream).await.is_empty());
}
