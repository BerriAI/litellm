use litellm_host::interceptors::RawResponse;
use litellm_host::{
    interceptors::{ExecutionFacts, ResultSource},
    lifecycle::ExecutionEvent,
};
use std::time::Duration;

use litellm_auth::{InputSource, SecretValue, Sourced};
use litellm_http::transport::Error as TransportError;
use litellm_inference::Connection;
use litellm_inference_chat::{Error, types::ChatCompletionsCall};
use litellm_llms_types::formats::chat_completions::ChatCompletionsResponse;
use rstest::{fixture, rstest};
use serde_json::{Map, Value, json};
use wiremock::ResponseTemplate;

mod support;
use support::*;

const ANTHROPIC_MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20260101","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":4}}"#;

async fn complete(request: ChatCompletionsCall) -> Result<ChatCompletionsResponse, Error> {
    chat_completions_route().execute(request, &(), None).await
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

fn deployment<T>(value: T) -> Option<Sourced<T>> {
    Some(Sourced::new(value, InputSource::Deployment))
}

fn hi() -> Value {
    json!([{"role": "user", "content": "hi"}])
}

#[fixture]
fn request() -> ChatCompletionsCall {
    ChatCompletionsCall {
        model: "anthropic/claude-sonnet-4-5".into(),
        custom_llm_provider: None,
        messages: hi(),
        optional_params: object(json!({"max_tokens": 16})),
        connection: Connection {
            api_key: deployment(SecretValue::new("sk-test")),
            timeout: Some(Duration::from_secs(10)),
            ..Connection::default()
        },
    }
}

fn at(base: &str, request: ChatCompletionsCall) -> ChatCompletionsCall {
    ChatCompletionsCall {
        connection: Connection {
            api_base: deployment(base.into()),
            ..request.connection
        },
        ..request
    }
}

#[rstest]
#[tokio::test]
async fn anthropic_round_trip_translates_the_conversation_and_normalizes_the_response(
    request: ChatCompletionsCall,
) {
    let upstream = upstream([anthropic_response(ANTHROPIC_MESSAGE)]).await;
    let base = upstream.uri();

    let response = complete(ChatCompletionsCall {
        messages: json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"}
        ]),
        ..at(&base, request)
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
async fn the_deployment_key_replaces_a_caller_supplied_x_api_key(request: ChatCompletionsCall) {
    let upstream = upstream([anthropic_response(ANTHROPIC_MESSAGE)]).await;
    let base = upstream.uri();

    let request = at(&base, request);
    complete(ChatCompletionsCall {
        connection: Connection {
            extra_headers: deployment(object(
                json!({"x-api-key": "caller-key", "x-trace": "kept"}),
            )),
            ..request.connection
        },
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
async fn bedrock_round_trip_is_signed_and_normalized(request: ChatCompletionsCall) {
    let upstream = upstream([json_response(json!({
        "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}
    }))])
    .await;
    let base = upstream.uri();

    let response = complete(ChatCompletionsCall {
        model: "bedrock/anthropic.claude-sonnet-4-5".into(),
        optional_params: object(json!({
            "aws_access_key_id": "access-key",
            "aws_secret_access_key": "secret-key",
            "aws_region_name": "eu-west-1"
        })),
        connection: Connection {
            api_base: deployment(base),
            ..Connection::default()
        },
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

#[rstest]
#[case::missing_usage(
    r#"{"model":"m","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn"}"#
)]
#[case::tool_use_block(r#"{"model":"m","content":[{"type":"tool_use","id":"t","name":"f","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}"#)]
#[case::not_json("not json")]
#[tokio::test]
async fn a_response_it_cannot_normalize_is_reported_as_already_sent(
    request: ChatCompletionsCall,
    #[case] body: &str,
) {
    let upstream = upstream([anthropic_response(body)]).await;
    let base = upstream.uri();

    let error = complete(at(&base, request))
        .await
        .expect_err("response cannot be normalized");

    assert!(matches!(error, Error::InvalidResponse(_)), "{error:?}");
}

#[rstest]
#[case::rate_limited(429)]
#[case::server_error(500)]
#[tokio::test]
async fn an_upstream_error_status_keeps_its_code_and_body(
    request: ChatCompletionsCall,
    #[case] status: u16,
) {
    let upstream = upstream([ResponseTemplate::new(status).set_body_string("slow down")]).await;
    let base = upstream.uri();

    let error = complete(at(&base, request))
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

#[rstest]
#[tokio::test]
async fn a_connection_that_is_never_established_returns_a_connect_error(
    request: ChatCompletionsCall,
) {
    let error = complete(at(UNREACHABLE_BASE, request))
        .await
        .expect_err("nothing is listening");

    assert!(
        matches!(error, Error::Transport(TransportError::Connect(_))),
        "{error:?}"
    );
}

#[rstest]
#[tokio::test]
async fn a_timeout_after_sending_returns_a_network_error(request: ChatCompletionsCall) {
    let upstream =
        upstream([anthropic_response(ANTHROPIC_MESSAGE).set_delay(Duration::from_secs(5))]).await;
    let base = upstream.uri();

    let error = complete(ChatCompletionsCall {
        connection: Connection {
            api_base: deployment(base),
            timeout: Some(Duration::from_millis(100)),
            ..request.connection
        },
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
#[case::direct(false)]
#[case::hosted(true)]
#[tokio::test]
async fn direct_and_hosted_calls_share_hooks_and_lifecycle(
    request: ChatCompletionsCall,
    #[case] hosted: bool,
) {
    use litellm_host::{call::HostedCompletion, lifecycle::CallEvent};
    use litellm_inference_chat::route::ChatCompletions;

    let upstream = upstream([anthropic_response(ANTHROPIC_MESSAGE)]).await;
    let base = upstream.uri();
    let host = RecordingCall::<ChatCompletions>::new(at(&base, request));
    let response = if hosted {
        let result = litellm_host_native::in_process::run_hosted(
            chat_completions_route()
                .machine(host.request().unwrap(), Some(host.events.0.sender.clone())),
            host.runtime(),
        )
        .await
        .unwrap();
        let HostedCompletion::Complete(response) = result else {
            panic!("expected a complete response")
        };
        response
    } else {
        let call = host.request.lock().unwrap().take().unwrap();
        chat_completions_route()
            .execute(call, &host, Some(host.events.0.sender.clone()))
            .await
            .unwrap()
    };
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello")
    );
    assert_eq!(
        only_request(&upstream).await.header("x-hook"),
        Some("called")
    );
    let events = host.events.0.lock().unwrap();
    assert!(matches!(
        &events[..],
        [
            CallEvent::Started { .. },
            CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { .. }),
            CallEvent::Execution(ExecutionEvent::ResultReady {
                facts: ExecutionFacts {
                    source: ResultSource::Provider,
                    ..
                }
            }),
            CallEvent::Succeeded { .. }
        ]
    ));
}

#[rstest]
#[tokio::test]
async fn a_post_call_hook_failure_never_looks_safe_to_retry(request: ChatCompletionsCall) {
    use litellm_host::interceptors::{Interceptors, RequestContext, WireRequest};
    struct FailingHook;
    impl Interceptors<Error> for FailingHook {
        async fn before_provider_request(
            &self,
            wire: WireRequest,
            _: RequestContext,
        ) -> Result<WireRequest, Error> {
            Ok(wire)
        }
        async fn after_provider_response(&self, _: RawResponse) -> Result<(), Error> {
            Err(Error::InvalidRequest("callback rejected".into()))
        }
    }
    let upstream = upstream([anthropic_response(ANTHROPIC_MESSAGE)]).await;
    let base = upstream.uri();
    let error = chat_completions_route()
        .execute(at(&base, request), &FailingHook, None)
        .await
        .unwrap_err();
    let Error::PostCallHook(source) = error else {
        panic!("expected retained callback error")
    };
    assert_eq!(*source, Error::InvalidRequest("callback rejected".into()));
    assert_eq!(received(&upstream).await.len(), 1);
}

#[rstest]
#[tokio::test]
async fn completed_chat_records_route_and_resolved_provider(
    request: ChatCompletionsCall,
    traces: TraceCapture,
) {
    let upstream = upstream([anthropic_response(ANTHROPIC_MESSAGE)]).await;
    let base = upstream.uri();
    let model = request.model.clone();
    traces
        .logger()
        .instrument(complete(at(&base, request)))
        .await
        .unwrap();
    let summaries = traces.summaries("litellm.route");
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0]["route"], "chat_completions");
    assert_eq!(summaries[0]["model"], model);
    assert_eq!(summaries[0]["provider"], "anthropic");
    assert_eq!(
        summaries[0]["resolved_model"],
        only_request(&upstream).await.json()["model"]
    );
    assert_eq!(summaries[0]["outcome"], "success");
    assert_eq!(summaries[0]["stream"], false);
}
