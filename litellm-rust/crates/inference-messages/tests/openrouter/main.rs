use std::{sync::Arc, time::Duration};

use futures_util::TryStreamExt;
use litellm_inference_messages::{Error, MessagesCall, MessagesCallResponse, MessagesShaping};
use litellm_inference_testing::{RecordingSecrets, no_secrets};
use rstest::{fixture, rstest};
use serde_json::json;
use wiremock::ResponseTemplate;

#[path = "../support/mod.rs"]
mod support;
use support::*;

#[fixture]
fn call() -> MessagesCall {
    MessagesCall {
        body: serde_json::from_value(json!({"model": "openrouter/anthropic/test-model", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})).unwrap(),
        api_key: None,
        api_base: None,
        custom_llm_provider: None,
        litellm_params: Default::default(),
        extra_headers: None,
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: MessagesShaping::default(),
    }
}

fn message_response() -> ResponseTemplate {
    json_response(
        json!({"id": "msg_1", "type": "message", "role": "assistant", "model": "anthropic/test-model", "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}),
    )
}

#[rstest]
#[case::explicit(Some("explicit"), None, "Bearer explicit")]
#[case::managed(None, None, "Bearer managed")]
#[case::forwarded(None, Some("bEaReR caller"), "bEaReR caller")]
#[case::blank(Some("explicit"), Some("Bearer "), "Bearer explicit")]
#[case::malformed(Some("explicit"), Some("invali\u{00e9}auth"), "Bearer explicit")]
#[tokio::test]
async fn only_a_valid_bearer_can_override_the_configured_key(
    call: MessagesCall,
    #[case] api_key: Option<&str>,
    #[case] authorization: Option<&str>,
    #[case] expected: &str,
) {
    let upstream = upstream([message_response()]).await;
    let extra_headers = [("x-api-key".to_string(), json!("caller-anthropic-key"))]
        .into_iter()
        .chain(authorization.map(|value| ("Authorization".to_string(), json!(value))))
        .collect();
    let secrets = Arc::new(RecordingSecrets::new([("OPENROUTER_API_KEY", "managed")]));
    let output = messages_route(secrets)
        .execute(
            MessagesCall {
                api_key: api_key.map(str::to_string),
                api_base: Some(format!("{}/api/v1", upstream.uri())),
                extra_headers: Some(extra_headers),
                ..call
            },
            &(),
            None,
        )
        .await
        .unwrap();
    assert!(matches!(output, MessagesCallResponse::Complete(_)));
    let request = only_request(&upstream).await;
    assert_eq!(request.url.path(), "/api/v1/messages");
    assert_eq!(request.header_values("authorization"), [expected]);
    assert_eq!(request.header("x-api-key"), None);
    assert_eq!(request.json()["model"], "anthropic/test-model");
}

#[rstest]
#[tokio::test]
async fn an_anthropic_key_alone_is_not_an_openrouter_credential(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let result = messages_route(no_secrets())
        .execute(
            MessagesCall {
                api_base: Some(upstream.uri()),
                extra_headers: Some(
                    json!({"x-api-key": "caller-key"})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
                ..call
            },
            &(),
            None,
        )
        .await;
    assert!(matches!(
        result,
        Err(Error::Auth(litellm_auth::Error::MissingApiKey {
            provider: "OpenRouter",
            ..
        }))
    ));
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[tokio::test]
async fn the_secret_manager_supplies_the_native_base_and_key(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let base = format!("{}/api/v1", upstream.uri());
    let secrets = Arc::new(RecordingSecrets::new([
        ("OPENROUTER_API_KEY", "managed"),
        ("OPENROUTER_API_BASE", base.as_str()),
    ]));
    messages_route(secrets)
        .execute(call, &(), None)
        .await
        .unwrap();
    let request = only_request(&upstream).await;
    assert_eq!(request.url.path(), "/api/v1/messages");
    assert_eq!(request.header("authorization"), Some("Bearer managed"));
}

#[rstest]
#[tokio::test]
async fn streaming_relay_preserves_every_event_and_billed_usage(call: MessagesCall) {
    let sse = "event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"cost\":\"0.001\",\"cost_details\":\"unexpected\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let upstream =
        upstream([ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")]).await;
    let streamed = MessagesCall {
        body: serde_json::from_value(json!({"model": "openrouter/anthropic/test-model", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}], "stream": true})).unwrap(),
        api_key: Some("key".into()),
        api_base: Some(upstream.uri()),
        ..call
    };
    let output = messages_route(no_secrets())
        .execute(streamed, &(), None)
        .await
        .unwrap();
    let MessagesCallResponse::Stream { chunks, .. } = output else {
        panic!("expected SSE");
    };
    assert_eq!(
        chunks.try_collect::<Vec<_>>().await.unwrap().concat(),
        sse.as_bytes()
    );
    assert_eq!(only_request(&upstream).await.json()["stream"], true);
}
