use std::{sync::Arc, time::Duration};

use futures_util::TryStreamExt;
use litellm_inference_messages::{Error, MessagesCall, MessagesCallResponse, MessagesShaping};
use litellm_inference_testing::{RecordingSecrets, no_secrets};
use litellm_llms_types::formats::messages::{MessagesRequest, MessagesResponse};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use wiremock::ResponseTemplate;

#[path = "../support/mod.rs"]
mod support;
use support::*;

const MODEL: &str = "claude-test";

#[fixture]
fn call() -> MessagesCall {
    MessagesCall {
        body: serde_json::from_value(json!({"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})).unwrap(),
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

fn with_fields(call: MessagesCall, fields: Value) -> MessagesCall {
    let body = serde_json::to_value(call.body).unwrap();
    MessagesCall {
        body: serde_json::from_value(Value::Object(
            body.as_object()
                .unwrap()
                .clone()
                .into_iter()
                .chain(fields.as_object().unwrap().clone())
                .collect(),
        ))
        .unwrap(),
        ..call
    }
}

fn with_model(call: MessagesCall, model: &str) -> MessagesCall {
    MessagesCall {
        body: MessagesRequest {
            model: model.into(),
            ..call.body
        },
        ..call
    }
}

fn message_response() -> ResponseTemplate {
    json_response(
        json!({"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "text", "text": "hi"}], "model": MODEL, "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 2}}),
    )
}

async fn run_with(
    secrets: Arc<RecordingSecrets>,
    call: MessagesCall,
) -> Result<MessagesCallResponse, Error> {
    messages_route(secrets).execute(call, &(), None).await
}

async fn run(call: MessagesCall) -> Result<MessagesCallResponse, Error> {
    run_with(Arc::new(RecordingSecrets::empty()), call).await
}

async fn run_message(call: MessagesCall) -> MessagesResponse {
    match run(call).await.unwrap() {
        MessagesCallResponse::Complete(message) => *message,
        MessagesCallResponse::Stream { .. } => panic!("expected a completed message"),
    }
}

#[rstest]
#[case::platform(
    "bedrock/claude_platform/claude-test",
    "/v1/messages",
    "x-api-key",
    "key"
)]
#[case::mantle(
    "bedrock_mantle/anthropic.claude-test",
    "/anthropic/v1/messages",
    "authorization",
    "Bearer key"
)]
#[case::mantle_alias(
    "bedrock/mantle/anthropic.claude-test",
    "/anthropic/v1/messages",
    "authorization",
    "Bearer key"
)]
#[tokio::test]
async fn prefixes_select_native_messages_and_strip_connection_fields(
    call: MessagesCall,
    #[case] model: &str,
    #[case] path: &str,
    #[case] header: &str,
    #[case] credential: &str,
) {
    let upstream = upstream([message_response()]).await;
    let message = run_message(MessagesCall {
        api_key: Some("key".into()),
        api_base: Some(upstream.uri()),
        custom_llm_provider: None,
        ..with_fields(
            call,
            json!({"model": model, "workspace_id": "request-workspace"}),
        )
    })
    .await;
    assert_eq!(message.model, MODEL);
    let request = only_request(&upstream).await;
    assert_eq!(request.url.path(), path);
    assert_eq!(request.header(header), Some(credential));
    assert_eq!(request.json()["model"], model.rsplit('/').next().unwrap());
    if path == "/v1/messages" {
        assert_eq!(
            request.header("anthropic-workspace-id"),
            Some("request-workspace")
        );
        assert_eq!(request.json().get("workspace_id"), None);
    }
}

#[rstest]
#[case::platform(
    "bedrock/claude_platform/claude-test",
    "ANTHROPIC_AWS_API_KEY",
    "ANTHROPIC_AWS_BASE_URL",
    "/v1/messages",
    "x-api-key",
    "managed"
)]
#[case::mantle(
    "bedrock_mantle/claude-test",
    "AWS_BEARER_TOKEN_BEDROCK",
    "BEDROCK_MANTLE_API_BASE",
    "/anthropic/v1/messages",
    "authorization",
    "Bearer managed"
)]
#[tokio::test]
async fn managed_keys_bases_and_workspaces_reach_the_upstream(
    call: MessagesCall,
    #[case] model: &str,
    #[case] key_name: &str,
    #[case] base_name: &str,
    #[case] path: &str,
    #[case] header: &str,
    #[case] credential: &str,
) {
    let upstream = upstream([message_response()]).await;
    let base = upstream.uri();
    let secrets = Arc::new(RecordingSecrets::new([
        (key_name, "managed"),
        (base_name, base.as_str()),
        ("ANTHROPIC_AWS_WORKSPACE_ID", "managed-workspace"),
    ]));
    let output = run_with(
        secrets,
        MessagesCall {
            custom_llm_provider: None,
            ..with_model(call, model)
        },
    )
    .await
    .unwrap();
    assert!(matches!(output, MessagesCallResponse::Complete(_)));
    let request = only_request(&upstream).await;
    assert_eq!(request.url.path(), path);
    assert_eq!(request.header(header), Some(credential));
    if path == "/v1/messages" {
        assert_eq!(
            request.header("anthropic-workspace-id"),
            Some("managed-workspace")
        );
    }
}

#[rstest]
#[case::platform("bedrock/claude_platform/claude-test", "aws-external-anthropic")]
#[case::mantle("bedrock_mantle/claude-test", "bedrock")]
#[tokio::test]
async fn managed_aws_credentials_sign_the_exact_native_request(
    call: MessagesCall,
    #[case] model: &str,
    #[case] service: &str,
) {
    let upstream = upstream([message_response()]).await;
    let secrets = Arc::new(RecordingSecrets::new([
        ("AWS_ACCESS_KEY_ID", "test-access"),
        ("AWS_SECRET_ACCESS_KEY", "test-secret"),
        ("AWS_SESSION_TOKEN", "test-session"),
        ("AWS_REGION_NAME", "us-west-2"),
        ("ANTHROPIC_AWS_WORKSPACE_ID", "managed-workspace"),
    ]));
    run_with(
        secrets,
        MessagesCall {
            api_base: Some(upstream.uri()),
            custom_llm_provider: None,
            ..with_model(call, model)
        },
    )
    .await
    .unwrap();
    let request = only_request(&upstream).await;
    let auth = request.header("authorization").unwrap();
    assert!(auth.starts_with("AWS4-HMAC-SHA256 Credential=test-access/"));
    assert!(auth.contains(&format!("/us-west-2/{service}/aws4_request")));
    assert_eq!(request.header("x-amz-security-token"), Some("test-session"));
    assert_eq!(request.header("x-api-key"), None);
    assert_eq!(request.json()["model"], "claude-test");
}

#[rstest]
#[tokio::test]
async fn private_mantle_endpoint_is_used_and_signed(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let params = litellm_router_types::LitellmParams {
        aws: litellm_auth::AwsParams {
            aws_bedrock_runtime_endpoint: Some(upstream.uri()),
            aws_access_key_id: Some("private-access".into()),
            aws_secret_access_key: Some("private-secret".into()),
            aws_region_name: Some("us-west-2".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    run(MessagesCall {
        custom_llm_provider: None,
        litellm_params: params,
        ..with_model(call, "bedrock/mantle/claude-test")
    })
    .await
    .unwrap();
    let request = only_request(&upstream).await;
    assert_eq!(request.url.path(), "/anthropic/v1/messages");
    assert!(
        request
            .header("authorization")
            .unwrap()
            .contains("/us-west-2/bedrock/aws4_request")
    );
}

#[rstest]
#[case::platform("bedrock/claude_platform/claude-test")]
#[case::mantle("bedrock_mantle/claude-test")]
#[tokio::test]
async fn native_streams_relay_every_sse_byte(call: MessagesCall, #[case] model: &str) {
    let sse = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let upstream =
        upstream([ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")]).await;
    let response = messages_route(no_secrets())
        .execute(
            MessagesCall {
                api_key: Some("key".into()),
                api_base: Some(upstream.uri()),
                custom_llm_provider: None,
                ..with_fields(
                    call,
                    json!({"model": model, "stream": true, "workspace_id": "workspace"}),
                )
            },
            &(),
            None,
        )
        .await
        .unwrap();
    let MessagesCallResponse::Stream { chunks, .. } = response else {
        panic!("expected SSE");
    };
    assert_eq!(
        chunks.try_collect::<Vec<_>>().await.unwrap().concat(),
        sse.as_bytes()
    );
    assert_eq!(only_request(&upstream).await.json()["stream"], true);
}

#[rstest]
#[case::missing(false)]
#[case::configured(true)]
#[tokio::test]
async fn deployment_workspace_is_required_and_sent_as_a_header(
    call: MessagesCall,
    #[case] configured: bool,
) {
    let upstream = upstream([message_response()]).await;
    let params = if configured {
        serde_json::from_value(json!({"model": "bedrock/claude_platform/claude-test", "workspace_id": "deployment-workspace"})).unwrap()
    } else {
        Default::default()
    };
    let result = run(MessagesCall {
        api_key: Some("key".into()),
        api_base: Some(upstream.uri()),
        litellm_params: params,
        ..with_model(call, "bedrock/claude_platform/claude-test")
    })
    .await;
    if !configured {
        assert!(matches!(result, Err(Error::Auth(_))));
        assert!(received(&upstream).await.is_empty());
        return;
    }
    assert!(matches!(result.unwrap(), MessagesCallResponse::Complete(_)));
    let sent = only_request(&upstream).await;
    assert_eq!(
        sent.header("anthropic-workspace-id"),
        Some("deployment-workspace")
    );
    assert_eq!(sent.json().get("workspace_id"), None);
}

const INVOKE_MODEL: &str = "bedrock/us.anthropic.claude-opus-5-5";

fn signed_invoke_call(call: MessagesCall, base: String, fields: Value) -> MessagesCall {
    MessagesCall {
        api_base: Some(base),
        custom_llm_provider: None,
        litellm_params: litellm_router_types::LitellmParams {
            aws: litellm_auth::AwsParams {
                aws_access_key_id: Some("static-access".into()),
                aws_secret_access_key: Some("static-secret".into()),
                aws_region_name: Some("us-east-1".into()),
                ..Default::default()
            },
            ..Default::default()
        },
        provider_specific_header: Some(
            serde_json::from_value(json!({"custom_llm_provider": "bedrock", "extra_headers": {
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "context-1m-2025-08-07,unknown-beta"
            }}))
            .unwrap(),
        ),
        ..with_fields(with_model(call, INVOKE_MODEL), fields)
    }
}

fn aws_event(chunk: &Value) -> Vec<u8> {
    use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
    use base64::Engine;
    let payload =
        json!({"bytes": base64::engine::general_purpose::STANDARD.encode(chunk.to_string())});
    let message = Message::new(serde_json::to_vec(&payload).unwrap())
        .add_header(Header::new(
            ":event-type",
            HeaderValue::String("chunk".into()),
        ))
        .add_header(Header::new(
            ":message-type",
            HeaderValue::String("event".into()),
        ));
    let mut wire = Vec::new();
    aws_smithy_eventstream::frame::write_message_to(&message, &mut wire).unwrap();
    wire
}

#[rstest]
#[tokio::test]
async fn invoke_sends_a_signed_bedrock_body_and_returns_the_message(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let message = run_message(signed_invoke_call(
        call,
        upstream.uri(),
        json!({"metadata": {"user_id": "u1"}, "speed": "fast"}),
    ))
    .await;
    assert_eq!(message.model, MODEL);
    let request = only_request(&upstream).await;
    assert_eq!(
        request.url.path(),
        "/model/us.anthropic.claude-opus-5-5/invoke"
    );
    assert!(
        request
            .header("authorization")
            .unwrap()
            .starts_with("AWS4-HMAC-SHA256 Credential=static-access/")
    );
    assert_eq!(
        request.json(),
        json!({
            "anthropic_version": "bedrock-2023-05-31",
            "anthropic_beta": ["context-1m-2025-08-07"],
            "max_tokens": 16,
            "metadata": {"user_id": "u1"},
            "messages": [{"role": "user", "content": "hi"}]
        })
    );
}

#[rstest]
#[tokio::test]
async fn invoke_streams_decode_aws_events_into_anthropic_sse(call: MessagesCall) {
    let chunks = [
        json!({"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": MODEL, "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 3, "output_tokens": 1}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}),
        json!({"type": "message_stop", "amazon-bedrock-invocationMetrics": {"inputTokenCount": 3, "outputTokenCount": 2}}),
    ];
    let body: Vec<u8> = chunks.iter().flat_map(aws_event).collect();
    let upstream = upstream([
        ResponseTemplate::new(200).set_body_raw(body, "application/vnd.amazon.eventstream")
    ])
    .await;
    let response = run(signed_invoke_call(
        call,
        upstream.uri(),
        json!({"stream": true}),
    ))
    .await
    .unwrap();
    let MessagesCallResponse::Stream { chunks, .. } = response else {
        panic!("expected SSE");
    };
    let sse = String::from_utf8(chunks.try_collect::<Vec<_>>().await.unwrap().concat()).unwrap();
    let events: Vec<&str> = sse
        .lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .collect();
    assert_eq!(
        events,
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]
    );
    assert!(sse.contains(r#""text":"hi""#));
    let request = only_request(&upstream).await;
    assert_eq!(
        request.url.path(),
        "/model/us.anthropic.claude-opus-5-5/invoke-with-response-stream"
    );
    assert_eq!(request.json().get("stream"), None);
}
