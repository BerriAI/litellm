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
