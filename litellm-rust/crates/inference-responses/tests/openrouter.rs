use std::sync::Arc;

use futures_util::TryStreamExt;
use litellm_inference_responses::{
    Error,
    types::{ResponsesCall, ResponsesOutput},
};
use litellm_inference_testing::{RecordingSecrets, no_secrets};
use rstest::{fixture, rstest};
use serde_json::json;
use wiremock::ResponseTemplate;

mod support;
use support::*;

#[fixture]
fn call() -> ResponsesCall {
    ResponsesCall {
        model: "openrouter/anthropic/test-model".into(),
        input: json!("hello"),
        optional_params: Default::default(),
        api_key: None,
        api_base: None,
        custom_llm_provider: None,
        api_version: None,
        litellm_params: Default::default(),
        extra_headers: None,
        timeout: None,
    }
}

fn response_body() -> serde_json::Value {
    json!({"id": "resp_1", "model": "anthropic/test-model", "output": [{"type": "message", "content": []}], "usage": {"total_tokens": 7, "cost": 0.00017}})
}

#[rstest]
#[case::explicit_key(Some("explicit"), &[], "Bearer explicit")]
#[case::managed_key(None, &[("OPENROUTER_API_KEY", "managed")], "Bearer managed")]
#[case::legacy_name(None, &[("OR_API_KEY", "legacy")], "Bearer legacy")]
#[tokio::test]
async fn the_payload_goes_to_the_openrouter_responses_endpoint_with_a_bearer(
    call: ResponsesCall,
    #[case] api_key: Option<&str>,
    #[case] secrets: &[(&str, &str)],
    #[case] expected: &str,
) {
    let upstream = upstream([json_response(response_body())]).await;
    let output = responses_route(Arc::new(RecordingSecrets::new(secrets.iter().copied())))
        .execute(
            ResponsesCall {
                api_key: api_key.map(str::to_owned),
                api_base: Some(format!("{}/api/v1", upstream.uri())),
                optional_params: json!({"max_output_tokens": 16})
                    .as_object()
                    .unwrap()
                    .clone(),
                ..call
            },
            &(),
            None,
        )
        .await
        .unwrap();
    let ResponsesOutput::Complete(response) = output else {
        panic!("expected a complete response");
    };
    assert_eq!(serde_json::to_value(response).unwrap(), response_body());
    let request = only_request(&upstream).await;
    assert_eq!(request.url.path(), "/api/v1/responses");
    assert_eq!(request.header("authorization"), Some(expected));
    assert_eq!(
        request.json(),
        json!({"max_output_tokens": 16, "model": "anthropic/test-model", "input": "hello"})
    );
}

#[rstest]
#[tokio::test]
async fn the_secret_manager_supplies_the_base_and_key(call: ResponsesCall) {
    let upstream = upstream([json_response(response_body())]).await;
    let base = format!("{}/api/v1/", upstream.uri());
    let secrets = Arc::new(RecordingSecrets::new([
        ("OPENROUTER_API_KEY", "managed"),
        ("OPENROUTER_API_BASE", base.as_str()),
    ]));
    responses_route(secrets.clone())
        .execute(call, &(), None)
        .await
        .unwrap();
    assert_eq!(
        secrets.requested(),
        ["OPENROUTER_API_KEY", "OR_API_KEY", "OPENROUTER_API_BASE"]
    );
    let request = only_request(&upstream).await;
    assert_eq!(request.url.path(), "/api/v1/responses");
    assert_eq!(request.header("authorization"), Some("Bearer managed"));
}

#[rstest]
#[tokio::test]
async fn a_missing_key_fails_before_transport(call: ResponsesCall) {
    let upstream = upstream([json_response(response_body())]).await;
    let result = responses_route(no_secrets())
        .execute(
            ResponsesCall {
                api_base: Some(upstream.uri()),
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
async fn streaming_relays_every_openrouter_event_untouched(call: ResponsesCall) {
    let sse = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"cost\":0.0002}}}\n\n";
    let upstream =
        upstream([ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")]).await;
    let output = responses_route(no_secrets())
        .execute(
            ResponsesCall {
                api_key: Some("key".into()),
                api_base: Some(upstream.uri()),
                optional_params: json!({"stream": true}).as_object().unwrap().clone(),
                ..call
            },
            &(),
            None,
        )
        .await
        .unwrap();
    let ResponsesOutput::Stream { chunks, .. } = output else {
        panic!("expected SSE");
    };
    assert_eq!(
        chunks.try_collect::<Vec<_>>().await.unwrap().concat(),
        sse.as_bytes()
    );
    let request = only_request(&upstream).await;
    assert_eq!(request.json()["stream"], true);
    assert_eq!(request.json()["model"], "anthropic/test-model");
}
