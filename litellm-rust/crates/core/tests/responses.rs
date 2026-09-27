use std::sync::Arc;

use futures_util::TryStreamExt;
use litellm_core::responses::{
    route::Responses,
    types::{ResponsesCall, ResponsesOutput},
};
use litellm_host::{call::HostedCompletion, event::CallEvent};
use rstest::{fixture, rstest};
use serde_json::json;
use wiremock::ResponseTemplate;

mod support;
use support::*;

#[fixture]
fn call() -> ResponsesCall {
    ResponsesCall {
        model: "openai/test-model".into(),
        input: json!("hello"),
        optional_params: Default::default(),
        api_key: Some("test-key".into()),
        api_base: None,
        custom_llm_provider: None,
        extra_headers: None,
        timeout: None,
    }
}

#[rstest]
#[case::direct(false)]
#[case::hosted(true)]
#[tokio::test]
async fn http_responses_share_execution_and_hooks(call: ResponsesCall, #[case] hosted: bool) {
    let body = json!({"id": "response-1", "model": "test-model", "output": [{"type":"message", "content":[]}], "usage":{"total_tokens":7}, "provider_extra": true});
    let upstream = upstream([json_response(body.clone())]).await;
    let host = RecordingCall::<Responses>::new(ResponsesCall {
        api_base: Some(upstream.uri()),
        ..call
    });
    let response = if hosted {
        let machine = client().responses_machine().unwrap();
        let HostedCompletion::Complete(response) =
            litellm_host::in_process::run_hosted(machine(host.request().unwrap()), host.runtime())
                .await
                .unwrap()
        else {
            panic!()
        };
        response
    } else {
        let call = host.request.lock().unwrap().take().unwrap();
        let ResponsesOutput::Complete(response) =
            client().responses_with_hooks(call, &host).await.unwrap()
        else {
            panic!()
        };
        response
    };
    assert_eq!(serde_json::to_value(response).unwrap(), body);
    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/responses");
    assert_eq!(sent.header("authorization"), Some("Bearer test-key"));
    assert_eq!(sent.header("x-hook"), Some("called"));
    assert_eq!(sent.json(), json!({"model":"test-model", "input":"hello"}));
    assert!(matches!(
        &host.events.0.lock().unwrap()[..],
        [
            CallEvent::Started { .. },
            CallEvent::Machine(_),
            CallEvent::Succeeded { .. }
        ]
    ));
}

#[rstest]
#[case::direct(false)]
#[case::hosted(true)]
#[tokio::test]
async fn streaming_keeps_headers_and_bytes_and_finishes_after_consumption(
    call: ResponsesCall,
    #[case] hosted: bool,
) {
    let body = "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n";
    let upstream = upstream([ResponseTemplate::new(200)
        .insert_header("x-request-id", "response-stream")
        .set_body_raw(body, "text/event-stream")])
    .await;
    let host = RecordingCall::<Responses>::new(ResponsesCall {
        api_base: Some(upstream.uri()),
        optional_params: json!({"stream":true}).as_object().unwrap().clone(),
        ..call
    });
    let (headers, bytes) = if hosted {
        let machine = client().responses_machine().unwrap();
        assert_eq!(
            litellm_host::in_process::run_hosted(machine(host.request().unwrap()), host.runtime())
                .await
                .unwrap(),
            HostedCompletion::StreamEnded
        );
        (
            host.head.lock().unwrap().take().unwrap().headers,
            host.chunks.lock().unwrap().concat(),
        )
    } else {
        let call = host.request.lock().unwrap().take().unwrap();
        let ResponsesOutput::Stream { head, chunks } =
            client().responses_with_hooks(call, &host).await.unwrap()
        else {
            panic!()
        };
        assert_eq!(host.events.0.lock().unwrap().len(), 1);
        (
            head.headers,
            chunks.try_collect::<Vec<_>>().await.unwrap().concat(),
        )
    };
    assert!(headers.contains(&("x-request-id".into(), "response-stream".into())));
    assert_eq!(bytes, body.as_bytes());
    assert!(matches!(
        &host.events.0.lock().unwrap()[..],
        [CallEvent::Started { .. }, CallEvent::Succeeded { .. }]
    ));
}

#[rstest]
#[case::http(429, json!({"error":"limited"}))]
#[case::invalid_response(200, json!({"unexpected":true}))]
#[tokio::test]
async fn provider_failures_emit_failure_once(
    call: ResponsesCall,
    #[case] status: u16,
    #[case] body: serde_json::Value,
) {
    let upstream = upstream([ResponseTemplate::new(status).set_body_json(body)]).await;
    let host = RecordingCall::<Responses>::new(ResponsesCall {
        api_base: Some(upstream.uri()),
        ..call
    });
    let call = host.request.lock().unwrap().take().unwrap();
    assert!(client().responses_with_hooks(call, &host).await.is_err());
    assert_eq!(received(&upstream).await.len(), 1);
    let events = host.events.0.lock().unwrap();
    assert!(matches!(events.last(), Some(CallEvent::Failed { .. })));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                CallEvent::Failed { .. } | CallEvent::Succeeded { .. }
            ))
            .count(),
        1
    );
}

#[rstest]
#[case::explicit(true)]
#[case::from_secrets(false)]
#[tokio::test]
async fn credentials_and_endpoint_are_resolved_only_when_needed(
    call: ResponsesCall,
    #[case] explicit: bool,
) {
    let upstream = upstream([json_response(
        json!({"id":"response", "model":"test-model", "output":[]}),
    )])
    .await;
    let base = upstream.uri();
    let key = "resolved-test-key";
    let secrets = Arc::new(RecordingSecrets::new([
        ("OPENAI_API_KEY", key),
        ("OPENAI_BASE_URL", base.as_str()),
    ]));
    let call = ResponsesCall {
        api_key: explicit.then(|| key.into()),
        api_base: explicit.then(|| base.clone()),
        ..call
    };
    client_with_secrets(secrets.clone())
        .responses(call)
        .await
        .unwrap();
    assert_eq!(
        only_request(&upstream).await.header("authorization"),
        Some(format!("Bearer {key}").as_str())
    );
    if explicit {
        assert!(secrets.requested().is_empty());
    } else {
        assert!(secrets.requested().contains(&"OPENAI_API_KEY".into()));
        assert!(secrets.requested().contains(&"OPENAI_BASE_URL".into()));
    }
}

#[rstest]
#[case::provider("other/model", "other")]
#[case::conflicting_prefix("other/model", "openai")]
#[tokio::test]
async fn unsupported_providers_fail_before_secrets_or_transport(
    call: ResponsesCall,
    #[case] model: &str,
    #[case] provider: &str,
) {
    let upstream = upstream([]).await;
    let secrets = Arc::new(RecordingSecrets::failing());
    let call = ResponsesCall {
        model: model.into(),
        custom_llm_provider: Some(provider.into()),
        api_base: Some(upstream.uri()),
        ..call
    };
    assert!(
        client_with_secrets(secrets.clone())
            .responses(call)
            .await
            .is_err()
    );
    assert!(secrets.requested().is_empty());
    assert!(received(&upstream).await.is_empty());
}
