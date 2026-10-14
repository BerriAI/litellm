#![cfg(feature = "switchyard")]

mod support;

use std::{sync::Arc, time::Duration};

use futures_util::StreamExt;
use litellm_config::Config;
use litellm_gateway_auth::{AccessRequest, Permissions};
use litellm_gateway_inference::{Gateway, ModelRouter};
use litellm_http::{HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver};
use litellm_inference::resources::CoreResources;
use rstest::rstest;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, method, path},
};

#[rstest]
#[tokio::test]
async fn configured_switchyard_routes_answer_and_fall_back() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({"model": "judge-model"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "judge-model",
            "choices": [{
                "message": {
                    "content": r#"{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#
                }
            }]
        })))
        .expect(1)
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "fast-model",
            "messages": [{"role": "user", "content": "fix this"}],
            "service_tier": "flex"
        })))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({
            "error": {"message": "unavailable"}
        })))
        .expect(1)
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "strong-model",
            "messages": [{"role": "user", "content": "fix this"}],
            "service_tier": "flex"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "strong-model",
            "choices": [{
                "message": {"content": "fallback answer"}
            }]
        })))
        .expect(1)
        .mount(&upstream)
        .await;

    let config = Config::from_yaml(&format!(
        r#"
model_list:
  - model_name: fast
    litellm_params:
      model: fast-model
      api_base: {base}/v1
      api_key: test-key
      custom_llm_provider: openai_like
      switchyard_wire_format: openai_chat
  - model_name: strong
    litellm_params:
      model: strong-model
      api_base: {base}/v1
      api_key: test-key
      custom_llm_provider: openai_like
      switchyard_wire_format: openai_chat
  - model_name: judge
    litellm_params:
      model: judge-model
      api_base: {base}/v1
      api_key: test-key
      custom_llm_provider: openai_like
      switchyard_wire_format: openai_chat
  - model_name: coding-router
    litellm_params:
      model: switchyard/algorithm
      switchyard_config:
        type: llm_classifier
        mode: capability
        classifier_target: judge
        weak_target: fast
        strong_target: strong
        base_threshold: 0.5
  - model_name: noop-router
    litellm_params:
      model: switchyard/algorithm
      switchyard_config:
        type: noop
"#,
        base = upstream.uri()
    ))
    .expect("valid config");
    let app = switchyard_app(
        &config,
        Permissions::Only(Arc::from([
            AccessRequest::Model {
                name: "coding-router".into(),
                deployment: "switchyard/algorithm".into(),
            },
            AccessRequest::Model {
                name: "noop-router".into(),
                deployment: "switchyard/algorithm".into(),
            },
        ])),
    );

    let response = support::post(
        app.clone(),
        "/v1/chat/completions",
        json!({
            "model": "coding-router",
            "messages": [{"role": "user", "content": "fix this"}],
            "service_tier": "flex"
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body = support::json(response).await;
    assert_eq!(body["model"], "strong");
    assert_eq!(body["choices"][0]["message"]["content"], "fallback answer");

    let response = support::post(
        app,
        "/v1/chat/completions",
        json!({
            "model": "noop-router",
            "messages": [{"role": "user", "content": "health check"}]
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body = support::json(response).await;
    assert_eq!(body["choices"][0]["message"]["content"], "OK");
}

// The caller API and selected target API are independent and use the same host bridge.
#[rstest]
#[tokio::test]
async fn configured_switchyard_routes_translate_responses_and_messages() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({"model": "responses-model"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1",
            "model": "responses-model",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "responses answer"}]
            }],
            "provider_extra": true,
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        })))
        .expect(2)
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({"model": "messages-model"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "messages-model",
            "content": [{"type": "text", "text": "messages answer"}],
            "provider_extra": true,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .expect(1)
        .mount(&upstream)
        .await;

    let config = Config::from_yaml(&format!(
        r#"
model_list:
  - model_name: responses-target
    litellm_params:
      model: openai/responses-model
      api_base: {base}
      api_key: test-key
      custom_llm_provider: openai
      switchyard_wire_format: openai_responses
  - model_name: messages-target
    litellm_params:
      model: anthropic/messages-model
      api_base: {base}
      api_key: test-key
      custom_llm_provider: anthropic
      switchyard_wire_format: anthropic_messages
  - model_name: responses-router
    litellm_params:
      model: switchyard/algorithm
      switchyard_config:
        type: passthrough
        target: responses-target
  - model_name: messages-router
    litellm_params:
      model: switchyard/algorithm
      switchyard_config:
        type: passthrough
        target: messages-target
"#,
        base = upstream.uri()
    ))
    .expect("valid config");
    let app = switchyard_app(&config, Permissions::All);

    let response = support::post(
        app.clone(),
        "/v1/responses",
        json!({"model": "responses-router", "input": "hello"}),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body = support::json(response).await;
    assert_eq!(body["model"], "responses-target");
    assert_eq!(body["output"][0]["content"][0]["text"], "responses answer");
    assert_eq!(body["provider_extra"], true);

    let response = support::post(
        app.clone(),
        "/v1/messages",
        json!({
            "model": "messages-router",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body = support::json(response).await;
    assert_eq!(body["model"], "messages-target");
    assert_eq!(body["content"][0]["text"], "messages answer");
    assert_eq!(body["provider_extra"], true);

    let response = support::post(
        app,
        "/v1/chat/completions",
        json!({
            "model": "responses-router",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body = support::json(response).await;
    assert_eq!(body["model"], "responses-target");
    assert_eq!(body["choices"][0]["message"]["content"], "responses answer");
}

#[rstest]
#[tokio::test]
async fn streaming_chat_target_keeps_litellm_core_rejection() {
    let upstream = MockServer::start().await;
    let config = Config::from_yaml(&format!(
        r#"
model_list:
  - model_name: target
    litellm_params: {{ model: test-model, api_base: {base}/v1,
      custom_llm_provider: openai_like, switchyard_wire_format: openai_chat }}
  - model_name: route
    litellm_params: {{ model: switchyard/algorithm,
      switchyard_config: {{ type: passthrough, target: target }} }}
"#,
        base = upstream.uri(),
    ))
    .expect("valid config");
    let response = support::post(
        switchyard_app(&config, Permissions::All),
        "/v1/chat/completions",
        json!({
            "model": "route",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(response.status(), 501);
    let body = support::json(response).await;
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("streaming")
    );
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn switchyard_streams_across_formats_and_cancels_on_drop() {
    let (base, upstream) = streaming_responses_upstream().await;
    let config = Config::from_yaml(&format!(
        r#"
model_list:
  - model_name: responses-target
    litellm_params: {{ model: openai/responses-model, api_base: {base}, api_key: test-key,
      custom_llm_provider: openai, switchyard_wire_format: openai_responses }}
  - model_name: responses-router
    litellm_params: {{ model: switchyard/algorithm,
      switchyard_config: {{ type: passthrough, target: responses-target }} }}
"#
    ))
    .expect("valid config");
    let response = tokio::time::timeout(
        Duration::from_secs(2),
        support::post(
            switchyard_app(&config, Permissions::All),
            "/v1/messages",
            json!({
                "model": "responses-router",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
                "stream": true
            }),
        ),
    )
    .await
    .expect("response starts while upstream remains open");

    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let mut body = response.into_body().into_data_stream();
    let first = tokio::time::timeout(Duration::from_secs(2), body.next())
        .await
        .expect("first event arrives while upstream remains open")
        .expect("stream has a first event")
        .expect("first event is readable");
    assert!(
        std::str::from_utf8(&first)
            .expect("event is UTF-8")
            .contains("event: message_start")
    );

    drop(body);
    tokio::time::timeout(Duration::from_secs(2), upstream)
        .await
        .expect("dropping the caller closes the upstream stream")
        .expect("upstream task succeeds");
}

async fn streaming_responses_upstream() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let base = format!(
        "http://{}",
        listener.local_addr().expect("upstream address")
    );
    let upstream = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let response = concat!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n",
            "content-length: 4096\r\n\r\n",
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":",
            "{\"id\":\"resp_1\",\"model\":\"responses-model\"}}\n\n"
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write first event");
        let mut pending = [0; 1];
        while matches!(socket.read(&mut pending).await, Ok(read) if read > 0) {}
    });
    (base, upstream)
}

fn switchyard_app(config: &Config, permissions: Permissions) -> axum::Router {
    let resources = CoreResources::new(Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver))));
    let http = Resolution::from(&HttpSettings::default()).config;
    let gateway = Gateway::new(
        resources,
        http,
        Arc::new(support::NoSecrets),
        ModelRouter::from_model_list(&config.model_list),
    )
    .expect("gateway")
    .with_switchyard_models(&config.model_list)
    .expect("Switchyard routes");
    support::authenticated_app(gateway, permissions, None)
}
