#![cfg(feature = "switchyard")]

mod support;

use std::sync::Arc;

use litellm_config::Config;
use litellm_inference::resources::CoreResources;
use litellm_gateway_auth::{AccessRequest, Permissions};
use litellm_gateway_inference::{Gateway, ModelRouter};
use litellm_http::{HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver};
use rstest::rstest;
use serde_json::json;
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
            "stream": true
        }),
    )
    .await;
    assert_eq!(response.status(), 501);

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
