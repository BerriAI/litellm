mod support;

use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

const MESSAGE_SSE: &str = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

#[rstest]
#[case::anthropic("anthropic/test-model", "/v1/messages", true)]
#[case::azure("azure_ai/test-model", "/anthropic/v1/messages", false)]
#[case::deepseek("deepseek/test-model", "/anthropic/v1/messages", true)]
#[tokio::test]
async fn messages_reaches_the_provider_and_preserves_json_or_sse(
    #[case] model: &str,
    #[case] upstream_path: &str,
    #[case] anthropic_headers: bool,
    #[values(false, true)] streaming: bool,
) {
    let upstream = MockServer::start().await;
    let message = json!({"id": "msg_test", "type": "message", "role": "assistant",
        "model": "test-model", "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}});
    let sse = MESSAGE_SSE;
    let template = if streaming {
        ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")
    } else {
        ResponseTemplate::new(200).set_body_json(message.clone())
    };
    let messages = json!([{"role": "user", "content": "hi"}]);
    let mut mock = Mock::given(method("POST"))
        .and(path(upstream_path))
        .and(header("x-api-key", "test-key"));
    if anthropic_headers {
        mock = mock
            .and(header("anthropic-beta", "test-feature"))
            .and(header("anthropic-version", "test-version"));
    }
    mock.and(body_json(
        json!({"model": "test-model", "messages": messages, "max_tokens": 16, "stream": streaming}),
    ))
    .respond_with(template)
    .expect(1)
    .mount(&upstream)
    .await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json").header("anthropic-beta", "test-feature")
        .header("anthropic-version", "test-version")
        .header("x-api-key", "caller-key")
        .header("authorization", "Bearer proxy-key")
        .header("x-request-id", "caller-request")
        .body(Body::from(json!({"model": "public/model", "messages": messages, "max_tokens": 16, "stream": streaming}).to_string())).unwrap();
    let response = support::app(model, &upstream.uri())
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    if streaming {
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        assert_eq!(to_bytes(response.into_body(), 4096).await.unwrap(), sse);
    } else {
        let body = support::json(response).await;
        assert_eq!(body["content"], message["content"]);
        assert_eq!(body["usage"], message["usage"]);
    }
    let requests = upstream.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert!(!requests[0].headers.contains_key("authorization"));
    assert!(!requests[0].headers.contains_key("x-request-id"));
}

#[rstest]
#[case::the_defaults_fill_what_the_client_omits(
    json!({}),
    json!({"temperature": 0.2, "top_p": 0.9, "top_k": 5, "max_tokens": 7, "stop_sequences": ["default"]}),
)]
#[case::client_values_win(
    json!({"temperature": 0.7, "top_p": 0.8, "top_k": 6, "max_tokens": 16, "stop_sequences": ["client"]}),
    json!({"temperature": 0.7, "top_p": 0.8, "top_k": 6, "max_tokens": 16, "stop_sequences": ["client"]}),
)]
#[case::a_client_null_clears_each_default(
    json!({"temperature": null, "top_p": null, "top_k": null, "stop_sequences": null, "max_tokens": 16}),
    json!({"max_tokens": 16}),
)]
#[tokio::test]
async fn deployment_sampling_params_sit_under_the_client_body(
    #[case] client_params: Value,
    #[case] expected_params: Value,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(messages_request("test-model", expected_params)))
        .respond_with(message_response())
        .expect(1)
        .mount(&upstream)
        .await;
    let app = support::app_with_litellm_params(
        "anthropic/test-model",
        &upstream.uri(),
        json!({
            "model": "anthropic/test-model",
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 5,
            "max_tokens": 7,
            "stop_sequences": ["default"],
            "a_litellm_only_setting": "stays-out-of-the-provider-body",
        }),
    );

    let response = support::post(
        app,
        "/v1/messages",
        messages_request("public/model", client_params),
    )
    .await;

    assert_eq!(response.status(), 200);
}

#[rstest]
#[case::max_tokens("max_tokens", json!(7))]
#[case::metadata("metadata", json!({"user_id": "deployment-user"}))]
#[case::stop_sequences("stop_sequences", json!(["default"]))]
#[case::stream("stream", json!(false))]
#[case::system("system", json!("Answer in one word."))]
#[case::temperature("temperature", json!(0.2))]
#[case::thinking("thinking", json!({"type": "enabled", "budget_tokens": 1024}))]
#[case::tool_choice("tool_choice", json!({"type": "auto"}))]
#[case::top_k("top_k", json!(5))]
#[case::inference_geo("inference_geo", json!("us"))]
#[case::top_p("top_p", json!(0.9))]
#[case::mcp_servers(
    "mcp_servers",
    json!([{"type": "url", "url": "https://mcp.example.com/sse", "name": "example"}]),
)]
#[case::context_management(
    "context_management",
    json!({"edits": [{"type": "clear_tool_uses_20250919"}]}),
)]
#[case::compaction("compaction", json!({"type": "summarize"}))]
#[case::container(
    "container",
    json!({"skills": [{"type": "anthropic", "skill_id": "pptx", "version": "latest"}]}),
)]
#[case::output_format(
    "output_format",
    json!({"type": "json_schema", "schema": {"type": "object", "properties": {}}}),
)]
#[case::speed("speed", json!("fast"))]
#[case::output_config("output_config", json!({"effort": "low"}))]
#[case::cache_control("cache_control", json!({"type": "ephemeral"}))]
#[case::reasoning_effort("reasoning_effort", json!("low"))]
#[case::safeguards(
    "safeguards",
    json!([{"type": "dangerous_tool_use", "classifier_context": {"v": 1}}]),
)]
#[case::service_tier("service_tier", json!("auto"))]
#[tokio::test]
async fn a_deployment_default_reaches_the_provider_as_the_same_client_value(
    #[case] name: &str,
    #[case] value: Value,
) {
    let required = required_params_besides(name);
    let from_deployment = upstream_body(json!({name: value.clone()}), required.clone()).await;
    let from_client = upstream_body(json!({}), with_param(required, name, value)).await;
    let baseline = upstream_body(json!({}), json!({"max_tokens": 16})).await;

    assert_eq!(from_deployment, from_client);
    assert_ne!(from_deployment, baseline);
}

#[rstest]
#[case::a_client_thinking_wins(
    json!({"thinking": {"type": "enabled", "budget_tokens": 1024}}),
    json!({"thinking": {"type": "disabled"}}),
    json!({"thinking": {"type": "disabled"}}),
)]
#[case::a_client_null_clears_the_thinking_default(
    json!({"thinking": {"type": "enabled", "budget_tokens": 1024}}),
    json!({"thinking": null}),
    json!({}),
)]
#[case::a_client_system_wins(
    json!({"system": "From the deployment."}),
    json!({"system": "From the client."}),
    json!({"system": "From the client."}),
)]
#[case::a_client_null_clears_the_system_default(
    json!({"system": "From the deployment."}),
    json!({"system": null}),
    json!({}),
)]
#[case::deployment_tools_come_before_client_tools(
    json!({"tools": [tool("deployment_lookup")]}),
    json!({"tools": [tool("client_lookup")]}),
    json!({"tools": [tool("deployment_lookup"), tool("client_lookup")]}),
)]
#[case::deployment_tools_fill_omitted_tools(
    json!({"tools": [tool("deployment_lookup")]}),
    json!({}),
    json!({"tools": [tool("deployment_lookup")]}),
)]
#[case::deployment_tools_replace_null_tools(
    json!({"tools": [tool("deployment_lookup")]}),
    json!({"tools": null}),
    json!({"tools": [tool("deployment_lookup")]}),
)]
#[case::a_client_tool_choice_wins(
    json!({"tools": [tool("deployment_lookup")], "tool_choice": {"type": "auto"}}),
    json!({"tool_choice": {"type": "any"}}),
    json!({"tools": [tool("deployment_lookup")], "tool_choice": {"type": "any"}}),
)]
#[case::extra_body_stays_out(
    json!({"extra_body": {"deployment_only": true}}),
    json!({}),
    json!({}),
)]
#[case::an_unknown_key_stays_out(
    json!({"a_litellm_only_setting": "stays-out-of-the-provider-body"}),
    json!({}),
    json!({}),
)]
#[tokio::test]
async fn a_deployment_param_reaches_the_provider_as_its_client_equivalent(
    #[case] deployment_params: Value,
    #[case] client_params: Value,
    #[case] client_equivalent: Value,
) {
    assert_eq!(
        upstream_body(
            deployment_params,
            with_param(client_params, "max_tokens", json!(16))
        )
        .await,
        upstream_body(
            json!({}),
            with_param(client_equivalent, "max_tokens", json!(16))
        )
        .await,
    );
}

#[tokio::test]
async fn deployment_tools_leave_client_tools_that_are_not_an_array_as_sent() {
    let upstream = MockServer::start().await;
    let app = support::app_with_litellm_params(
        "anthropic/test-model",
        &upstream.uri(),
        deployment_params(json!({"tools": [tool("deployment_lookup")]})),
    );

    let response = support::post(
        app,
        "/v1/messages",
        messages_request(
            "public/model",
            json!({"max_tokens": 16, "tools": tool("client_lookup")}),
        ),
    )
    .await;

    assert_eq!(response.status(), 400);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn a_deployment_stream_default_streams_a_request_that_omits_stream() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(messages_request(
            "test-model",
            json!({"max_tokens": 16, "stream": true}),
        )))
        .respond_with(ResponseTemplate::new(200).set_body_raw(MESSAGE_SSE, "text/event-stream"))
        .expect(1)
        .mount(&upstream)
        .await;
    let app = support::app_with_litellm_params(
        "anthropic/test-model",
        &upstream.uri(),
        deployment_params(json!({"stream": true})),
    );

    let response = support::post(
        app,
        "/v1/messages",
        messages_request("public/model", json!({"max_tokens": 16})),
    )
    .await;

    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(
        to_bytes(response.into_body(), 4096).await.unwrap(),
        MESSAGE_SSE
    );
}

async fn upstream_body(deployment: Value, client: Value) -> Value {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(message_response())
        .mount(&upstream)
        .await;
    let app = support::app_with_shaping(
        "anthropic/test-model",
        &upstream.uri(),
        deployment_params(deployment),
        json!({"capabilities": {
            "supports_reasoning": true,
            "supports_adaptive_thinking": true,
            "supports_output_config": true,
            "supports_speed": true,
            "effort_tiers": {"low": true, "medium": true, "high": true},
        }}),
    );
    let response = support::post(
        app,
        "/v1/messages",
        messages_request("public/model", client),
    )
    .await;
    assert_eq!(response.status(), 200, "{}", support::json(response).await);
    upstream.received_requests().await.unwrap()[0]
        .body_json()
        .unwrap()
}

fn message_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
        "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }))
}

fn tool(name: &str) -> Value {
    json!({"name": name, "description": "Looks something up.", "input_schema": {"type": "object"}})
}

fn required_params_besides(name: &str) -> Value {
    match name {
        "max_tokens" => json!({}),
        _ => json!({"max_tokens": 16}),
    }
}

fn with_param(params: Value, name: &str, value: Value) -> Value {
    let mut with_param = params;
    with_param[name] = value;
    with_param
}

fn deployment_params(params: Value) -> Value {
    let mut litellm_params = params;
    litellm_params["model"] = json!("anthropic/test-model");
    litellm_params
}

fn messages_request(model: &str, params: Value) -> Value {
    let mut request = params;
    request["model"] = json!(model);
    request["messages"] = json!([{"role": "user", "content": "hi"}]);
    request
}

#[tokio::test]
async fn invalid_messages_stays_an_anthropic_error() {
    let response = support::post(
        support::app("anthropic/test-model", "http://127.0.0.1:1"),
        "/v1/messages",
        json!({"model": "public/model", "messages": "invalid", "max_tokens": 16}),
    )
    .await;
    assert_eq!(response.status(), 400);
    let body = support::json(response).await;
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

const FIRST_EVENT: &str = "event: message_start\ndata: {}\n\n";

#[tokio::test]
async fn a_stream_that_fails_after_opening_ends_with_an_sse_error_frame() {
    let base = support::truncating_upstream(FIRST_EVENT).await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16, "stream": true})
            .to_string(),
        ))
        .unwrap();

    let response = support::app("anthropic/test-model", &base)
        .oneshot(request)
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    let frame = text
        .strip_prefix(FIRST_EVENT)
        .and_then(|rest| rest.strip_prefix("event: error\ndata: "))
        .unwrap_or_else(|| panic!("the delivered event then one error frame, got {text:?}"));
    let error: serde_json::Value = serde_json::from_str(frame.trim_end()).unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["type"], "api_error");
}

#[rstest]
#[tokio::test]
async fn hosted_provider_failure_preserves_status_body_and_request_id() {
    let upstream = MockServer::start().await;
    let error =
        json!({"type": "error", "error": {"type": "rate_limit_error", "message": "retry later"}});
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_json(error.clone()))
        .expect(1)
        .mount(&upstream)
        .await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .header("x-request-id", "host-http-request")
        .body(Body::from(json!({
            "model": "public/model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16,
        }).to_string()))
        .unwrap();
    let response = support::app("anthropic/test-model", &upstream.uri())
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(response.status(), 429);
    let body = support::json(response).await;
    assert_eq!(body["type"], error["type"]);
    assert_eq!(body["error"], error["error"]);
    assert_eq!(body["request_id"], "host-http-request");
}

#[rstest]
#[case::messages("/v1/messages")]
#[case::chat("/v1/chat/completions")]
#[case::model_path("/engines/path-model/chat/completions")]
#[case::ocr("/ocr")]
#[case::transcription("/audio/transcriptions")]
#[tokio::test]
async fn model_permissions_prevent_provider_calls(#[case] path: &str) {
    let upstream = MockServer::start().await;
    let app = support::app_with_permissions(
        "anthropic/test-model",
        &upstream.uri(),
        litellm_gateway_auth::Permissions::None,
    );
    let response = support::post(app, path, json!({"model": "public/model"})).await;
    assert_eq!(response.status(), 403);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::completion(false)]
#[case::streaming(true)]
#[tokio::test]
async fn openrouter_receives_native_messages_headers_and_its_deployment_bearer(
    #[case] streaming: bool,
) {
    let upstream = MockServer::start().await;
    let message = json!({"id": "msg_test", "type": "message", "role": "assistant", "model": "anthropic/test-model", "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn", "stop_sequence": null, "usage": {"input_tokens": 1, "output_tokens": 1, "cost": 0.001, "cost_details": "unexpected"}, "openrouter_metadata": {"provider": "test-provider"}});
    let sse = "event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"cost\":0.001}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let template = if streaming {
        ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")
    } else {
        ResponseTemplate::new(200).set_body_json(message.clone())
    };
    Mock::given(method("POST"))
        .and(path("/api/v1/messages"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("anthropic-beta", "test-feature"))
        .and(header("anthropic-version", "test-version"))
        .and(header("x-openrouter-metadata", "enabled"))
        .and(body_json(json!({"model": "anthropic/test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16, "stream": streaming, "provider": {"order": ["test-provider"]}})))
        .respond_with(template).expect(1).mount(&upstream).await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .header("anthropic-beta", "test-feature")
        .header("anthropic-version", "test-version")
        .header("x-openrouter-metadata", "enabled")
        .header("x-api-key", "proxy-key")
        .header("authorization", "Bearer proxy-key")
        .body(Body::from(json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16, "stream": streaming, "provider": {"order": ["test-provider"]}}).to_string())).unwrap();
    let response = support::app(
        "openrouter/anthropic/test-model",
        &format!("{}/api/v1", upstream.uri()),
    )
    .oneshot(request)
    .await
    .unwrap();
    assert_eq!(response.status(), 200);
    if streaming {
        assert_eq!(to_bytes(response.into_body(), 4096).await.unwrap(), sse);
    } else {
        assert_eq!(support::json(response).await, message);
    }
    let requests = upstream.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert!(!requests[0].headers.contains_key("x-api-key"));
}
