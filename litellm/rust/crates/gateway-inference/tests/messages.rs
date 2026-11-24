mod support;

use std::time::{Duration, Instant};

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
#[case::thread("thread", json!({"type": "create"}))]
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
#[case::deployment_tools_that_are_not_a_list_stay_out(
    json!({"tools": tool("deployment_lookup")}),
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

#[rstest]
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

#[rstest]
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

fn requesting_api_version_2026_10_21(temperature: Value) -> Request<Body> {
    Request::post("/v1/messages")
        .header("content-type", "application/json")
        .header("anthropic-version", "2026-10-21")
        .body(Body::from(
            messages_request(
                "public/model",
                json!({"max_tokens": 16, "temperature": temperature}),
            )
            .to_string(),
        ))
        .unwrap()
}

#[tokio::test]
async fn sampling_params_under_an_api_version_anthropic_removed_them_from_are_refused_before_calling_anthropic()
 {
    let response = support::app("anthropic/test-model", "https://api.anthropic.com")
        .oneshot(requesting_api_version_2026_10_21(json!(1)))
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body = support::json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(
        body["error"]["message"],
        "invalid request: test-model does not support temperature=1. anthropic-version \
         2026-10-21 removed temperature, top_p and top_k. To drop unsupported params, set \
         `litellm.drop_params = True`, or `drop_params: true` under `litellm_settings` in the \
         proxy config."
    );
}

#[rstest]
#[tokio::test]
async fn sampling_params_reach_a_third_party_anthropic_api_under_an_api_version_anthropic_removed_them_from(
    #[values(false, true)] drop_params: bool,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .and(header("anthropic-version", "2026-10-21"))
        .and(body_json(messages_request(
            "test-model",
            json!({"max_tokens": 16, "temperature": 0.2}),
        )))
        .respond_with(message_response())
        .expect(1)
        .mount(&upstream)
        .await;
    let app = support::app_with_shaping(
        "anthropic/test-model",
        &format!("{}/anthropic", upstream.uri()),
        deployment_params(json!({})),
        json!({"drop_params": drop_params}),
    );

    let response = app
        .oneshot(requesting_api_version_2026_10_21(json!(0.2)))
        .await
        .unwrap();

    assert_eq!(response.status(), 200, "{}", support::json(response).await);
}

/// What the provider is sent for `client` params under the deployment `shaping`, or the status
/// the gateway answers with when it sends nothing.
async fn sent_under(shaping: Value, client: Value) -> Result<Value, u16> {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(message_response())
        .mount(&upstream)
        .await;
    let app = support::app_with_shaping(
        "anthropic/test-model",
        &upstream.uri(),
        deployment_params(json!({})),
        shaping,
    );

    let response = support::post(
        app,
        "/v1/messages",
        messages_request("public/model", client),
    )
    .await;

    let sent = upstream.received_requests().await.unwrap();
    match sent.first() {
        Some(sent) if response.status() == 200 => Ok(sent.body_json().unwrap()),
        _ => Err(response.status().as_u16()),
    }
}

/// A request's `drop_params` decides for its call over the deployment's, and
/// `litellm_settings.drop_params` keeps dropping on whatever the request says. The key never
/// reaches the provider.
#[rstest]
#[case::the_request_turning_it_on(json!({}), json!(true), Ok(()))]
#[case::a_flag_word(json!({}), json!("yes"), Ok(()))]
#[case::the_request_turning_the_deployments_off(
    json!({"drop_params": true, "drops_unsupported_params": true}),
    json!(false),
    Err(400)
)]
#[case::the_config_over_the_request(
    json!({"drop_params": true, "drops_unsupported_params": true, "global_drop_params": true}),
    json!(false),
    Ok(())
)]
#[tokio::test]
async fn a_requests_drop_params_decides_whether_its_unsupported_params_are_dropped(
    #[case] shaping: Value,
    #[case] requested: Value,
    #[case] expected: Result<(), u16>,
) {
    let sent = sent_under(
        shaping,
        json!({"max_tokens": 16, "speed": "fast", "drop_params": requested}),
    )
    .await;

    assert_eq!(
        sent,
        expected.map(|()| messages_request("test-model", json!({"max_tokens": 16})))
    );
}

#[rstest]
#[case::the_deployments_when_the_request_names_none(
    json!({}),
    json!({"max_tokens": 16, "top_k": 5})
)]
#[case::the_requests_in_their_place(
    json!({"additional_drop_params": ["top_k"]}),
    json!({"max_tokens": 16, "temperature": 0.5})
)]
#[tokio::test]
async fn a_requests_additional_drop_params_replace_the_deployments(
    #[case] requested: Value,
    #[case] expected: Value,
) {
    let client = json!({"max_tokens": 16, "temperature": 0.5, "top_k": 5})
        .as_object()
        .unwrap()
        .clone()
        .into_iter()
        .chain(requested.as_object().unwrap().clone())
        .collect();

    let sent = sent_under(
        json!({"additional_drop_params": ["temperature"]}),
        Value::Object(client),
    )
    .await;

    assert_eq!(sent, Ok(messages_request("test-model", expected)));
}

#[rstest]
#[tokio::test]
async fn reasoning_auto_summary_asks_for_summarized_thinking(#[values(false, true)] on: bool) {
    let thinking = json!({"type": "enabled", "budget_tokens": 1024});

    let sent = sent_under(
        json!({"reasoning_auto_summary": on, "capabilities": {"supports_reasoning": true}}),
        json!({"max_tokens": 2048, "thinking": thinking}),
    )
    .await
    .unwrap();

    assert_eq!(
        sent["thinking"].get("display"),
        on.then(|| json!("summarized")).as_ref()
    );
}

/// A call waits for the deployment's `timeout`, else its `request_timeout`, else
/// `litellm_settings.request_timeout`, the order of Python's `resolve_llm_passthrough_timeout`.
#[rstest]
#[case::the_deployments_timeout_first(
    json!({"timeout": 5, "request_timeout": 0.1, "num_retries": 0}),
    json!({"request_timeout": 0.1}),
    true
)]
#[case::the_deployments_request_timeout(
    json!({"timeout": null, "request_timeout": 0.1, "num_retries": 0}),
    json!({"request_timeout": 5}),
    false
)]
#[case::the_deployments_request_timeout_before_the_configs(
    json!({"timeout": null, "request_timeout": 5, "num_retries": 0}),
    json!({"request_timeout": 0.1}),
    true
)]
#[case::the_configs_request_timeout_last(
    json!({"timeout": null, "num_retries": 0}),
    json!({"request_timeout": 0.1}),
    false
)]
#[tokio::test]
async fn a_call_waits_for_the_first_timeout_python_reads(
    #[case] litellm_params: Value,
    #[case] shaping: Value,
    #[case] answered: bool,
) {
    let delay = Duration::from_secs(1);
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(message_response().set_delay(delay))
        .mount(&upstream)
        .await;
    let app = support::app_with_shaping(
        "anthropic/test-model",
        &upstream.uri(),
        deployment_params(litellm_params),
        shaping,
    );
    let started = Instant::now();

    let response = support::post(
        app,
        "/v1/messages",
        messages_request("public/model", json!({"max_tokens": 16})),
    )
    .await;

    assert_eq!(
        (response.status() == 200, started.elapsed() >= delay),
        (answered, answered)
    );
}

const FIRST_EVENT: &str = "event: message_start\ndata: {}\n\n";

/// A stream of `public/model`, whose deployment retries `num_retries` times and whose provider
/// sends one event and then drops the connection, never to answer again.
async fn broken_stream(num_retries: u32) -> axum::response::Response {
    let base = support::truncating_upstream(FIRST_EVENT).await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16, "stream": true})
            .to_string(),
        ))
        .unwrap();
    support::app_with_litellm_params(
        "anthropic/test-model",
        &base,
        json!({"num_retries": num_retries}),
    )
    .oneshot(request)
    .await
    .unwrap()
}

#[tokio::test]
async fn a_stream_that_breaks_with_no_attempt_left_to_answer_ends_with_an_sse_error_frame() {
    let response = broken_stream(0).await;

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

#[tokio::test]
async fn a_stream_that_breaks_before_content_and_is_not_answered_for_ends_on_an_error_answer() {
    let response = broken_stream(1).await;

    let routed: Vec<(&str, &str)> = [
        "x-litellm-model-group",
        "x-litellm-attempted-retries",
        "x-litellm-attempted-fallbacks",
    ]
    .into_iter()
    .filter_map(|name| Some((name, response.headers().get(name)?.to_str().ok()?)))
    .collect();
    assert_eq!(
        (response.status().as_u16(), routed),
        (
            500,
            vec![
                ("x-litellm-model-group", "public/model"),
                ("x-litellm-attempted-retries", "0"),
                ("x-litellm-attempted-fallbacks", "0"),
            ]
        )
    );
    let error = support::json(response).await;
    assert_eq!(
        (&error["type"], &error["error"]["type"]),
        (&json!("error"), &json!("api_error"))
    );
}

/// The Python proxy answers a provider's 413 as a 400 `invalid_request_error`, and a 408 with
/// the type `api_error`. The gateway answers what the provider said.
#[rstest]
#[case::too_large(413, "request_too_large")]
#[case::timed_out(408, "timeout_error")]
#[tokio::test]
async fn a_providers_refusal_is_answered_with_its_own_status_and_error_type(
    #[case] status: u16,
    #[case] error_type: &str,
) {
    let upstream = MockServer::start().await;
    let error = json!({"type": "error", "error": {"type": error_type, "message": "refused"}});
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(status).set_body_json(error.clone()))
        .mount(&upstream)
        .await;
    let request = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(json!({
            "model": "public/model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16,
        }).to_string()))
        .unwrap();

    let response = support::app_with_litellm_params(
        "anthropic/test-model",
        &upstream.uri(),
        json!({"num_retries": 0}),
    )
    .oneshot(request)
    .await
    .unwrap();

    assert_eq!(
        (response.status().as_u16(), support::json(response).await),
        (status, error)
    );
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
        .expect(3)
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
        let served = message
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, value)| match name.as_str() {
                "model" => (name.clone(), json!("public/model")),
                _ => (name.clone(), value.clone()),
            })
            .collect();
        assert_eq!(support::json(response).await, Value::Object(served));
    }
    let requests = upstream.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert!(!requests[0].headers.contains_key("x-api-key"));
}

#[rstest]
#[tokio::test]
async fn the_body_is_read_as_json_whatever_the_content_type(
    #[values(None, Some("application/json"), Some("text/plain"))] content_type: Option<&str>,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(message_response())
        .expect(1)
        .mount(&upstream)
        .await;
    let request = Request::post("/v1/messages");
    let request = match content_type {
        Some(content_type) => request.header("content-type", content_type),
        None => request,
    };
    let response = support::app("anthropic/test-model", &upstream.uri())
        .oneshot(
            request
                .body(Body::from(
                    messages_request("public/model", json!({"max_tokens": 16})).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}
