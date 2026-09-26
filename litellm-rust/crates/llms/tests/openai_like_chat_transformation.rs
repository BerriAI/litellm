use litellm_llms::{
    Error,
    base_llm::{
        auth::AuthScheme,
        chat::transformation::{BaseConfig, ProviderChatResponseData, Unsupported},
    },
    openai_like::chat::transformation::OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG,
};
use litellm_types::{llms::openai::ChatMessage, utils::ChatCompletionsResponse};
use rstest::rstest;
use serde_json::{Map, Value, json};

fn messages(value: Value) -> Vec<ChatMessage> {
    serde_json::from_value(value).expect("valid messages")
}

fn params(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => panic!("params must be an object, got {other}"),
    }
}

fn no_env(_: &str) -> Option<String> {
    None
}

fn env_with<'a>(name: &'a str, value: &'a str) -> impl Fn(&str) -> Option<String> + 'a {
    move |key| (key == name).then(|| value.to_string())
}

fn transform(model: &str, msgs: Value, opts: Value) -> Value {
    OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
        .transform_request(model, messages(msgs), params(opts))
        .expect("request transforms")
        .body
}

fn transform_response(body: Value) -> Result<ChatCompletionsResponse, Error> {
    OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
        .transform_response("some-model", ProviderChatResponseData { body })
}

fn reason(msgs: Value, opts: Value) -> Option<Unsupported> {
    OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG.unsupported_reason(&messages(msgs), &params(opts))
}

#[test]
fn builds_the_openai_shaped_body() {
    let body = transform(
        "my-model",
        json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]),
        json!({"temperature": 0.5, "max_tokens": 8}),
    );
    assert_eq!(body["model"], json!("my-model"));
    assert_eq!(
        body["messages"],
        json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ])
    );
    assert_eq!(body["temperature"], json!(0.5));
    assert_eq!(body["max_tokens"], json!(8));
}

#[test]
fn renames_max_completion_tokens_to_max_tokens() {
    // `OpenAILikeChatConfig.map_openai_params`: most OpenAI-compatible providers
    // support `max_tokens`, not `max_completion_tokens`.
    let body = transform(
        "my-model",
        json!([{"role": "user", "content": "hi"}]),
        json!({"max_completion_tokens": 12}),
    );
    assert_eq!(body["max_tokens"], json!(12));
    assert!(body.get("max_completion_tokens").is_none());
}

#[test]
fn call_configuration_never_enters_the_body() {
    let body = transform(
        "my-model",
        json!([{"role": "user", "content": "hi"}]),
        json!({"custom_endpoint": true, "extra_headers": {"x": "y"}, "max_retries": 2}),
    );
    assert_eq!(
        body.as_object().unwrap().keys().collect::<Vec<_>>(),
        vec!["model", "messages"]
    );
}

#[rstest]
#[case::appends_the_chat_completions_suffix("https://vllm.example.com/v1", json!({}), "https://vllm.example.com/v1/chat/completions")]
#[case::trims_a_trailing_slash("https://vllm.example.com/v1/", json!({}), "https://vllm.example.com/v1/chat/completions")]
#[case::a_custom_endpoint_is_used_as_is("https://vllm.example.com/v1/chat/completions", json!({"custom_endpoint": true}), "https://vllm.example.com/v1/chat/completions")]
fn complete_url(#[case] api_base: &str, #[case] opts: Value, #[case] expected: &str) {
    assert_eq!(
        OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
            .get_complete_url(Some(api_base), "my-model", &params(opts), &no_env)
            .expect("url resolves"),
        expected
    );
}

#[test]
fn api_base_falls_back_to_the_environment() {
    assert_eq!(
        OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
            .get_complete_url(
                None,
                "my-model",
                &params(json!({})),
                &env_with("OPENAI_LIKE_API_BASE", "https://env.example.com/v1"),
            )
            .expect("url resolves"),
        "https://env.example.com/v1/chat/completions"
    );
}

#[test]
fn a_missing_api_base_is_an_error() {
    assert!(matches!(
        OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
            .get_complete_url(None, "my-model", &params(json!({})), &no_env),
        Err(Error::InvalidRequest(message)) if message.starts_with("Missing API Base")
    ));
}

#[test]
fn the_resolved_key_authenticates_as_a_bearer() {
    let validated = OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
        .validate_environment(
            vec![],
            Some("sk-test"),
            "my-model",
            &params(json!({})),
            &no_env,
        )
        .expect("validates");
    assert!(matches!(
        validated.auth,
        AuthScheme::Credential {
            placement: litellm_auth::CredentialPlacement::Bearer,
            ..
        }
    ));
}

#[test]
fn the_key_falls_back_to_the_environment() {
    let validated = OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
        .validate_environment(
            vec![],
            None,
            "my-model",
            &params(json!({})),
            &env_with("OPENAI_LIKE_API_KEY", "sk-env"),
        )
        .expect("validates");
    let AuthScheme::Credential { secret, .. } = validated.auth else {
        panic!("expected a bearer credential");
    };
    assert_eq!(secret.expose(), "sk-env");
}

#[test]
fn a_forwarded_authorization_is_the_whole_credential() {
    // Python adds `Bearer <key>` only when the caller did not already send
    // `Authorization`, so the forwarded header wins over the deployment key.
    let headers = vec![("Authorization".to_string(), "Bearer caller".to_string())];
    let validated = OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
        .validate_environment(
            headers,
            Some("sk-test"),
            "my-model",
            &params(json!({})),
            &no_env,
        )
        .expect("validates");
    assert!(matches!(validated.auth, AuthScheme::Forwarded));
}

#[test]
fn keyless_calls_still_validate_for_endpoints_that_take_no_key() {
    // vllm-compatible endpoints require no api key; Python resolves `""` and
    // sends `Bearer `, so validation must not fail on the missing key.
    let validated = OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
        .validate_environment(vec![], None, "my-model", &params(json!({})), &no_env)
        .expect("validates");
    let AuthScheme::Credential { secret, .. } = validated.auth else {
        panic!("expected a bearer credential");
    };
    assert_eq!(secret.expose(), "");
}

#[test]
fn normalizes_an_openai_response() {
    let response = transform_response(json!({
        "created": 1_700_000_000,
        "model": "served-model-name",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }))
    .expect("response normalizes");
    assert_eq!(response.created, 1_700_000_000);
    assert_eq!(response.model, "served-model-name");
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello")
    );
    assert_eq!(response.choices[0].finish_reason, "stop");
    assert_eq!(response.usage.prompt_tokens, 3);
    assert_eq!(response.usage.completion_tokens, 5);
    assert_eq!(response.usage.total_tokens, 8);
}

#[test]
fn null_token_fields_in_usage_become_zero() {
    // `_sanitize_usage_obj`: providers that return null token values break
    // OpenAI clients, so the response is scrubbed at the source.
    let response = transform_response(json!({
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": null, "total_tokens": null},
    }))
    .expect("response normalizes");
    assert_eq!(response.usage.completion_tokens, 0);
    assert_eq!(response.usage.total_tokens, 0);
    assert_eq!(response.usage.prompt_tokens, 3);
}

#[test]
fn a_tool_call_response_declines_instead_of_dropping_the_calls() {
    // The `json_mode` rewrite needs a request flag the route does not carry, so
    // a tool-call answer falls back to Python rather than losing the calls.
    assert_eq!(
        transform_response(json!({
            "model": "m",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })),
        Err(Error::Unsupported("tool call response"))
    );
}

#[test]
fn a_non_text_response_content_declines() {
    assert_eq!(
        transform_response(json!({
            "model": "m",
            "choices": [{
                "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                "finish_reason": "stop",
            }],
        })),
        Err(Error::Unsupported("non-text response content"))
    );
}

#[rstest]
#[case::streaming(json!({"stream": true}), "streaming")]
#[case::unrecognized_param(json!({"some_provider_knob": 1}), "unrecognized request parameter")]
fn declines(#[case] opts: Value, #[case] expected: &'static str) {
    assert_eq!(
        reason(json!([{"role": "user", "content": "hi"}]), opts),
        Some(Unsupported(expected))
    );
}

#[test]
fn accepts_standard_openai_params() {
    assert_eq!(
        reason(
            json!([{"role": "user", "content": "hi"}]),
            json!({
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 16,
                "response_format": {"type": "json_object"},
                "custom_endpoint": true,
            }),
        ),
        None
    );
}

#[test]
fn tool_parameters_decline_before_the_call() {
    // A `tools` request would come back with tool calls this port cannot
    // normalize, so it declines at the gate instead of after the call.
    assert_eq!(
        reason(
            json!([{"role": "user", "content": "hi"}]),
            json!({"tools": [{"type": "function", "function": {"name": "f"}}]}),
        ),
        Some(Unsupported("unrecognized request parameter"))
    );
}
