use super::*;
use crate::Error;
use crate::chat_completions::transformation::Unsupported;

fn messages(value: serde_json::Value) -> Vec<ChatMessage> {
    serde_json::from_value(value).expect("valid messages")
}

fn params(value: serde_json::Value) -> Map<String, serde_json::Value> {
    match value {
        serde_json::Value::Object(map) => map,
        other => panic!("params must be an object, got {other}"),
    }
}

fn transform(model: &str, msgs: serde_json::Value, opts: serde_json::Value) -> serde_json::Value {
    OLLAMA_CHAT_COMPLETIONS_CONFIG
        .transform_request(model, messages(msgs), params(opts))
        .expect("request transforms")
        .body
}

fn transform_response(
    model: &str,
    body: serde_json::Value,
) -> Result<ChatCompletionsResponse, Error> {
    OLLAMA_CHAT_COMPLETIONS_CONFIG.transform_response(model, ProviderChatResponseData { body })
}

fn reason(msgs: serde_json::Value, opts: serde_json::Value) -> Option<Unsupported> {
    OLLAMA_CHAT_COMPLETIONS_CONFIG.unsupported_reason(&messages(msgs), &params(opts))
}

#[test]
fn complete_url_appends_api_chat_and_honors_a_trailing_slash() {
    let config = &OLLAMA_CHAT_COMPLETIONS_CONFIG;
    assert_eq!(
        config
            .complete_url(None, "llama3.2", &Map::new(), &|_| None)
            .expect("url builds"),
        "http://localhost:11434/api/chat"
    );
    assert_eq!(
        config
            .complete_url(Some("http://host:11434/"), "llama3.2", &Map::new(), &|_| {
                None
            })
            .expect("url builds"),
        "http://host:11434/api/chat"
    );
}

#[test]
fn complete_url_passes_through_a_caller_supplied_full_endpoint() {
    let config = &OLLAMA_CHAT_COMPLETIONS_CONFIG;
    assert_eq!(
        config
            .complete_url(
                Some("http://host:11434/api/chat"),
                "llama3.2",
                &Map::new(),
                &|_| None
            )
            .expect("url builds"),
        "http://host:11434/api/chat"
    );
}

#[test]
fn complete_url_prefers_ollama_api_base_env_over_the_default() {
    let with_env =
        |key: &str| (key == OLLAMA_API_BASE_ENV).then(|| "http://env-host:11434".to_string());
    assert_eq!(
        OLLAMA_CHAT_COMPLETIONS_CONFIG
            .complete_url(Some("  "), "llama3.2", &Map::new(), &with_env)
            .expect("url builds"),
        "http://env-host:11434/api/chat"
    );
}

#[test]
fn auth_returns_none_when_no_key_or_env_key_is_present() {
    assert_eq!(
        OLLAMA_CHAT_COMPLETIONS_CONFIG
            .auth(None, "llama3.2", &Map::new(), &|_| None)
            .expect("auth resolves"),
        ChatCompletionsAuth::None
    );
    // Empty or whitespace-only credentials are treated as absent.
    assert_eq!(
        OLLAMA_CHAT_COMPLETIONS_CONFIG
            .auth(Some("   "), "llama3.2", &Map::new(), &|_| None)
            .expect("auth resolves"),
        ChatCompletionsAuth::None
    );
}

#[test]
fn auth_returns_bearer_for_a_param_key_and_for_an_env_key() {
    assert_eq!(
        OLLAMA_CHAT_COMPLETIONS_CONFIG
            .auth(Some("test-key"), "llama3.2", &Map::new(), &|_| None)
            .expect("auth resolves"),
        ChatCompletionsAuth::Bearer {
            token: "test-key".to_string()
        }
    );
    let with_env = |key: &str| (key == OLLAMA_API_KEY_ENV).then(|| "env-key".to_string());
    assert_eq!(
        OLLAMA_CHAT_COMPLETIONS_CONFIG
            .auth(None, "llama3.2", &Map::new(), &with_env)
            .expect("auth resolves"),
        ChatCompletionsAuth::Bearer {
            token: "env-key".to_string()
        }
    );
    // A param key outranks the env value.
    let with_env = |key: &str| (key == OLLAMA_API_KEY_ENV).then(|| "env-key".to_string());
    assert_eq!(
        OLLAMA_CHAT_COMPLETIONS_CONFIG
            .auth(Some("param-key"), "llama3.2", &Map::new(), &with_env)
            .expect("auth resolves"),
        ChatCompletionsAuth::Bearer {
            token: "param-key".to_string()
        }
    );
}

#[test]
fn transform_request_builds_the_ollama_chat_body_from_string_content() {
    let body = transform(
        "llama3.2",
        json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"}
        ]),
        json!({"temperature": 0.5, "num_predict": 32}),
    );
    assert_eq!(
        body,
        json!({
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": "be terse", "images": []},
                {"role": "user", "content": "hi", "images": []}
            ],
            "options": {"temperature": 0.5, "num_predict": 32},
            "stream": false
        })
    );
}

#[test]
fn transform_request_flattens_text_parts_into_a_single_content_string() {
    let body = transform(
        "llama3.2",
        json!([{"role": "user", "content": [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two"}
        ]}]),
        json!({}),
    );
    assert_eq!(body["messages"][0]["content"], json!("onetwo"));
}

#[test]
fn transform_request_pulls_stream_out_of_options_and_hardcodes_it_false() {
    let body = transform(
        "llama3.2",
        json!([{"role": "user", "content": "hi"}]),
        json!({"stream": false, "temperature": 0.1}),
    );
    assert_eq!(body["stream"], json!(false));
    assert!(
        body["options"]
            .as_object()
            .is_some_and(|options| !options.contains_key("stream")),
        "stream must not also sit in options, got {body}"
    );
    assert_eq!(body["options"]["temperature"], json!(0.1));
}

#[test]
fn transform_response_normalizes_a_representative_ollama_reply() {
    let response = transform_response(
        "llama3.2",
        json!({
            "model": "llama3.2:latest",
            "message": {"role": "assistant", "content": "Paris"},
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 8,
            "eval_count": 1
        }),
    )
    .expect("response transforms");

    assert_eq!(response.model, "ollama_chat/llama3.2");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].index, 0);
    assert_eq!(response.choices[0].message.role, "assistant");
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("Paris")
    );
    assert_eq!(response.choices[0].finish_reason, "stop");
    assert_eq!(response.usage.prompt_tokens, 8);
    assert_eq!(response.usage.completion_tokens, 1);
    assert_eq!(response.usage.total_tokens, 9);
}

#[test]
fn transform_response_defaults_missing_usage_and_done_reason() {
    let response = transform_response(
        "llama3.2",
        json!({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "hi"},
            "done": true
        }),
    )
    .expect("response transforms");
    assert_eq!(response.choices[0].finish_reason, "stop");
    assert_eq!(response.usage.prompt_tokens, 0);
    assert_eq!(response.usage.completion_tokens, 0);
    assert_eq!(response.usage.total_tokens, 0);
}

#[test]
fn transform_response_maps_a_length_done_reason() {
    let response = transform_response(
        "llama3.2",
        json!({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "truncated"},
            "done": true,
            "done_reason": "length",
            "prompt_eval_count": 3,
            "eval_count": 2
        }),
    )
    .expect("response transforms");
    assert_eq!(response.choices[0].finish_reason, "length");
}

#[test]
fn transform_response_reports_no_content_when_the_message_content_is_absent() {
    let response = transform_response(
        "llama3.2",
        json!({
            "model": "llama3.2",
            "message": {"role": "assistant"},
            "done": true,
            "done_reason": "stop"
        }),
    )
    .expect("response transforms");
    assert_eq!(response.choices[0].message.content, None);
}

#[test]
fn transform_response_errors_on_a_non_object_or_messageless_body() {
    assert_eq!(
        transform_response("llama3.2", json!("nope")).expect_err("not an object"),
        Error::InvalidResponse("ollama chat response is not an object".to_string())
    );
    assert_eq!(
        transform_response("llama3.2", json!({"done": true})).expect_err("no message"),
        Error::MissingField("message")
    );
}

#[test]
fn unsupported_reason_declines_out_of_scope_params_and_message_shapes() {
    for opts in [
        json!({"tools": []}),
        json!({"think": true}),
        json!({"format": "json"}),
        json!({"keep_alive": "5m"}),
    ] {
        assert_eq!(
            reason(json!([{"role": "user", "content": "hi"}]), opts.clone()),
            Some(Unsupported("unrecognized request parameter")),
            "expected {opts} to decline"
        );
    }
    // `stream: true` is reported as streaming before the param-allowlist check.
    assert_eq!(
        reason(
            json!([{"role": "user", "content": "hi"}]),
            json!({"stream": true})
        ),
        Some(Unsupported("streaming"))
    );

    assert_eq!(
        reason(
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}}
                ]}
            ]),
            json!({})
        ),
        Some(Unsupported("unrecognized message field"))
    );
    assert_eq!(
        reason(
            json!([
                {"role": "user", "content": "hi"},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}
            ]),
            json!({})
        ),
        Some(Unsupported("unrecognized message field"))
    );
    assert_eq!(
        reason(
            json!([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
            ]}]),
            json!({})
        ),
        Some(Unsupported("non-text message content"))
    );
}

#[test]
fn unsupported_reason_accepts_supported_sampling_params_and_text_messages() {
    assert_eq!(
        reason(
            json!([
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": [{"type": "text", "text": "again"}]}
            ]),
            json!({
                "num_predict": 32,
                "temperature": 0.7,
                "top_p": 0.9,
                "seed": 42,
                "repeat_penalty": 1.1,
                "stop": ["STOP"]
            })
        ),
        None
    );
    // `stream: false` is allowed through (streaming is the only declined shape).
    assert_eq!(
        reason(
            json!([{"role": "user", "content": "hi"}]),
            json!({"stream": false, "temperature": 0.5})
        ),
        None
    );
}

#[test]
fn response_serializes_with_the_ollama_chat_prefix_a_unix_created_and_no_id() {
    let response = transform_response(
        "llama3.2",
        json!({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "hi"},
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1
        }),
    )
    .expect("response transforms");
    let value = serde_json::to_value(&response).expect("serializable");
    assert_eq!(value["model"], json!("ollama_chat/llama3.2"));
    assert!(
        value.get("id").is_none(),
        "the rust response must not carry an id, got {value}"
    );
    assert!(value["created"].as_u64().is_some(), "created is a unix int");
}
