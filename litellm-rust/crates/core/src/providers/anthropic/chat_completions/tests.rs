use super::*;
use serde_json::json;

fn messages(value: Value) -> Vec<ChatMessage> {
    serde_json::from_value(value).expect("valid messages")
}

fn params(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => panic!("params must be an object, got {other}"),
    }
}

fn transform(model: &str, msgs: Value, opts: Value) -> Value {
    ANTHROPIC_CHAT_COMPLETIONS_CONFIG
        .transform_request(model, messages(msgs), params(opts))
        .expect("request transforms")
        .body
}

fn transform_response(body: Value) -> CoreResult<ChatCompletionsResponse> {
    ANTHROPIC_CHAT_COMPLETIONS_CONFIG
        .transform_response("claude-sonnet-4-5", ProviderChatResponseData { body })
}

fn reason(msgs: Value, opts: Value) -> Option<Unsupported> {
    ANTHROPIC_CHAT_COMPLETIONS_CONFIG.unsupported_reason(&messages(msgs), &params(opts))
}

#[test]
fn builds_the_messages_body_python_builds() {
    let body = transform(
        "claude-sonnet-4-5",
        json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"}
        ]),
        json!({"max_tokens": 128, "temperature": 0.2}),
    );
    assert_eq!(
        body,
        json!({
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]}
            ],
            "system": [{"type": "text", "text": "be terse"}],
            "max_tokens": 128,
            "temperature": 0.2
        })
    );
}

#[test]
fn omits_system_when_no_system_message_is_present() {
    let body = transform(
        "claude-sonnet-4-5",
        json!([{"role": "user", "content": "hi"}]),
        json!({"max_tokens": 16}),
    );
    assert!(body.get("system").is_none());
}

#[test]
fn merges_consecutive_turns_and_wraps_every_text_in_a_block() {
    let body = transform(
        "claude-sonnet-4-5",
        json!([
            {"role": "user", "content": "one"},
            {"role": "user", "content": [{"type": "text", "text": "two"}]},
            {"role": "assistant", "content": "ack"}
        ]),
        json!({"max_tokens": 16}),
    );
    assert_eq!(
        body["messages"],
        json!([
            {"role": "user", "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "ack"}]}
        ])
    );
}

#[test]
fn right_strips_a_trailing_assistant_prefill_like_python() {
    let body = transform(
        "claude-sonnet-4-5",
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Argentina  "}
        ]),
        json!({"max_tokens": 16}),
    );
    assert_eq!(
        body["messages"][1]["content"][0]["text"],
        json!("Argentina")
    );
}

#[test]
fn passes_every_supported_param_through_untouched() {
    let body = transform(
        "claude-sonnet-4-5",
        json!([{"role": "user", "content": "hi"}]),
        json!({
            "max_tokens": 64,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop_sequences": ["STOP"]
        }),
    );
    assert_eq!(body["max_tokens"], json!(64));
    assert_eq!(body["temperature"], json!(0.1));
    assert_eq!(body["top_p"], json!(0.9));
    assert_eq!(body["stop_sequences"], json!(["STOP"]));
}

#[test]
fn declines_top_k_because_python_gates_it_by_model_below_this_point() {
    // `temperature` and `top_p` arrive already resolved, because
    // `map_openai_params` applies `_apply_sampling_param` to them before the
    // gate runs. `top_k` bypasses that and is gated inside `transform_request`,
    // the function this route replaces, so forwarding it would send `top_k` to
    // a model that removed sampling params and take a 400 after the call, where
    // Python drops it and succeeds.
    assert_eq!(
        reason(
            json!([{"role": "user", "content": "hi"}]),
            json!({"top_k": 40})
        ),
        Some(Unsupported("unrecognized request parameter"))
    );
}

#[test]
fn declines_streaming_before_anything_else() {
    assert_eq!(
        reason(
            json!([{"role": "user", "content": "hi"}]),
            json!({"stream": true, "max_tokens": 16})
        ),
        Some(Unsupported("streaming"))
    );
}

#[test]
fn accepts_an_explicit_stream_false() {
    assert_eq!(
        reason(
            json!([{"role": "user", "content": "hi"}]),
            json!({"stream": false, "max_tokens": 16})
        ),
        None
    );
}

#[test]
fn declines_any_param_outside_the_allowlist() {
    for param in [
        json!({"tools": []}),
        json!({"tool_choice": {"type": "auto"}}),
        json!({"thinking": {"type": "enabled"}}),
        json!({"system": "injected"}),
        json!({"metadata": {"user_id": "u1"}}),
        json!({"output_config": {"effort": "high"}}),
    ] {
        assert_eq!(
            reason(json!([{"role": "user", "content": "hi"}]), param.clone()),
            Some(Unsupported("unrecognized request parameter")),
            "expected {param} to decline"
        );
    }
}

#[test]
fn declines_tool_calls_tool_results_and_multimodal_content() {
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
    assert_eq!(
        reason(
            json!([{"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
            ]}]),
            json!({})
        ),
        Some(Unsupported("non-text message content"))
    );
}

#[test]
fn declines_a_message_whose_content_list_is_empty() {
    // An empty list passes every per-part check, so without this it would reach
    // the provider as an empty `content` array and fail after the call rather
    // than declining to Python before it.
    assert_eq!(
        reason(json!([{"role": "user", "content": []}]), json!({})),
        Some(Unsupported("message without content"))
    );
    assert_eq!(
        reason(
            json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
            json!({})
        ),
        None
    );
}

#[test]
fn declines_a_conversation_that_does_not_open_on_a_user_turn() {
    assert_eq!(
        reason(
            json!([
                {"role": "system", "content": "be terse"},
                {"role": "assistant", "content": "prefill"}
            ]),
            json!({})
        ),
        Some(Unsupported("conversation does not open on a user turn"))
    );
}

#[test]
fn accepts_a_plain_text_conversation() {
    assert_eq!(
        reason(
            json!([
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": [{"type": "text", "text": "again"}]}
            ]),
            json!({"max_tokens": 16, "temperature": 0.5})
        ),
        None
    );
}

#[test]
fn normalizes_a_text_response_into_openai_shape() {
    let response = transform_response(json!({
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20260101",
        "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " there"}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 11, "output_tokens": 4}
    }))
    .expect("response transforms");

    assert_eq!(response.model, "claude-sonnet-4-5-20260101");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].index, 0);
    assert_eq!(response.choices[0].message.role, "assistant");
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("hello there")
    );
    assert_eq!(response.choices[0].finish_reason, "stop");
    assert_eq!(response.usage.prompt_tokens, 11);
    assert_eq!(response.usage.completion_tokens, 4);
    assert_eq!(response.usage.total_tokens, 15);
}

#[test]
fn folds_cache_tokens_into_prompt_tokens_like_python() {
    let response = transform_response(json!({
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_read_input_tokens": 5,
            "cache_creation_input_tokens": 3
        }
    }))
    .expect("response transforms");
    assert_eq!(response.usage.prompt_tokens, 18);
    assert_eq!(response.usage.total_tokens, 20);
    assert_eq!(response.usage.prompt_tokens_details.cached_tokens, 5);
    assert_eq!(
        response.usage.prompt_tokens_details.cache_creation_tokens,
        3
    );
    assert_eq!(response.usage.prompt_tokens_details.text_tokens, 10);
}

#[test]
fn maps_max_tokens_stop_reason_to_length() {
    let response = transform_response(json!({
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
    .expect("response transforms");
    assert_eq!(response.choices[0].finish_reason, "length");
}

#[test]
fn a_refusal_returns_the_completion_python_returns() {
    // `refusal` is a stop_reason, not a content block type, so the content is
    // ordinary text and this normalizes rather than declining. Python maps it
    // to content_filter in _FINISH_REASON_MAP and returns the completion.
    let response = transform_response(json!({
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "I can't help with that."}],
        "stop_reason": "refusal",
        "usage": {"input_tokens": 9, "output_tokens": 6}
    }))
    .expect("a refusal still transforms");
    assert_eq!(response.choices[0].finish_reason, "content_filter");
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("I can't help with that.")
    );
}

#[test]
fn reports_no_content_rather_than_an_empty_string() {
    let response = transform_response(json!({
        "model": "claude-sonnet-4-5",
        "content": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 0}
    }))
    .expect("response transforms");
    assert_eq!(response.choices[0].message.content, None);
}

#[test]
fn response_carries_no_id_so_python_keeps_its_chatcmpl_id() {
    let response = transform_response(json!({
        "id": "msg_should_not_leak",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
    .expect("response transforms");
    let value = serde_json::to_value(response).expect("serializable");
    assert!(
        value.get("id").is_none(),
        "the rust response must not carry an id, got {value}"
    );
}

#[test]
fn declines_a_response_carrying_a_non_text_block() {
    let err = transform_response(json!({
        "model": "claude-sonnet-4-5",
        "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
    .expect_err("non-text block");
    assert_eq!(
        err,
        CoreError::Unsupported("non-text response content block")
    );
}

#[test]
fn errors_on_a_response_missing_required_fields() {
    assert_eq!(
        transform_response(json!("nope")).expect_err("not an object"),
        CoreError::InvalidResponse("messages response is not an object".to_string())
    );
    assert_eq!(
        transform_response(json!({"model": "m", "usage": {}})).expect_err("no content"),
        CoreError::MissingField("content")
    );
    assert_eq!(
        transform_response(json!({"model": "m", "content": []})).expect_err("no usage"),
        CoreError::MissingField("usage")
    );
    assert_eq!(
        transform_response(json!({"content": [], "usage": {}})).expect_err("no model"),
        CoreError::MissingField("model")
    );
}

#[test]
fn resolves_the_messages_url_and_x_api_key_auth() {
    let config = &ANTHROPIC_CHAT_COMPLETIONS_CONFIG;
    assert_eq!(
        config
            .complete_url(None, "claude-sonnet-4-5", &Map::new(), &|_| None)
            .expect("url builds"),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(
        config
            .auth(Some("sk-x"), "claude-sonnet-4-5", &Map::new(), &|_| None)
            .expect("auth resolves"),
        ChatCompletionsAuth::Header {
            name: "x-api-key",
            value: "sk-x".to_string()
        }
    );
    assert_eq!(
        config.default_headers(),
        &[
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
        ]
    );
}
