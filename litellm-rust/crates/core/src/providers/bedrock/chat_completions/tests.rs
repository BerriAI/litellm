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

fn transform(msgs: Value, opts: Value) -> Value {
    BEDROCK_CHAT_COMPLETIONS_CONFIG
        .transform_request(
            "anthropic.claude-sonnet-4-5-v1:0",
            messages(msgs),
            params(opts),
        )
        .expect("request transforms")
        .body
}

fn transform_response(body: Value) -> CoreResult<ChatCompletionsResponse> {
    BEDROCK_CHAT_COMPLETIONS_CONFIG.transform_response(
        "anthropic.claude-sonnet-4-5-v1:0",
        ProviderChatResponseData { body },
    )
}

fn reason(msgs: Value, opts: Value) -> Option<Unsupported> {
    BEDROCK_CHAT_COMPLETIONS_CONFIG.unsupported_reason(&messages(msgs), &params(opts))
}

#[test]
fn builds_the_converse_body_python_builds() {
    let body = transform(
        json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"}
        ]),
        json!({"maxTokens": 128, "temperature": 0.2}),
    );
    assert_eq!(
        body,
        json!({
            "inferenceConfig": {"maxTokens": 128, "temperature": 0.2},
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "system": [{"text": "be terse"}]
        })
    );
}

#[test]
fn always_emits_inference_config_even_when_empty() {
    let body = transform(json!([{"role": "user", "content": "hi"}]), json!({}));
    assert_eq!(body["inferenceConfig"], json!({}));
    assert!(body.get("system").is_none());
}

#[test]
fn places_only_inference_params_in_inference_config() {
    let body = transform(
        json!([{"role": "user", "content": "hi"}]),
        json!({
            "maxTokens": 64,
            "temperature": 0.1,
            "topP": 0.9,
            "stopSequences": ["STOP"]
        }),
    );
    assert_eq!(
        body["inferenceConfig"],
        json!({"maxTokens": 64, "temperature": 0.1, "topP": 0.9, "stopSequences": ["STOP"]})
    );
    assert!(body.get("additionalModelRequestFields").is_none());
}

#[test]
fn merges_consecutive_user_turns_into_one_message() {
    let body = transform(
        json!([
            {"role": "user", "content": "one"},
            {"role": "user", "content": [{"type": "text", "text": "two"}]},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "three"}
        ]),
        json!({}),
    );
    assert_eq!(
        body["messages"],
        json!([
            {"role": "user", "content": [{"text": "one"}, {"text": "two"}]},
            {"role": "assistant", "content": [{"text": "ack"}]},
            {"role": "user", "content": [{"text": "three"}]}
        ])
    );
}

#[test]
fn declines_streaming() {
    assert_eq!(
        reason(
            json!([{"role": "user", "content": "hi"}]),
            json!({"stream": true})
        ),
        Some(Unsupported("streaming"))
    );
}

#[test]
fn declines_top_k_because_python_routes_it_by_base_model() {
    assert_eq!(
        reason(
            json!([{"role": "user", "content": "hi"}]),
            json!({"topK": 40})
        ),
        Some(Unsupported("unrecognized request parameter"))
    );
}

#[test]
fn declines_tools_and_other_params_outside_the_allowlist() {
    for param in [
        json!({"tools": []}),
        json!({"tool_choice": {"auto": {}}}),
        json!({"thinking": {"type": "enabled"}}),
        json!({"requestMetadata": {"k": "v"}}),
        json!({"outputConfig": {}}),
        json!({"_parallel_tool_use_config": {}}),
    ] {
        assert_eq!(
            reason(json!([{"role": "user", "content": "hi"}]), param.clone()),
            Some(Unsupported("unrecognized request parameter")),
            "expected {param} to decline"
        );
    }
}

#[test]
fn declines_blank_text_rather_than_substituting_the_anthropic_placeholder() {
    for content in [
        json!(""),
        json!("   "),
        json!([{"type": "text", "text": " "}]),
    ] {
        assert_eq!(
            reason(
                json!([{"role": "user", "content": content}, {"role": "user", "content": "hi"}]),
                json!({})
            ),
            Some(Unsupported("blank message text")),
            "expected blank content {content} to decline"
        );
    }
}

#[test]
fn declines_a_message_whose_content_list_is_empty() {
    // The blank-text check scans parts, so an empty list clears it; Converse
    // rejects an empty `content` array, which is a decline the core owes the
    // host before the call rather than an error after it.
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
fn declines_a_conversation_that_opens_or_closes_on_an_assistant_turn() {
    assert_eq!(
        reason(
            json!([
                {"role": "assistant", "content": "prefill"},
                {"role": "user", "content": "hi"}
            ]),
            json!({})
        ),
        Some(Unsupported(
            "conversation does not run user turn to user turn"
        ))
    );
    assert_eq!(
        reason(
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "prefill"}
            ]),
            json!({})
        ),
        Some(Unsupported(
            "conversation does not run user turn to user turn"
        ))
    );
}

#[test]
fn accepts_a_user_to_user_text_conversation() {
    assert_eq!(
        reason(
            json!([
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"}
            ]),
            json!({"maxTokens": 16})
        ),
        None
    );
}

#[test]
fn builds_the_converse_url_from_the_region_in_the_model_id() {
    let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
    assert_eq!(
        config
            .complete_url(None, "us-east-1/anthropic.claude-v2", &Map::new(), &|_| {
                None
            })
            .expect("url builds"),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/converse"
    );
}

#[test]
fn falls_back_to_the_region_env_then_the_default_region() {
    let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
    let with_env = |key: &str| (key == "AWS_REGION_NAME").then(|| "eu-west-1".to_string());
    assert_eq!(
        config
            .complete_url(None, "anthropic.claude-v2", &Map::new(), &with_env)
            .expect("url builds"),
        "https://bedrock-runtime.eu-west-1.amazonaws.com/model/anthropic.claude-v2/converse"
    );
    assert_eq!(
        config
            .complete_url(None, "anthropic.claude-v2", &Map::new(), &|_| None)
            .expect("url builds"),
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-v2/converse"
    );
}

#[test]
fn prefers_an_explicit_runtime_endpoint_over_the_api_base() {
    let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
    let overrides = params(json!({"aws_bedrock_runtime_endpoint": "https://vpce.internal/"}));
    assert_eq!(
        config
            .complete_url(
                Some("https://ignored.example"),
                "anthropic.claude-v2",
                &overrides,
                &|_| None
            )
            .expect("url builds"),
        "https://vpce.internal/model/anthropic.claude-v2/converse"
    );
}

#[test]
fn signs_with_sigv4_in_the_resolved_region() {
    let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
    assert_eq!(
        config
            .auth(
                None,
                "eu-central-1/anthropic.claude-v2",
                &Map::new(),
                &|_| None
            )
            .expect("auth resolves"),
        ChatCompletionsAuth::AwsSigV4 {
            region: "eu-central-1".to_string()
        }
    );
}

#[test]
fn a_bearer_token_outranks_sigv4_the_way_python_resolves_it() {
    // Python's get_request_headers reads `api_key` as the Bedrock bearer token
    // and only falls back to the env when the caller passed none, so each case
    // pins one of its precedence rules. Signing as the host principal when a
    // bearer identity is configured would cross an account and quota boundary.
    let bedrock_env =
        |key: &str| (key == "AWS_BEARER_TOKEN_BEDROCK").then(|| "from-env".to_string());
    let no_env = |_: &str| None;
    let resolve = |api_key, env: &dyn Fn(&str) -> Option<String>| {
        BEDROCK_CHAT_COMPLETIONS_CONFIG
            .auth(
                api_key,
                "eu-central-1/anthropic.claude-v2",
                &Map::new(),
                env,
            )
            .expect("auth resolves")
    };
    let bearer = |token: &str| ChatCompletionsAuth::Bearer {
        token: token.to_string(),
    };
    let sigv4 = ChatCompletionsAuth::AwsSigV4 {
        region: "eu-central-1".to_string(),
    };

    // A caller-supplied key is the bearer token, and outranks the env.
    assert_eq!(
        resolve(Some("bedrock-api-key"), &bedrock_env),
        bearer("bedrock-api-key")
    );
    // No key, so the env supplies it.
    assert_eq!(resolve(None, &bedrock_env), bearer("from-env"));
    // An empty key is not a bearer token, and deliberately does NOT reach for
    // the env, which is what Python's `is not None` check does.
    assert_eq!(resolve(Some(""), &bedrock_env), sigv4);
    // Whitespace is truthy in Python, so it stays a bearer token rather than
    // silently becoming a host-credentialed SigV4 request.
    assert_eq!(resolve(Some("  "), &no_env), bearer("  "));
    // Neither present, so SigV4 as before.
    assert_eq!(resolve(None, &no_env), sigv4);
}

#[test]
fn normalizes_a_converse_response_into_openai_shape() {
    let response = transform_response(json!({
        "output": {"message": {"role": "assistant", "content": [
            {"text": "hello"}, {"text": " there"}
        ]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}
    }))
    .expect("response transforms");

    assert_eq!(response.model, "anthropic.claude-sonnet-4-5-v1:0");
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
fn maps_converse_stop_reasons_python_maps() {
    for (provider_reason, expected) in [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("guardrail_intervened", "content_filter"),
        // Converse emits this one, and Python's `_FINISH_REASON_MAP` carries
        // it. Folding it into `stop` reports a filtered completion as a normal
        // one to anything keying on the finish reason.
        ("content_filtered", "content_filter"),
        ("content_filter", "content_filter"),
    ] {
        let response = transform_response(json!({
            "output": {"message": {"content": [{"text": "x"}]}},
            "stopReason": provider_reason,
            "usage": {"inputTokens": 1, "outputTokens": 1}
        }))
        .expect("response transforms");
        assert_eq!(
            response.choices[0].finish_reason, expected,
            "stopReason {provider_reason}"
        );
    }
}

#[test]
fn reports_an_empty_converse_answer_as_an_empty_string_not_null() {
    // Converse assigns the joined text unconditionally
    // (`chat_completion_message["content"] = content_str`), unlike Anthropic's
    // `merged_text or None`, so an empty answer is `""` on both paths. A caller
    // calling `.strip()` on it would break on the Rust path alone. Reachable
    // through a filtered or guardrail-intervened response.
    for content in [json!([]), json!([{"text": ""}])] {
        let response = transform_response(json!({
            "output": {"message": {"content": content}},
            "stopReason": "content_filtered",
            "usage": {"inputTokens": 1, "outputTokens": 0}
        }))
        .expect("response transforms");
        assert_eq!(response.choices[0].message.content, Some(String::new()));
    }
}

#[test]
fn reports_the_total_tokens_converse_sent_rather_than_recomputing_them() {
    // Python reads `usage["totalTokens"]` straight through here, where Anthropic
    // has no such field and adds the two counts instead. The two agree while the
    // gate declines every cache_control request, so this is what keeps them
    // agreeing if that ever widens.
    let response = transform_response(json!({
        "output": {"message": {"content": [{"text": "x"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 4, "cacheReadInputTokens": 7, "totalTokens": 14}
    }))
    .expect("response transforms");
    assert_eq!(
        response.usage.total_tokens, 14,
        "provider total was recomputed"
    );
    assert_eq!(response.usage.prompt_tokens, 17);
    assert_eq!(response.usage.completion_tokens, 4);
}

#[test]
fn falls_back_to_the_computed_total_when_converse_omits_it() {
    // Python raises a KeyError on a body with no `totalTokens`. Reporting a zero
    // instead would be a worse divergence than the one above, so the computed
    // total stands in.
    let response = transform_response(json!({
        "output": {"message": {"content": [{"text": "x"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 4}
    }))
    .expect("response transforms");
    assert_eq!(response.usage.total_tokens, 14);
}

#[test]
fn declines_a_cache_control_message_so_widening_the_gate_is_a_red_test() {
    // Converse only reports cache token counts when the request carries a
    // cachePoint block, which is why the provider total and the computed one
    // cannot disagree today. This is the tripwire: whoever widens the gate to
    // admit prompt caching has to come back and re-check the usage mapping
    // rather than discovering a silent number change in production.
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
fn folds_converse_cache_tokens_into_prompt_tokens() {
    let response = transform_response(json!({
        "output": {"message": {"content": [{"text": "x"}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 2,
            "cacheReadInputTokens": 5,
            "cacheWriteInputTokens": 3
        }
    }))
    .expect("response transforms");
    assert_eq!(response.usage.prompt_tokens, 18);
    assert_eq!(response.usage.prompt_tokens_details.cached_tokens, 5);
    assert_eq!(
        response.usage.prompt_tokens_details.cache_creation_tokens,
        3
    );
    assert_eq!(response.usage.prompt_tokens_details.text_tokens, 10);
}

#[test]
fn declines_a_response_carrying_a_tool_use_block() {
    let err = transform_response(json!({
        "output": {"message": {"content": [
            {"toolUse": {"toolUseId": "t1", "name": "f", "input": {}}}
        ]}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": 1, "outputTokens": 1}
    }))
    .expect_err("tool use block");
    assert_eq!(
        err,
        CoreError::Unsupported("non-text response content block")
    );
}

#[test]
fn errors_on_a_response_missing_required_fields() {
    assert_eq!(
        transform_response(json!("nope")).expect_err("not an object"),
        CoreError::InvalidResponse("converse response is not an object".to_string())
    );
    assert_eq!(
        transform_response(json!({"usage": {}})).expect_err("no output"),
        CoreError::MissingField("output.message.content")
    );
    assert_eq!(
        transform_response(json!({"output": {"message": {"content": []}}})).expect_err("no usage"),
        CoreError::MissingField("usage")
    );
}

#[test]
fn accepts_aws_call_configuration_without_serializing_it() {
    let call_config = json!({
        "maxTokens": 16,
        "aws_access_key_id": "AKIA",
        "aws_secret_access_key": "secret",
        "aws_session_token": "token",
        "aws_region_name": "us-east-1",
        "aws_profile_name": "litellm-stage",
        "aws_role_name": "role",
        "aws_session_name": "session",
        "aws_web_identity_token": "wit",
        "aws_sts_endpoint": "https://sts.example",
        "aws_external_id": "ext",
        "aws_bedrock_runtime_endpoint": "https://vpce.internal"
    });
    assert_eq!(
        reason(
            json!([{"role": "user", "content": "hi"}]),
            call_config.clone()
        ),
        None
    );
    let body = transform(json!([{"role": "user", "content": "hi"}]), call_config);
    assert_eq!(
        body,
        json!({
            "inferenceConfig": {"maxTokens": 16},
            "messages": [{"role": "user", "content": [{"text": "hi"}]}]
        }),
        "aws call configuration must not reach the Converse body"
    );
}

#[test]
fn leaves_a_complete_converse_url_untouched() {
    let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
    let already_built =
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-v2%3A0/converse";
    assert_eq!(
        config
            .complete_url(
                Some(already_built),
                "anthropic.claude-v2",
                &Map::new(),
                &|_| None
            )
            .expect("url builds"),
        already_built,
        "a host that encoded the model id itself must not have it re-derived"
    );
}

#[test]
fn host_supplied_credentials_outrank_ambient_profile_and_role_state() {
    use crate::providers::bedrock::aws_base::host_supplied_credentials;

    let supplied = params(json!({
        "aws_access_key_id": "AKIAHOST",
        "aws_secret_access_key": "hostsecret",
        "aws_session_token": "hosttoken"
    }));
    let credentials = host_supplied_credentials(&supplied).expect("host credentials");
    assert_eq!(credentials.access_key_id(), "AKIAHOST");
    assert_eq!(credentials.secret_access_key(), "hostsecret");
    assert_eq!(credentials.session_token(), Some("hosttoken"));

    // Without a full static pair there is nothing to honor, so the core falls
    // back to deriving credentials itself.
    assert!(host_supplied_credentials(&params(json!({"aws_access_key_id": "AKIA"}))).is_none());
    assert!(
        host_supplied_credentials(&params(
            json!({"aws_access_key_id": "  ", "aws_secret_access_key": "s"})
        ))
        .is_none()
    );
    assert!(host_supplied_credentials(&Map::new()).is_none());
}
