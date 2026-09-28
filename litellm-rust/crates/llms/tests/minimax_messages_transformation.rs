use litellm_auth::CredentialPlacement;
use litellm_llms::{
    Error,
    base_llm::{
        auth::AuthScheme,
        messages::{
            context::{MessagesModelCapabilities, MessagesTransformContext},
            transformation::BaseAnthropicMessagesConfig,
        },
    },
    minimax::messages::transformation::MINIMAX_MESSAGES_CONFIG,
};
use litellm_types::llms::anthropic_messages::{
    anthropic_request::AnthropicMessagesRequest, anthropic_response::AnthropicMessagesResponse,
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

#[fixture]
fn request() -> AnthropicMessagesRequest {
    serde_json::from_value(json!({
        "model": "test-model", "max_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap()
}

#[rstest]
#[case::default(None, None, "https://api.minimax.io/anthropic/v1/messages")]
#[case::explicit(
    Some("https://custom.example/anthropic"),
    Some("https://env.example"),
    "https://custom.example/anthropic/v1/messages"
)]
#[case::environment(
    None,
    Some("https://env.example/anthropic"),
    "https://env.example/anthropic/v1/messages"
)]
#[case::trailing_slash(
    Some("https://custom.example/anthropic/"),
    None,
    "https://custom.example/anthropic/v1/messages"
)]
#[case::complete(
    Some("https://api.minimaxi.com/anthropic/v1/messages"),
    None,
    "https://api.minimaxi.com/anthropic/v1/messages"
)]
#[case::complete_trailing_slash(
    Some("https://custom.example/v1/messages/"),
    None,
    "https://custom.example/v1/messages"
)]
#[case::blank_explicit(
    Some(" "),
    Some("https://env.example/anthropic"),
    "https://env.example/anthropic/v1/messages"
)]
fn endpoint_resolution(
    #[case] explicit: Option<&str>,
    #[case] environment: Option<&str>,
    #[case] expected: &str,
) {
    let lookup = |name: &str| {
        (name == "MINIMAX_API_BASE")
            .then_some(environment)
            .flatten()
            .map(str::to_string)
    };
    assert_eq!(
        MINIMAX_MESSAGES_CONFIG
            .get_complete_url(explicit, "test-model", &lookup)
            .unwrap(),
        expected
    );
    assert_eq!(
        MINIMAX_MESSAGES_CONFIG
            .complete_stream_url(explicit, "test-model", &lookup)
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::environment_only(None, Some("env-key"), "env-key")]
#[case::explicit_wins(Some("explicit-key"), Some("env-key"), "explicit-key")]
#[case::explicit_only(Some("explicit-key"), None, "explicit-key")]
#[case::blank_explicit(Some(" "), Some("env-key"), "env-key")]
fn credential_resolution(
    #[case] explicit: Option<&str>,
    #[case] environment: Option<&str>,
    #[case] expected: &str,
) {
    let lookup = |name: &str| {
        (name == "MINIMAX_API_KEY")
            .then_some(environment)
            .flatten()
            .map(str::to_string)
    };
    let validated = MINIMAX_MESSAGES_CONFIG
        .validate_environment(Vec::new(), explicit, "test-model", &lookup)
        .unwrap();
    assert!(matches!(validated.auth, AuthScheme::Credential {
        placement: CredentialPlacement::Header("x-api-key"), secret
    } if secret.expose() == expected));
}

#[rstest]
#[case::anthropic_api_key("ANTHROPIC_API_KEY")]
#[case::anthropic_auth_token("ANTHROPIC_AUTH_TOKEN")]
fn anthropic_credentials_do_not_authenticate_minimax(#[case] name: &str) {
    let lookup = |key: &str| (key == name).then(|| "wrong-provider-key".to_string());
    assert!(matches!(
        MINIMAX_MESSAGES_CONFIG.validate_environment(Vec::new(), None, "test-model", &lookup),
        Err(Error::Auth(litellm_auth::Error::MissingApiKey {
            environment_variable: "MINIMAX_API_KEY",
            ..
        }))
    ));
}

#[rstest]
#[case::api_key("X-Api-Key", "caller-key")]
#[case::bearer("Authorization", "Bearer caller-token")]
fn forwarded_credentials_are_preserved(#[case] name: &str, #[case] value: &str) {
    let headers = vec![(name.to_string(), value.to_string())];
    let validated = MINIMAX_MESSAGES_CONFIG
        .validate_environment(headers.clone(), Some("explicit-key"), "test-model", &|_| {
            None
        })
        .unwrap();
    assert_eq!(validated.headers, headers);
    assert!(matches!(validated.auth, AuthScheme::Forwarded));
}

#[rstest]
fn blank_bearer_uses_minimax_key() {
    let validated = MINIMAX_MESSAGES_CONFIG
        .validate_environment(
            vec![("Authorization".into(), "Bearer ".into())],
            Some("key"),
            "test-model",
            &|_| None,
        )
        .unwrap();
    assert!(
        matches!(validated.auth, AuthScheme::Credential { secret, .. } if secret.expose() == "key")
    );
}

#[rstest]
#[case::billing_string(json!("x-anthropic-billing-header: cc_version=1"), None)]
#[case::ordinary_string(json!("instructions"), Some(json!("instructions")))]
#[case::all_billing(json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"}]), None)]
#[case::mixed_blocks(json!([
    {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
    {"type": "text", "text": "instructions", "cache_control": {"type": "ephemeral"}}
]), Some(json!([{ "type": "text", "text": "instructions", "cache_control": {"type": "ephemeral"}}])))]
fn billing_metadata_is_removed(
    request: AnthropicMessagesRequest,
    #[case] system: Value,
    #[case] expected: Option<Value>,
) {
    let input = AnthropicMessagesRequest {
        params: litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesOptionalParams {
            system: Some(serde_json::from_value(system).unwrap()), ..request.params
        }, ..request
    };
    let transformed = MINIMAX_MESSAGES_CONFIG
        .transform_anthropic_messages_request(input, &MessagesTransformContext::default())
        .unwrap();
    let wire = serde_json::to_value(transformed).unwrap();
    assert_eq!(wire.get("system"), expected.as_ref());
    assert_eq!(wire["messages"], json!([{"role": "user", "content": "hi"}]));
}

#[rstest]
fn system_role_is_folded_before_billing_metadata_is_removed() {
    let request = serde_json::from_value(json!({
        "model": "test-model", "max_tokens": 128,
        "messages": [
            {"role": "system", "content": "x-anthropic-billing-header: cc_version=1"},
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "hi"}
        ]
    }))
    .unwrap();
    let transformed = MINIMAX_MESSAGES_CONFIG
        .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
        .unwrap();
    let wire = serde_json::to_value(transformed).unwrap();
    assert_eq!(
        wire["system"],
        json!([{"type": "text", "text": "instructions"}])
    );
    assert_eq!(wire["messages"], json!([{"role": "user", "content": "hi"}]));
}

#[rstest]
fn request_preserves_thinking_tools_and_media() {
    let input = serde_json::from_value(json!({
        "model": "test-model", "max_tokens": 128,
        "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"},
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}], "tool_choice": {"type": "auto"},
        "messages": [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "reasoning", "signature": "signature"},
                {"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "result"},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/image.png"}},
                {"type": "video", "source": {"type": "url", "url": "https://example.com/video.mp4"}}]}
        ]
    })).unwrap();
    let context = MessagesTransformContext::with_lookup(
        MessagesModelCapabilities {
            supports_reasoning: true,
            supports_adaptive_thinking: true,
            supports_output_config: true,
            ..Default::default()
        },
        false,
        &|_: &str| None,
    );
    let shaped = MINIMAX_MESSAGES_CONFIG.shape_request(input, false).unwrap();
    assert_eq!(
        MINIMAX_MESSAGES_CONFIG
            .transform_anthropic_messages_request(shaped.clone(), &context)
            .unwrap(),
        shaped
    );
}

#[rstest]
fn max_tokens_is_required(request: AnthropicMessagesRequest) {
    let input = AnthropicMessagesRequest {
        params: litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesOptionalParams {
            max_tokens: None, ..request.params
        }, ..request
    };
    assert_eq!(
        MINIMAX_MESSAGES_CONFIG
            .transform_anthropic_messages_request(input, &MessagesTransformContext::default()),
        Err(Error::MissingField("max_tokens"))
    );
}

#[rstest]
fn response_is_preserved() {
    let response: AnthropicMessagesResponse = serde_json::from_value(json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "test-model",
        "content": [{"type": "thinking", "thinking": "reasoning", "signature": "signature"},
            {"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {}}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 12, "output_tokens": 8,
            "cache_read_input_tokens": 4, "cache_creation_input_tokens": 2}
    }))
    .unwrap();
    assert_eq!(
        MINIMAX_MESSAGES_CONFIG
            .transform_anthropic_messages_response("test-model", response.clone())
            .unwrap(),
        response
    );
}
