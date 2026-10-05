use litellm_auth::AuthServices;
use litellm_llms::{
    base_llm::{
        auth::resolve_auth,
        chat::transformation::{BaseConfig, Unsupported},
    },
    baseten::chat::transformation::BASETEN_CHAT_COMPLETIONS_CONFIG,
};
use litellm_llms_types::formats::chat_completions::ChatMessage;
use rstest::{fixture, rstest};
use serde_json::{Map, Value, json};

#[fixture]
fn messages() -> Vec<ChatMessage> {
    serde_json::from_value(json!([{"role": "user", "content": "hello"}])).unwrap()
}

#[rstest]
#[case::default(None, "https://inference.baseten.co/v1/chat/completions")]
#[case::root(
    Some("https://gateway.test/"),
    "https://gateway.test/v1/chat/completions"
)]
#[case::versioned(
    Some("https://gateway.test/v1/"),
    "https://gateway.test/v1/chat/completions"
)]
#[case::complete(
    Some("https://gateway.test/v1/chat/completions"),
    "https://gateway.test/v1/chat/completions"
)]
fn resolves_the_chat_endpoint(#[case] base: Option<&str>, #[case] expected: &str) {
    assert_eq!(
        BASETEN_CHAT_COMPLETIONS_CONFIG
            .get_complete_url(base, "test-model", &Map::new(), &|_| None)
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::with_legacy_limit(Some(8))]
#[case::without_legacy_limit(None)]
fn maps_the_token_limit_and_forwards_supported_parameters(
    messages: Vec<ChatMessage>,
    #[case] legacy_limit: Option<u64>,
) {
    let params: Map<String, Value> = [
        ("max_completion_tokens".into(), json!(12)),
        ("temperature".into(), json!(0.5)),
        ("response_format".into(), json!({"type": "json_object"})),
        ("user".into(), json!("test-user")),
        ("extra_headers".into(), json!({"x-trace": "trace"})),
        ("max_retries".into(), json!(2)),
    ]
    .into_iter()
    .chain(legacy_limit.map(|limit| ("max_tokens".into(), json!(limit))))
    .collect();
    assert_eq!(
        BASETEN_CHAT_COMPLETIONS_CONFIG.unsupported_reason(&messages, &params),
        None
    );
    let body = BASETEN_CHAT_COMPLETIONS_CONFIG
        .transform_request("test-model", messages.clone(), params)
        .unwrap()
        .body;
    assert_eq!(
        body,
        json!({
            "model": "test-model", "messages": messages, "max_tokens": 12,
            "temperature": 0.5, "response_format": {"type": "json_object"}, "user": "test-user"
        })
    );
}

#[rstest]
#[case::streaming("stream", json!(true), "streaming")]
#[case::tools("tools", json!([{"type": "function", "function": {"name": "lookup"}}]), "unrecognized request parameter")]
#[case::unknown("unknown", json!(true), "unrecognized request parameter")]
fn rejects_requests_the_shared_adapter_cannot_complete(
    messages: Vec<ChatMessage>,
    #[case] name: &str,
    #[case] value: Value,
    #[case] reason: &'static str,
) {
    assert_eq!(
        BASETEN_CHAT_COMPLETIONS_CONFIG
            .unsupported_reason(&messages, &Map::from_iter([(name.to_string(), value)])),
        Some(Unsupported(reason))
    );
}

#[rstest]
#[tokio::test]
async fn uses_baseten_secrets_for_chat_auth_and_url() {
    let env = |name: &str| match name {
        "BASETEN_API_KEY" => Some("baseten-key".into()),
        "BASETEN_API_BASE" => Some("https://gateway.test/v1".into()),
        _ => None,
    };
    let validated = BASETEN_CHAT_COMPLETIONS_CONFIG
        .validate_environment(vec![], None, "test-model", &Map::new(), &env)
        .unwrap();
    assert_eq!(
        resolve_auth(&AuthServices::default(), validated, &env)
            .await
            .unwrap()
            .headers,
        vec![("authorization".into(), "Bearer baseten-key".into())]
    );
    assert_eq!(
        BASETEN_CHAT_COMPLETIONS_CONFIG
            .get_complete_url(None, "test-model", &Map::new(), &env)
            .unwrap(),
        "https://gateway.test/v1/chat/completions"
    );
}
