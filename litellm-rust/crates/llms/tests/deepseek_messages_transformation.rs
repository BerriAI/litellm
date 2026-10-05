use litellm_auth::AuthServices;
use litellm_llms::{
    Error,
    base_llm::{
        auth::resolve_auth,
        messages::{context::MessagesTransformContext, transformation::BaseMessagesConfig},
    },
    deepseek::messages::transformation::DEEPSEEK_MESSAGES_CONFIG,
};
use litellm_llms_types::formats::messages::MessagesRequest;
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case::default(None, &[], "https://api.deepseek.com/anthropic/v1/messages")]
#[case::root(Some("https://gateway.test/"), &[], "https://gateway.test/anthropic/v1/messages")]
#[case::versioned(Some("https://gateway.test/v1"), &[], "https://gateway.test/anthropic/v1/messages")]
#[case::beta(Some("https://gateway.test/beta"), &[], "https://gateway.test/anthropic/v1/messages")]
#[case::anthropic(Some("https://gateway.test/anthropic/"), &[], "https://gateway.test/anthropic/v1/messages")]
#[case::anthropic_versioned(Some("https://gateway.test/anthropic/v1"), &[], "https://gateway.test/anthropic/v1/messages")]
#[case::complete(Some("https://gateway.test/anthropic/v1/messages/"), &[], "https://gateway.test/anthropic/v1/messages")]
#[case::unscoped_complete(Some("https://gateway.test/v1/messages"), &[], "https://gateway.test/anthropic/v1/messages")]
#[case::environment(None, &[("DEEPSEEK_API_BASE", "https://environment.test")], "https://environment.test/anthropic/v1/messages")]
#[case::anthropic_env_wins(None, &[("DEEPSEEK_ANTHROPIC_API_BASE", "https://anthropic.test"), ("DEEPSEEK_API_BASE", "https://environment.test")], "https://anthropic.test/anthropic/v1/messages")]
#[case::explicit_wins(Some("https://gateway.test"), &[("DEEPSEEK_ANTHROPIC_API_BASE", "https://anthropic.test")], "https://gateway.test/anthropic/v1/messages")]
fn resolves_normal_and_stream_urls(
    #[case] base: Option<&str>,
    #[case] environment: &[(&str, &str)],
    #[case] expected: &str,
) {
    let env = |name: &str| {
        environment
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value.to_string())
    };
    assert_eq!(
        DEEPSEEK_MESSAGES_CONFIG
            .get_complete_url(base, "test-model", &env)
            .unwrap(),
        expected
    );
    assert_eq!(
        DEEPSEEK_MESSAGES_CONFIG
            .complete_stream_url(base, "test-model", &env)
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::explicit(Some("deployment-key"), "deployment-key")]
#[case::environment(None, "env-key")]
#[case::blank_explicit(Some(" "), "env-key")]
#[tokio::test]
async fn resolves_the_deepseek_key(#[case] key: Option<&str>, #[case] expected: &str) {
    let env = |name: &str| (name == "DEEPSEEK_API_KEY").then(|| "env-key".to_string());
    let validated = DEEPSEEK_MESSAGES_CONFIG
        .validate_environment(
            vec![("x-trace".into(), "trace".into())],
            key,
            "test-model",
            &env,
        )
        .unwrap();
    assert_eq!(
        resolve_auth(&AuthServices::default(), validated, &env)
            .await
            .unwrap()
            .headers,
        vec![
            ("x-trace".into(), "trace".into()),
            ("x-api-key".into(), expected.to_string())
        ]
    );
}

#[rstest]
#[case::api_key("X-Api-Key", "caller-key")]
#[case::bearer("Authorization", "Bearer caller-key")]
#[tokio::test]
async fn preserves_forwarded_credentials(#[case] name: &str, #[case] value: &str) {
    let headers = vec![(name.into(), value.into())];
    let validated = DEEPSEEK_MESSAGES_CONFIG
        .validate_environment(
            headers.clone(),
            Some("deployment-key"),
            "test-model",
            &|_| None,
        )
        .unwrap();
    assert_eq!(
        resolve_auth(&AuthServices::default(), validated, &|_| None)
            .await
            .unwrap()
            .headers,
        headers
    );
}

#[rstest]
fn missing_credentials_fail_before_sending() {
    assert!(matches!(
        DEEPSEEK_MESSAGES_CONFIG.validate_environment(vec![], None, "test-model", &|_| None),
        Err(Error::Auth(litellm_auth::Error::MissingApiKey { .. }))
    ));
}

#[rstest]
#[case::explicit_custom(Some("custom"), None)]
#[case::implicit_custom(None, None)]
#[case::server_tool(Some("web_search_20250305"), Some("web_search_20250305"))]
fn sanitizes_custom_tools_without_changing_thinking_history(
    #[case] tool_type: Option<&str>,
    #[case] expected_type: Option<&str>,
) {
    let tool = serde_json::Value::Object(
        [
            ("name".into(), json!("lookup")),
            ("input_schema".into(), json!({"type": "object"})),
        ]
        .into_iter()
        .chain(tool_type.map(|value| ("type".into(), json!(value))))
        .collect(),
    );
    let original = json!({
        "model": "test-model", "max_tokens": 32,
        "messages": [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "history", "signature": "test-signature"}, {"type": "text", "text": "hello"}]},
            {"role": "user", "content": "continue"}
        ],
        "thinking": {"type": "enabled", "budget_tokens": 16}, "output_config": {"effort": "high"},
        "tools": [tool], "tool_choice": {"type": "auto"}
    });
    let request: MessagesRequest = serde_json::from_value(original.clone()).unwrap();
    let transformed = DEEPSEEK_MESSAGES_CONFIG
        .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
        .unwrap();
    let body = serde_json::to_value(&transformed).unwrap();
    assert_eq!(
        body["tools"][0]
            .get("type")
            .and_then(serde_json::Value::as_str),
        expected_type
    );
    assert_eq!(body["tools"][0]["name"], original["tools"][0]["name"]);
    assert_eq!(
        body["tools"][0]["input_schema"],
        original["tools"][0]["input_schema"]
    );
    assert_eq!(body["messages"], original["messages"]);
    assert_eq!(body["thinking"], original["thinking"]);
    assert_eq!(body["output_config"], original["output_config"]);
    assert_eq!(
        DEEPSEEK_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                transformed.clone(),
                &MessagesTransformContext::default()
            )
            .unwrap(),
        transformed
    );
}

#[rstest]
fn folds_system_role_messages_into_the_system_prompt() {
    let request: MessagesRequest = serde_json::from_value(json!({
        "model": "test-model", "max_tokens": 16, "system": "existing prompt",
        "messages": [{"role": "system", "content": "extra prompt"}, {"role": "user", "content": "hello"}]
    })).unwrap();
    let transformed = DEEPSEEK_MESSAGES_CONFIG
        .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
        .unwrap();
    let body = serde_json::to_value(transformed).unwrap();
    assert_eq!(
        body["messages"],
        json!([{"role": "user", "content": "hello"}])
    );
    assert_eq!(
        body["system"],
        json!([{"type": "text", "text": "existing prompt"}, {"type": "text", "text": "extra prompt"}])
    );
}
