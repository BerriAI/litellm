use litellm_auth::AuthServices;
use litellm_llms::{
    Error,
    base_llm::{
        auth::resolve_auth,
        messages::{context::MessagesTransformContext, transformation::BaseMessagesConfig},
    },
    baseten::messages::transformation::BASETEN_MESSAGES_CONFIG,
};
use litellm_llms_types::formats::messages::{MessagesRequest, MessagesResponse};
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case::default(None, None, "https://inference.baseten.co/v1/messages")]
#[case::root(
    Some("https://gateway.test/"),
    None,
    "https://gateway.test/v1/messages"
)]
#[case::versioned(
    Some("https://gateway.test/v1/"),
    None,
    "https://gateway.test/v1/messages"
)]
#[case::complete(
    Some("https://gateway.test/v1/messages/"),
    None,
    "https://gateway.test/v1/messages"
)]
#[case::environment(
    None,
    Some("https://environment.test/v1"),
    "https://environment.test/v1/messages"
)]
#[case::explicit_wins(
    Some("https://gateway.test"),
    Some("https://environment.test"),
    "https://gateway.test/v1/messages"
)]
fn resolves_normal_and_stream_urls(
    #[case] base: Option<&str>,
    #[case] env_base: Option<&str>,
    #[case] expected: &str,
) {
    let env = |name: &str| {
        (name == "BASETEN_API_BASE")
            .then(|| env_base.map(str::to_string))
            .flatten()
    };
    assert_eq!(
        BASETEN_MESSAGES_CONFIG
            .get_complete_url(base, "model", &env)
            .unwrap(),
        expected
    );
    assert_eq!(
        BASETEN_MESSAGES_CONFIG
            .complete_stream_url(base, "model", &env)
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::explicit(Some("deployment-key"), Some("env-key"), "deployment-key")]
#[case::environment(None, Some("env-key"), "env-key")]
#[case::blank_explicit(Some(" "), Some("env-key"), "env-key")]
#[tokio::test]
async fn resolves_a_bearer_and_removes_anthropic_credentials(
    #[case] key: Option<&str>,
    #[case] env_key: Option<&str>,
    #[case] expected: &str,
) {
    let env = |name: &str| {
        (name == "BASETEN_API_KEY")
            .then(|| env_key.map(str::to_string))
            .flatten()
    };
    let validated = BASETEN_MESSAGES_CONFIG
        .validate_environment(
            vec![
                ("X-Api-Key".into(), "unrelated-key".into()),
                ("x-trace".into(), "trace".into()),
            ],
            key,
            "model",
            &env,
        )
        .unwrap();
    let authenticated = resolve_auth(&AuthServices::default(), validated, &env)
        .await
        .unwrap();
    assert_eq!(
        authenticated.headers,
        vec![
            ("x-trace".into(), "trace".into()),
            ("authorization".into(), format!("Bearer {expected}"))
        ]
    );
}

#[rstest]
#[case::bearer("Bearer caller-key")]
#[case::api_key("Api-Key caller-key")]
#[tokio::test]
async fn preserves_forwarded_authorization(#[case] authorization: &str) {
    let headers = vec![("Authorization".into(), authorization.into())];
    let validated = BASETEN_MESSAGES_CONFIG
        .validate_environment(headers.clone(), Some("deployment-key"), "model", &|_| None)
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
        BASETEN_MESSAGES_CONFIG.validate_environment(vec![], None, "model", &|_| None),
        Err(Error::Auth(litellm_auth::Error::MissingApiKey { .. }))
    ));
}

#[rstest]
fn preserves_native_messages_and_tool_responses() {
    let request: MessagesRequest = serde_json::from_value(json!({
        "model": "test-model", "max_tokens": 32, "system": "be terse",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto"}, "metadata": {"user_id": "test-user"},
        "top_k": 20, "stop_sequences": ["STOP"], "stream": true
    }))
    .unwrap();
    let expected = request.clone();
    let shaped = BASETEN_MESSAGES_CONFIG
        .shape_request(request, true)
        .unwrap();
    assert_eq!(
        BASETEN_MESSAGES_CONFIG
            .transform_anthropic_messages_request(shaped, &MessagesTransformContext::default())
            .unwrap(),
        expected
    );
    let response: MessagesResponse = serde_json::from_value(json!({
        "id": "msg-test", "type": "message", "role": "assistant", "model": "test-model",
        "content": [{"type": "tool_use", "id": "call-test", "name": "lookup", "input": {}}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 2, "output_tokens": 3, "cache_read_input_tokens": 1}
    })).unwrap();
    assert_eq!(
        BASETEN_MESSAGES_CONFIG
            .transform_anthropic_messages_response("test-model", response.clone())
            .unwrap(),
        response
    );
}
