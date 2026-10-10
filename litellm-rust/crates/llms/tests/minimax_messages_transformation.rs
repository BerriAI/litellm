use litellm_auth::CredentialPlacement;
use litellm_llms::{
    base_llm::{
        auth::AuthScheme,
        messages::{context::MessagesTransformContext, transformation::BaseMessagesConfig},
    },
    minimax::messages::transformation::MINIMAX_MESSAGES_CONFIG as CONFIG,
};
use rstest::rstest;
use serde_json::{Value, json};

fn no_env(_: &str) -> Option<String> {
    None
}

#[rstest]
#[case::root("https://proxy.test", "https://proxy.test/v1/messages")]
#[case::trailing_slash("https://proxy.test/", "https://proxy.test/v1/messages")]
#[case::full_endpoint("https://proxy.test/v1/messages", "https://proxy.test/v1/messages")]
#[case::custom_path(
    "https://proxy.test/anthropic",
    "https://proxy.test/anthropic/v1/messages"
)]
#[case::versioned("https://proxy.test/v1", "https://proxy.test/v1/v1/messages")]
fn url_preserves_python_endpoint_rules(#[case] base: &str, #[case] expected: &str) {
    assert_eq!(
        CONFIG
            .get_complete_url(
                Some(base),
                "test-model",
                &Default::default(),
                false,
                &no_env
            )
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::explicit(Some("https://explicit.test"), &[("MINIMAX_API_BASE", "https://env.test")], "https://explicit.test/v1/messages")]
#[case::environment(None, &[("MINIMAX_API_BASE", "https://env.test")], "https://env.test/v1/messages")]

fn base_precedence(
    #[case] base: Option<&str>,
    #[case] env: &[(&str, &str)],
    #[case] expected: &str,
) {
    let lookup = |key: &str| {
        env.iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| value.to_string())
    };
    assert_eq!(
        CONFIG
            .get_complete_url(base, "test-model", &Default::default(), false, &lookup)
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::explicit(Some("explicit"), Some("environment"), "explicit")]
#[case::environment(None, Some("environment"), "environment")]
#[case::empty_explicit(Some(" "), Some("environment"), "environment")]
fn key_precedence(
    #[case] key: Option<&str>,
    #[case] env_key: Option<&str>,
    #[case] expected: &str,
) {
    let lookup = |name: &str| {
        if name == "MINIMAX_API_KEY" {
            env_key.map(str::to_string)
        } else {
            None
        }
    };
    let environment = CONFIG
        .validate_environment(vec![], key, "test-model", &Default::default(), &lookup)
        .unwrap();
    assert!(
        matches!(environment.auth, AuthScheme::Credential { placement: CredentialPlacement::Header("x-api-key"), ref secret } if secret.expose() == expected)
    );
}

#[rstest]
#[case::x_api_key("X-Api-Key", "caller")]
#[case::bearer("Authorization", "Bearer caller")]
fn forwarded_credentials_need_no_provider_key(#[case] name: &str, #[case] value: &str) {
    let headers = vec![(name.to_string(), value.to_string())];
    let environment = CONFIG
        .validate_environment(
            headers.clone(),
            None,
            "test-model",
            &Default::default(),
            &no_env,
        )
        .unwrap();
    assert_eq!(environment.headers, headers);
    assert!(matches!(environment.auth, AuthScheme::Forwarded));
}

#[rstest]
fn missing_key_is_a_provider_auth_error() {
    let error = CONFIG
        .validate_environment(vec![], None, "test-model", &Default::default(), &no_env)
        .unwrap_err();
    assert!(matches!(
        error,
        litellm_llms::Error::Auth(litellm_auth::Error::MissingApiKey {
            environment_variable: "MINIMAX_API_KEY",
            ..
        })
    ));
}

#[rstest]
#[case::string(json!("x-anthropic-billing-header: private"), None)]
#[case::blocks(json!([{"type": "text", "text": "x-anthropic-billing-header: private"}, {"type": "text", "text": "keep"}]), Some(json!([{"type": "text", "text": "keep"}])))]
#[case::ordinary(json!("keep"), Some(json!("keep")))]
fn only_billing_metadata_is_removed(#[case] system: Value, #[case] expected: Option<Value>) {
    let input = json!({
        "model": "test-model", "max_tokens": 1024, "system": system,
        "thinking": {"type": "enabled", "budget_tokens": 512},
        "tools": [{"name": "echo", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]
    });
    let request = serde_json::from_value(input.clone()).unwrap();
    let output = serde_json::to_value(
        CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(output.get("system"), expected.as_ref());
    assert_eq!(output["messages"], input["messages"]);
    assert_eq!(output["tools"], input["tools"]);
    assert_eq!(output["thinking"], input["thinking"]);
}
