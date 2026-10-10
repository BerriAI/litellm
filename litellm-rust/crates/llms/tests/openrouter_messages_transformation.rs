use litellm_llms::{
    base_llm::{auth::AuthScheme, messages::transformation::BaseMessagesConfig},
    openrouter::messages::transformation::OPENROUTER_MESSAGES_CONFIG,
};
use litellm_llms_types::formats::messages::MessagesResponse;
use litellm_router_types::LitellmParams;
use rstest::rstest;
use serde_json::{Value, json};

fn no_env(_: &str) -> Option<String> {
    None
}

fn response(usage: Value, extra: Value) -> MessagesResponse {
    let Value::Object(extra) = extra else {
        unreachable!()
    };
    serde_json::from_value(Value::Object(
        json!({
            "id": "msg_test", "type": "message", "role": "assistant", "model": "native-test",
            "content": [], "stop_reason": "end_turn", "stop_sequence": null, "usage": usage,
        })
        .as_object()
        .unwrap()
        .clone()
        .into_iter()
        .chain(extra)
        .collect(),
    ))
    .unwrap()
}

#[rstest]
#[case::usage_cost(json!({"input_tokens": 10, "output_tokens": 12, "cost": 0.00021}), json!({}), Some(0.00021))]
#[case::free(json!({"input_tokens": 10, "output_tokens": 12, "cost": 0}), json!({}), Some(0.0))]
#[case::no_cost(json!({"input_tokens": 10, "output_tokens": 12}), json!({}), None)]
#[case::no_usage(Value::Null, json!({}), None)]
#[case::top_level_cost_is_not_openrouters(json!({"input_tokens": 1}), json!({"cost": 0.5}), None)]
fn reported_cost_is_the_usage_cost(
    #[case] usage: Value,
    #[case] extra: Value,
    #[case] expected: Option<f64>,
) {
    assert_eq!(
        OPENROUTER_MESSAGES_CONFIG
            .reported_cost(&response(usage, extra))
            .and_then(|cost| cost.as_f64()),
        expected
    );
}

#[rstest]
#[case::default(None, &[], "https://openrouter.ai/api/v1/messages")]
#[case::env_base(None, &[("OPENROUTER_API_BASE", "https://gateway.test/api/v1")], "https://gateway.test/api/v1/messages")]
#[case::explicit_base(Some("https://proxy.test/api"), &[("OPENROUTER_API_BASE", "https://ignored.test")], "https://proxy.test/api/v1/messages")]
fn the_url_is_the_openrouter_messages_endpoint(
    #[case] api_base: Option<&str>,
    #[case] env: &[(&str, &str)],
    #[case] expected: &str,
) {
    let lookup = |name: &str| {
        env.iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value.to_string())
    };
    assert_eq!(
        OPENROUTER_MESSAGES_CONFIG
            .get_complete_url(
                api_base,
                "native-test",
                &LitellmParams::default(),
                false,
                &lookup
            )
            .unwrap(),
        expected
    );
}

#[rstest]
fn extended_cache_hints_reach_openrouter_unchanged() {
    let body = json!({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]});
    let wire = OPENROUTER_MESSAGES_CONFIG.wire_body(body.clone());
    assert_eq!(wire, body);
}

#[rstest]
#[case::param(Some("sk-or"), &[], "sk-or")]
#[case::env(None, &[("OPENROUTER_API_KEY", "sk-env")], "sk-env")]
fn the_openrouter_key_is_sent_as_a_bearer(
    #[case] api_key: Option<&str>,
    #[case] env: &[(&str, &str)],
    #[case] expected: &str,
) {
    let lookup = |name: &str| {
        env.iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value.to_string())
    };
    let validated = OPENROUTER_MESSAGES_CONFIG
        .validate_environment(
            Vec::new(),
            api_key,
            "native-test",
            &LitellmParams::default(),
            &lookup,
        )
        .unwrap();
    assert!(matches!(
        validated.auth,
        AuthScheme::Credential { placement: litellm_auth::CredentialPlacement::Bearer, ref secret }
            if secret.expose() == expected
    ));
}

#[rstest]
fn a_call_without_a_key_names_the_openrouter_variable() {
    assert!(matches!(
        OPENROUTER_MESSAGES_CONFIG
            .validate_environment(
                Vec::new(),
                None,
                "native-test",
                &LitellmParams::default(),
                &no_env
            )
            .unwrap_err(),
        litellm_llms::Error::Auth(litellm_auth::Error::MissingApiKey {
            provider: "OpenRouter",
            environment_variable: "OPENROUTER_API_KEY",
        })
    ));
}
