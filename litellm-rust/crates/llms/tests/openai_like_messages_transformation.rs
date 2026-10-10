use litellm_llms::openai_like::messages::transformation::{
    complete_messages_url, portable_cache_control,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::root("https://native.test", "https://native.test/v1/messages")]
#[case::trailing_slash("https://native.test/", "https://native.test/v1/messages")]
#[case::versioned("https://native.test/v1/", "https://native.test/v1/messages")]
#[case::full_url(
    "https://native.test/v3/v1/messages/",
    "https://native.test/v3/v1/messages"
)]
#[case::gateway_prefix("https://native.test/v3", "https://native.test/v3/v1/messages")]
fn messages_url_appends_the_endpoint_once(#[case] base: &str, #[case] expected: &str) {
    assert_eq!(complete_messages_url(base), expected);
}

#[rstest]
#[case::extended(json!({"type": "ephemeral", "ttl": "1h", "scope": "global"}), Some(json!({"type": "ephemeral"})))]
#[case::missing_type(json!({"ttl": "1h"}), Some(json!({"type": "ephemeral"})))]
#[case::invalid_type(json!({"type": 7}), Some(json!({"type": "ephemeral"})))]
#[case::null(Value::Null, None)]
#[case::scalar(json!(false), None)]
fn portable_cache_only_rewrites_protocol_cache_hints(
    #[case] cache: Value,
    #[case] expected_cache: Option<Value>,
) {
    let application_data = json!({"cache_control": cache, "nested": {"cache_control": cache}});
    let body = json!({
        "cache_control": cache,
        "model": "native-test",
        "system": [{"type": "text", "text": "system", "cache_control": cache}],
        "tools": [{"name": "tool", "cache_control": cache, "input_schema": application_data}],
        "messages": [{"role": "user", "content": [
            {"type": "tool_use", "input": application_data, "cache_control": cache},
            {"type": "tool_result", "content": [{"type": "text", "text": "result", "cache_control": cache}], "cache_control": cache},
        ]}],
    });
    let hint = |value: Value| {
        let fields = value.as_object().unwrap().clone();
        Value::Object(
            fields
                .into_iter()
                .chain(
                    expected_cache
                        .clone()
                        .map(|cache| ("cache_control".into(), cache)),
                )
                .collect(),
        )
    };
    let expected = hint(json!({
        "model": "native-test",
        "system": [hint(json!({"type": "text", "text": "system"}))],
        "tools": [hint(json!({"name": "tool", "input_schema": application_data}))],
        "messages": [{"role": "user", "content": [
            hint(json!({"type": "tool_use", "input": application_data})),
            hint(json!({"type": "tool_result", "content": [hint(json!({"type": "text", "text": "result"}))]})),
        ]}],
    }));
    assert_eq!(portable_cache_control(body), expected);
}

mod compatible_host {
    use litellm_llms::{
        base_llm::auth::AuthScheme,
        openai_like::messages::transformation::{
            compatible_host_environment, compatible_host_url, without_billing_blocks,
        },
    };
    use rstest::rstest;
    use serde_json::json;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[rstest]
    #[case::default(None, &[], "https://host.test/v1/messages")]
    #[case::env_base(None, &[("HOST_API_BASE", "https://env.test/v1")], "https://env.test/v1/messages")]
    #[case::blank_env_is_absent(None, &[("HOST_API_BASE", " ")], "https://host.test/v1/messages")]
    #[case::explicit_base(Some("https://explicit.test/v3"), &[("HOST_API_BASE", "https://env.test")], "https://explicit.test/v3/v1/messages")]
    fn the_base_comes_from_the_call_then_the_environment_then_the_default(
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
            compatible_host_url(api_base, "HOST_API_BASE", "https://host.test", &lookup),
            expected
        );
    }

    #[rstest]
    #[case::param(Some("sk-host"), &[], "sk-host")]
    #[case::env(None, &[("HOST_API_KEY", "sk-env")], "sk-env")]
    #[case::blank_param_falls_back_to_env(Some(" "), &[("HOST_API_KEY", "sk-env")], "sk-env")]
    fn the_hosts_key_is_sent_as_a_bearer(
        #[case] api_key: Option<&str>,
        #[case] env: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        let lookup = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        let validated =
            compatible_host_environment(Vec::new(), api_key, "Host", "HOST_API_KEY", &lookup)
                .unwrap();
        assert!(matches!(
            validated.auth,
            AuthScheme::Credential { placement: litellm_auth::CredentialPlacement::Bearer, ref secret }
                if secret.expose() == expected
        ));
    }

    #[rstest]
    #[case::x_api_key(&[("X-Api-Key", "caller")])]
    #[case::bearer(&[("Authorization", "Bearer caller")])]
    fn a_callers_anthropic_credential_is_forwarded_over_the_hosts_key(
        #[case] forwarded: &[(&str, &str)],
    ) {
        let headers = forwarded
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        let validated =
            compatible_host_environment(headers, Some("sk-host"), "Host", "HOST_API_KEY", &no_env)
                .unwrap();
        assert!(matches!(validated.auth, AuthScheme::Forwarded));
    }

    #[rstest]
    fn the_hosts_key_is_still_required_when_a_credential_is_forwarded() {
        let headers = vec![("x-api-key".to_string(), "caller".to_string())];
        assert!(matches!(
            compatible_host_environment(headers, None, "Host", "HOST_API_KEY", &no_env)
                .unwrap_err(),
            litellm_llms::Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Host",
                environment_variable: "HOST_API_KEY",
            })
        ));
    }

    #[rstest]
    fn billing_system_blocks_never_leave_the_gateway() {
        let request = serde_json::from_value(json!({
            "model": "native-test", "max_tokens": 16,
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"}, {"type": "text", "text": "keep"}],
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        assert_eq!(
            serde_json::to_value(without_billing_blocks(request).params.system).unwrap(),
            json!([{"type": "text", "text": "keep"}])
        );
    }
}
