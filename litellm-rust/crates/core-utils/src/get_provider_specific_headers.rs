use litellm_llms_types::headers::{ProviderSpecificHeader, ProviderSpecificHeaders};
use serde_json::{Map, Value};

pub fn get_provider_specific_headers(
    provider_specific_header: Option<&ProviderSpecificHeaders>,
    custom_llm_provider: &str,
    model: &str,
    configured_headers: &Map<String, Value>,
) -> Map<String, Value> {
    let entries: &[ProviderSpecificHeader] = match provider_specific_header {
        None => &[],
        Some(ProviderSpecificHeaders::One(entry)) => std::slice::from_ref(entry),
        Some(ProviderSpecificHeaders::Many(entries)) => entries,
    };
    entries
        .iter()
        .filter(|entry| applies_to_call(entry, custom_llm_provider, model))
        .flat_map(|entry| contributed_headers(entry, configured_headers))
        .collect()
}

fn applies_to_call(entry: &ProviderSpecificHeader, custom_llm_provider: &str, model: &str) -> bool {
    (!entry.claude_models_only || model.to_lowercase().contains("claude"))
        && entry
            .custom_llm_provider
            .split(',')
            .any(|scoped| scoped.trim() == custom_llm_provider)
}

fn contributed_headers(
    entry: &ProviderSpecificHeader,
    configured_headers: &Map<String, Value>,
) -> Map<String, Value> {
    entry
        .extra_headers
        .iter()
        .filter(|(name, _)| {
            !entry.yields_to_configured
                || !configured_headers
                    .keys()
                    .any(|configured| configured.eq_ignore_ascii_case(name))
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::single_entry_for_the_provider(
        json!({"custom_llm_provider": "anthropic", "extra_headers": {"Authorization": "Bearer t", "Custom-Header": "v"}}),
        json!({"Authorization": "Bearer t", "Custom-Header": "v"}),
    )]
    #[case::single_entry_for_another_provider(
        json!({"custom_llm_provider": "openai", "extra_headers": {"Authorization": "Bearer t"}}),
        json!({}),
    )]
    #[case::provider_in_a_comma_separated_scope(
        json!({"custom_llm_provider": "bedrock,anthropic,vertex_ai", "extra_headers": {"anthropic-beta": "context-1m-2025-08-07"}}),
        json!({"anthropic-beta": "context-1m-2025-08-07"}),
    )]
    #[case::provider_missing_from_a_comma_separated_scope(
        json!({"custom_llm_provider": "bedrock,vertex_ai", "extra_headers": {"anthropic-beta": "test"}}),
        json!({}),
    )]
    #[case::scope_with_spaces(
        json!({"custom_llm_provider": "bedrock, anthropic , vertex_ai", "extra_headers": {"anthropic-beta": "test"}}),
        json!({"anthropic-beta": "test"}),
    )]
    #[case::scope_names_must_match_exactly(
        json!({"custom_llm_provider": "anthropic_text", "extra_headers": {"anthropic-beta": "test"}}),
        json!({}),
    )]
    #[case::entries_scope_independently(
        json!([
            {"custom_llm_provider": "anthropic,bedrock,vertex_ai", "extra_headers": {"anthropic-beta": "context-1m-2025-08-07"}},
            {"custom_llm_provider": "bedrock", "extra_headers": {"x-bedrock-only": "no"}},
            {"custom_llm_provider": "anthropic", "extra_headers": {"authorization": "Bearer sk-ant-oat01-fake-token"}}
        ]),
        json!({"anthropic-beta": "context-1m-2025-08-07", "authorization": "Bearer sk-ant-oat01-fake-token"}),
    )]
    #[case::later_entries_win(
        json!([
            {"custom_llm_provider": "anthropic", "extra_headers": {"x-scoped": "first"}},
            {"custom_llm_provider": "anthropic", "extra_headers": {"x-scoped": "second"}}
        ]),
        json!({"x-scoped": "second"}),
    )]
    #[case::empty_list(json!([]), json!({}))]
    #[case::entry_without_scope(json!({"extra_headers": {"x-scoped": "yes"}}), json!({}))]
    #[case::entry_without_headers(json!({"custom_llm_provider": "anthropic"}), json!({}))]
    fn provider_specific_headers_match_the_scoped_provider(
        #[case] configured: Value,
        #[case] expected: Value,
    ) {
        let configured: ProviderSpecificHeaders = serde_json::from_value(configured).unwrap();
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&configured),
                "anthropic",
                "claude-haiku-5-5",
                &Map::new()
            )),
            expected
        );
    }

    #[test]
    fn no_configured_headers_match_nothing() {
        assert_eq!(
            get_provider_specific_headers(None, "anthropic", "claude-haiku-5-5", &Map::new()),
            Map::new()
        );
    }

    #[rstest]
    #[case::claude_model("claude-sonnet-5-5", json!({"anthropic-user-profile-id": "uprof_pin"}))]
    #[case::claude_model_in_any_case("Claude-Sonnet-5-5", json!({"anthropic-user-profile-id": "uprof_pin"}))]
    #[case::non_claude_model("mistral-large-2411", json!({}))]
    fn claude_only_entries_skip_other_models(#[case] model: &str, #[case] expected: Value) {
        let configured: ProviderSpecificHeaders = serde_json::from_value(json!({
            "custom_llm_provider": "azure_ai",
            "extra_headers": {"anthropic-user-profile-id": "uprof_pin"},
            "claude_models_only": true
        }))
        .unwrap();
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&configured),
                "azure_ai",
                model,
                &Map::new()
            )),
            expected
        );
    }

    #[rstest]
    #[case::yielding_entry_keeps_the_configured_value(true, json!({"x-other": "client"}))]
    #[case::overriding_entry_replaces_it(
        false,
        json!({"Anthropic-User-Profile-Id": "uprof_client", "x-other": "client"})
    )]
    fn yielding_entries_never_replace_a_configured_header(
        #[case] yields_to_configured: bool,
        #[case] expected: Value,
    ) {
        let configured: ProviderSpecificHeaders = serde_json::from_value(json!({
            "custom_llm_provider": "anthropic",
            "extra_headers": {"Anthropic-User-Profile-Id": "uprof_client", "x-other": "client"},
            "yields_to_configured": yields_to_configured
        }))
        .unwrap();
        let deployment = json!({"anthropic-user-profile-id": "uprof_deployment"});
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&configured),
                "anthropic",
                "claude-haiku-5-5",
                deployment.as_object().unwrap()
            )),
            expected
        );
    }
}
