use litellm_types::utils::{ProviderSpecificHeader, ProviderSpecificHeaders};
use serde_json::{Map, Value};

pub fn get_provider_specific_headers(
    provider_specific_header: Option<&ProviderSpecificHeaders>,
    custom_llm_provider: &str,
) -> Map<String, Value> {
    let entries: &[ProviderSpecificHeader] = match provider_specific_header {
        None => &[],
        Some(ProviderSpecificHeaders::One(entry)) => std::slice::from_ref(entry),
        Some(ProviderSpecificHeaders::Many(entries)) => entries,
    };
    entries
        .iter()
        .filter(|entry| {
            entry
                .custom_llm_provider
                .split(',')
                .any(|scoped| scoped.trim() == custom_llm_provider)
        })
        .flat_map(|entry| entry.extra_headers.clone())
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
                "anthropic"
            )),
            expected
        );
    }

    #[test]
    fn no_configured_headers_match_nothing() {
        assert_eq!(get_provider_specific_headers(None, "anthropic"), Map::new());
    }
}
