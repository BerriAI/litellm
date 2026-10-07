use litellm_llms_types::headers::{ProviderSpecificHeader, ProviderSpecificHeaders};
use litellm_llms_types::recognized::Recognized;
use serde_json::{Map, Value};

pub fn get_provider_specific_headers(
    provider_specific_header: Option<&ProviderSpecificHeaders>,
    custom_llm_provider: Option<&str>,
) -> Map<String, Value> {
    let Some(custom_llm_provider) = custom_llm_provider else {
        return Map::new();
    };
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
        .map(|(key, value)| {
            (
                key,
                match value {
                    Recognized::Known(value) => Value::String(value),
                    Recognized::Unrecognized(value) => value,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
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
    #[case::non_string_values_are_not_dropped(
        json!({"custom_llm_provider":"anthropic","extra_headers":{"x-header":17,"x-null":null}}),
        json!({"x-header":17,"x-null":null}),
    )]
    fn provider_specific_headers_match_the_scoped_provider(
        #[case] configured: Value,
        #[case] expected: Value,
    ) {
        let configured: ProviderSpecificHeaders = serde_json::from_value(configured).unwrap();
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&configured),
                Some("anthropic")
            )),
            expected
        );
    }

    #[fixture]
    fn shared_headers() -> ProviderSpecificHeader {
        ProviderSpecificHeader {
            custom_llm_provider: "anthropic,bedrock,bedrock_converse,vertex_ai".into(),
            extra_headers: [(
                "anthropic-beta".into(),
                Recognized::Known("test-beta".into()),
            )]
            .into_iter()
            .collect(),
        }
    }

    #[rstest]
    #[case::anthropic("anthropic", json!({"anthropic-beta":"test-beta"}))]
    #[case::bedrock("bedrock", json!({"anthropic-beta":"test-beta"}))]
    #[case::bedrock_converse("bedrock_converse", json!({"anthropic-beta":"test-beta"}))]
    #[case::vertex_ai("vertex_ai", json!({"anthropic-beta":"test-beta"}))]
    #[case::unlisted("openai", json!({}))]
    #[case::empty_provider("", json!({}))]
    fn multi_provider_scope_matches_each_provider(
        shared_headers: ProviderSpecificHeader,
        #[case] provider: &str,
        #[case] expected: Value,
    ) {
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&ProviderSpecificHeaders::One(shared_headers)),
                Some(provider)
            )),
            expected
        );
    }

    #[rstest]
    #[case::anthropic("anthropic", json!({"anthropic-beta":"test-beta", "authorization":"Bearer test-token"}))]
    #[case::bedrock("bedrock", json!({"anthropic-beta":"test-beta"}))]
    #[case::unlisted("openai", json!({}))]
    fn each_entry_keeps_its_provider_scope(
        shared_headers: ProviderSpecificHeader,
        #[case] provider: &str,
        #[case] expected: Value,
    ) {
        let headers = ProviderSpecificHeaders::Many(vec![
            shared_headers,
            ProviderSpecificHeader {
                custom_llm_provider: "anthropic".into(),
                extra_headers: [(
                    "authorization".into(),
                    Recognized::Known("Bearer test-token".into()),
                )]
                .into_iter()
                .collect(),
            },
        ]);
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&headers),
                Some(provider)
            )),
            expected
        );
    }

    #[rstest]
    fn matching_provider_keeps_all_headers() {
        let configured = ProviderSpecificHeaders::One(ProviderSpecificHeader {
            custom_llm_provider: "openai".into(),
            extra_headers: [
                (
                    "Authorization".into(),
                    Recognized::Known("Bearer token123".into()),
                ),
                ("Custom-Header".into(), Recognized::Known("value".into())),
            ]
            .into_iter()
            .collect(),
        });
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&configured),
                Some("openai")
            )),
            json!({"Authorization":"Bearer token123", "Custom-Header":"value"})
        );
    }

    #[rstest]
    fn trims_bedrock_scope() {
        let configured = ProviderSpecificHeaders::One(ProviderSpecificHeader {
            custom_llm_provider: "anthropic, bedrock, vertex_ai".into(),
            extra_headers: [(
                "anthropic-beta".into(),
                Recognized::Known("test-beta".into()),
            )]
            .into_iter()
            .collect(),
        });
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&configured),
                Some("bedrock")
            )),
            json!({"anthropic-beta":"test-beta"})
        );
    }

    #[rstest]
    #[case::anthropic("anthropic")]
    #[case::openai("openai")]
    #[case::empty_provider("")]
    fn no_configured_headers_match_nothing(#[case] provider: &str) {
        assert_eq!(
            get_provider_specific_headers(None, Some(provider)),
            Map::new()
        );
    }

    #[rstest]
    #[case::scoped("anthropic")]
    #[case::empty_scope("")]
    fn none_requested_provider_matches_nothing(#[case] scope: &str) {
        let configured = ProviderSpecificHeaders::One(ProviderSpecificHeader {
            custom_llm_provider: scope.into(),
            extra_headers: [("x-scoped".into(), Recognized::Known("value".into()))]
                .into_iter()
                .collect(),
        });
        assert_eq!(
            get_provider_specific_headers(Some(&configured), None),
            Map::new()
        );
    }
}
