use std::collections::{HashMap, HashSet};

use litellm_llms_types::headers::{ProviderSpecificHeader, ProviderSpecificHeaders};
use serde_json::{Map, Value};

pub fn get_provider_specific_headers(
    provider_specific_header: Option<&ProviderSpecificHeaders>,
    custom_llm_provider: &str,
    takes_user_profile_id: bool,
    configured_headers: &Map<String, Value>,
) -> Map<String, Value> {
    let entries: &[ProviderSpecificHeader] = match provider_specific_header {
        None => &[],
        Some(ProviderSpecificHeaders::One(entry)) => std::slice::from_ref(entry),
        Some(ProviderSpecificHeaders::Many(entries)) => entries,
    };
    let configured_names: HashSet<String> = configured_headers
        .keys()
        .map(|name| name.to_ascii_lowercase())
        .collect();
    let applicable: Vec<&ProviderSpecificHeader> = entries
        .iter()
        .filter(|entry| applies_to_call(entry, custom_llm_provider, takes_user_profile_id))
        .collect();
    let replaced_names: HashSet<String> = applicable
        .iter()
        .filter(|entry| entry.configured_credential_headers.is_empty())
        .flat_map(|entry| {
            contributed_headers(entry, &configured_names)
                .into_iter()
                .map(|(name, _)| name)
        })
        .collect();
    let kept_configured = sorted_text_values_by_name(
        configured_headers
            .iter()
            .filter(|(name, _)| !replaced_names.contains(*name)),
    );
    applicable
        .into_iter()
        .filter(|entry| configures_a_listed_credential(entry, &kept_configured))
        .flat_map(|entry| contributed_headers(entry, &configured_names))
        .collect()
}

fn applies_to_call(
    entry: &ProviderSpecificHeader,
    custom_llm_provider: &str,
    takes_user_profile_id: bool,
) -> bool {
    (!entry.user_profile_id_models_only || takes_user_profile_id)
        && (!entry.non_user_profile_id_models_only || !takes_user_profile_id)
        && entry
            .custom_llm_provider
            .split(',')
            .any(|scoped| scoped.trim() == custom_llm_provider)
}

fn sorted_text_values_by_name<'a>(
    headers: impl Iterator<Item = (&'a String, &'a Value)>,
) -> HashMap<String, Vec<&'a str>> {
    let mut values_by_name: HashMap<String, Vec<&'a str>> = HashMap::new();
    for (name, value) in headers {
        values_by_name
            .entry(name.to_ascii_lowercase())
            .or_default()
            .push(value.as_str().unwrap_or(""));
    }
    values_by_name
        .values_mut()
        .for_each(|values| values.sort_unstable());
    values_by_name
}

fn carries(values_by_name: &HashMap<String, Vec<&str>>, name: &str, prefix: &str) -> bool {
    values_by_name.get(name).is_some_and(|values| {
        values
            .get(values.partition_point(|value| *value < prefix))
            .is_some_and(|value| value.starts_with(prefix))
    })
}

fn configures_a_listed_credential(
    entry: &ProviderSpecificHeader,
    configured_values: &HashMap<String, Vec<&str>>,
) -> bool {
    entry.configured_credential_headers.is_empty()
        || entry
            .configured_credential_headers
            .iter()
            .any(|credential| {
                carries(
                    configured_values,
                    &credential.to_ascii_lowercase(),
                    &entry.configured_credential_prefix,
                )
            })
}

fn contributed_headers(
    entry: &ProviderSpecificHeader,
    configured_names: &HashSet<String>,
) -> Map<String, Value> {
    entry
        .extra_headers
        .iter()
        .filter(|(name, _)| {
            !entry.yields_to_configured || !configured_names.contains(&name.to_ascii_lowercase())
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
                true,
                &Map::new()
            )),
            expected
        );
    }

    #[rstest]
    fn no_configured_headers_match_nothing() {
        assert_eq!(
            get_provider_specific_headers(None, "anthropic", true, &Map::new()),
            Map::new()
        );
    }

    #[rstest]
    #[case::model_takes_the_profile_id(true, json!({"anthropic-user-profile-id": "uprof_pin"}))]
    #[case::model_without_profile_id_support(false, json!({}))]
    fn profile_id_entries_apply_only_to_models_that_take_it(
        #[case] takes_user_profile_id: bool,
        #[case] expected: Value,
    ) {
        let configured: ProviderSpecificHeaders = serde_json::from_value(json!({
            "custom_llm_provider": "azure_ai",
            "extra_headers": {"anthropic-user-profile-id": "uprof_pin"},
            "user_profile_id_models_only": true
        }))
        .unwrap();
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&configured),
                "azure_ai",
                takes_user_profile_id,
                &Map::new()
            )),
            expected
        );
    }

    #[rstest]
    #[case::model_takes_the_profile_id(true, json!({}))]
    #[case::model_without_profile_id_support(false, json!({"authorization": "Bearer caller"}))]
    fn non_profile_id_entries_skip_models_that_take_it(
        #[case] takes_user_profile_id: bool,
        #[case] expected: Value,
    ) {
        let configured: ProviderSpecificHeaders = serde_json::from_value(json!({
            "custom_llm_provider": "azure_ai",
            "extra_headers": {"authorization": "Bearer caller"},
            "non_user_profile_id_models_only": true
        }))
        .unwrap();
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&configured),
                "azure_ai",
                takes_user_profile_id,
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
                true,
                deployment.as_object().unwrap()
            )),
            expected
        );
    }

    #[rstest]
    #[case::deployment_api_key(json!({"X-Api-Key": "sk-deployment"}), json!({"anthropic-user-profile-id": "uprof_pinned"}))]
    #[case::deployment_bearer(json!({"authorization": "Bearer deployment"}), json!({"anthropic-user-profile-id": "uprof_pinned"}))]
    #[case::unlisted_deployment_credential(json!({"api-key": "deployment"}), json!({}))]
    #[case::no_deployment_credential(json!({"x-other": "deployment"}), json!({}))]
    fn configured_credential_entries_apply_only_beside_a_listed_credential(
        #[case] configured: Value,
        #[case] expected: Value,
    ) {
        let pinned: ProviderSpecificHeaders = serde_json::from_value(json!({
            "custom_llm_provider": "anthropic",
            "extra_headers": {"anthropic-user-profile-id": "uprof_pinned"},
            "configured_credential_headers": ["authorization", "x-api-key"]
        }))
        .unwrap();
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&pinned),
                "anthropic",
                true,
                configured.as_object().unwrap()
            )),
            expected
        );
    }

    #[rstest]
    #[case::configured_oauth_token(json!({"Authorization": "Bearer sk-ant-oat01-deployment"}), json!({"anthropic-user-profile-id": "uprof_pinned"}))]
    #[case::configured_api_key_bearer(json!({"authorization": "Bearer sk-ant-deployment"}), json!({}))]
    #[case::prefix_on_an_unlisted_header(json!({"x-api-key": "Bearer sk-ant-oat01-deployment"}), json!({}))]
    fn a_configured_credential_prefix_applies_only_beside_a_matching_value(
        #[case] configured: Value,
        #[case] expected: Value,
    ) {
        let pinned: ProviderSpecificHeaders = serde_json::from_value(json!({
            "custom_llm_provider": "anthropic",
            "extra_headers": {"anthropic-user-profile-id": "uprof_pinned"},
            "configured_credential_headers": ["authorization"],
            "configured_credential_prefix": "Bearer sk-ant-oat"
        }))
        .unwrap();
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&pinned),
                "anthropic",
                true,
                configured.as_object().unwrap()
            )),
            expected
        );
    }

    #[rstest]
    #[case::caller_replaces_the_configured_token(
        json!({"authorization": "Bearer sk-ant-oat01-deployment"}),
        json!({"authorization": "Bearer sk-ant-oat01-caller"})
    )]
    #[case::configured_token_kept_under_another_name(
        json!({"Authorization": "Bearer sk-ant-oat01-deployment"}),
        json!({"authorization": "Bearer sk-ant-oat01-caller", "anthropic-user-profile-id": "uprof_pinned"})
    )]
    fn a_pin_applies_only_while_the_configured_credential_survives_the_merge(
        #[case] configured: Value,
        #[case] expected: Value,
    ) {
        let entries: ProviderSpecificHeaders = serde_json::from_value(json!([
            {
                "custom_llm_provider": "anthropic",
                "extra_headers": {"authorization": "Bearer sk-ant-oat01-caller"}
            },
            {
                "custom_llm_provider": "anthropic",
                "extra_headers": {"anthropic-user-profile-id": "uprof_pinned"},
                "configured_credential_headers": ["authorization"],
                "configured_credential_prefix": "Bearer sk-ant-oat"
            }
        ]))
        .unwrap();
        assert_eq!(
            Value::Object(get_provider_specific_headers(
                Some(&entries),
                "anthropic",
                true,
                configured.as_object().unwrap()
            )),
            expected
        );
    }
}
