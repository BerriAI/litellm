use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;
use serde_json::Value;

use crate::{
    anthropic::{
        ANTHROPIC_OAUTH_TOKEN_PREFIX,
        common_utils::{
            ANTHROPIC_OAUTH_BETA_HEADER, beta, has_advisor_tool, is_anthropic_oauth_key,
            is_tool_search_used, join_beta_values, requires_native_compaction_beta,
            split_beta_values,
        },
    },
    base_llm::anthropic_messages::transformation::Headers,
};

const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";
const BETA_HEADER: &str = "anthropic-beta";
const AUTHORIZATION: &str = "authorization";
const API_KEY_HEADER: &str = "x-api-key";
const DIRECT_BROWSER_ACCESS_HEADER: &str = "anthropic-dangerous-direct-browser-access";

fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

fn without(headers: Headers, names: &[&str]) -> Headers {
    headers
        .into_iter()
        .filter(|(header, _)| !names.iter().any(|name| header.eq_ignore_ascii_case(name)))
        .collect()
}

fn existing_betas(headers: &[(String, String)]) -> impl Iterator<Item = String> + '_ {
    headers
        .iter()
        .filter(|(header, _)| header.eq_ignore_ascii_case(BETA_HEADER))
        .flat_map(|(_, value)| split_beta_values(Some(value)))
}

fn with_oauth_bearer(headers: Headers, bearer: String) -> Headers {
    let beta =
        join_beta_values(existing_betas(&headers).chain([ANTHROPIC_OAUTH_BETA_HEADER.to_string()]));
    without(headers, &[API_KEY_HEADER, AUTHORIZATION, BETA_HEADER])
        .into_iter()
        .chain([
            (AUTHORIZATION.to_string(), bearer),
            (BETA_HEADER.to_string(), beta),
            (DIRECT_BROWSER_ACCESS_HEADER.to_string(), "true".to_string()),
        ])
        .collect()
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

pub fn authenticate(
    headers: Headers,
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Headers, litellm_auth::Error> {
    if let Some(forwarded) = header_value(&headers, AUTHORIZATION)
        && forwarded
            .strip_prefix("Bearer ")
            .is_some_and(|token| token.starts_with(ANTHROPIC_OAUTH_TOKEN_PREFIX))
    {
        let bearer = forwarded.to_string();
        return Ok(with_oauth_bearer(headers, bearer));
    }
    if let Some(key) = api_key.filter(|key| key.starts_with(ANTHROPIC_OAUTH_TOKEN_PREFIX)) {
        return Ok(with_oauth_bearer(headers, format!("Bearer {key}")));
    }
    if header_value(&headers, API_KEY_HEADER).is_some()
        || header_value(&headers, AUTHORIZATION).is_some()
    {
        return Ok(headers);
    }
    let resolved_key = non_empty(api_key)
        .map(str::to_string)
        .or_else(|| env_lookup(ANTHROPIC_API_KEY_ENV).filter(|value| !value.trim().is_empty()));
    let auth = match resolved_key {
        Some(key) if is_anthropic_oauth_key(&key) => {
            (AUTHORIZATION.to_string(), format!("Bearer {key}"))
        }
        Some(key) => (API_KEY_HEADER.to_string(), key),
        None => match env_lookup(ANTHROPIC_AUTH_TOKEN_ENV).filter(|value| !value.trim().is_empty())
        {
            Some(token) => (AUTHORIZATION.to_string(), format!("Bearer {token}")),
            None => {
                return Err(litellm_auth::Error::MissingApiKey {
                    provider: "Anthropic",
                    environment_variable: ANTHROPIC_API_KEY_ENV,
                });
            }
        },
    };
    Ok(headers.into_iter().chain([auth]).collect())
}

fn context_management_betas(
    context_management: Option<&Value>,
) -> impl Iterator<Item = &'static str> {
    let edits = context_management
        .and_then(|value| value.get("edits"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let (compact, other) = edits.iter().fold((false, false), |(compact, other), edit| {
        match edit.get("type").and_then(Value::as_str) {
            Some("compact_20260112") => (true, other),
            _ => (compact, true),
        }
    });
    compact
        .then_some(beta::COMPACT_2026_01_12)
        .into_iter()
        .chain(other.then_some(beta::CONTEXT_MANAGEMENT_2025_06_27))
}

fn uses_structured_output(request: &AnthropicMessagesRequest) -> bool {
    request.output_format.is_some()
        || request
            .output_config
            .as_ref()
            .and_then(|config| config.get("format"))
            .is_some_and(|format| !format.is_null())
}

fn messages_carry_output_config(request: &AnthropicMessagesRequest) -> bool {
    request
        .messages
        .iter()
        .any(|message| message.extra.contains_key("output_config"))
}

pub fn feature_betas(request: &AnthropicMessagesRequest) -> Vec<&'static str> {
    let tools = request.tools.as_deref();
    [
        requires_native_compaction_beta(request.compaction.as_ref(), &request.messages)
            .then_some(beta::COMPACT_2026_09_04),
        uses_structured_output(request).then_some(beta::STRUCTURED_OUTPUT),
        (request.speed.as_deref() == Some("fast")).then_some(beta::FAST_MODE_2026_02_01),
        messages_carry_output_config(request).then_some(beta::PER_TURN_CONTROL_2026_07_01),
        has_advisor_tool(tools).then_some(beta::ADVISOR_TOOL_2026_03_01),
        is_tool_search_used(tools).then_some(beta::ADVANCED_TOOL_USE_2025_11_20),
    ]
    .into_iter()
    .flatten()
    .chain(context_management_betas(
        request.context_management.as_ref(),
    ))
    .collect()
}

pub fn with_feature_betas(headers: Headers, request: &AnthropicMessagesRequest) -> Headers {
    let existing = existing_betas(&headers).collect::<Vec<_>>();
    let features = feature_betas(request);
    if existing.is_empty() && features.is_empty() {
        return headers;
    }
    let merged = join_beta_values(
        existing
            .into_iter()
            .chain(features.into_iter().map(str::to_string)),
    );
    without(headers, &[BETA_HEADER])
        .into_iter()
        .chain([(BETA_HEADER.to_string(), merged)])
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;

    const OAUTH_TOKEN: &str = "sk-ant-oat01-token";
    const OAUTH_BEARER: &str = "Bearer sk-ant-oat01-token";
    const REGULAR_KEY: &str = "sk-ant-api03-regular";
    const BROWSER_ACCESS: (&str, &str) = ("anthropic-dangerous-direct-browser-access", "true");

    type Env = &'static [(&'static str, &'static str)];

    fn request(fields: Value) -> AnthropicMessagesRequest {
        let mut body =
            json!({"model": "claude", "messages": [{"role": "user", "content": "Hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(body).unwrap()
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn betas(values: &[&str]) -> String {
        values.join(",")
    }

    #[fixture]
    fn no_env() -> Env {
        &[]
    }

    #[fixture]
    fn full_env() -> Env {
        &[
            ("ANTHROPIC_API_KEY", "sk-env"),
            ("ANTHROPIC_AUTH_TOKEN", "env-token"),
        ]
    }

    fn authenticate_with(
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        env: Env,
    ) -> Result<Headers, litellm_auth::Error> {
        let lookup = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        authenticate(headers(forwarded), api_key, &lookup)
    }

    #[rstest]
    #[case::forwarded_bearer_drops_forwarded_and_deployment_keys(
        &[("X-Api-Key", REGULAR_KEY), ("Authorization", OAUTH_BEARER)],
        Some(REGULAR_KEY),
        OAUTH_BEARER,
        &[],
    )]
    #[case::forwarded_bearer_in_uppercase_authorization_header(
        &[("AUTHORIZATION", OAUTH_BEARER)],
        None,
        OAUTH_BEARER,
        &[],
    )]
    #[case::forwarded_bearer_keeps_unrelated_headers_in_place(
        &[("anthropic-version", "2023-06-01"), ("authorization", OAUTH_BEARER)],
        None,
        OAUTH_BEARER,
        &[("anthropic-version", "2023-06-01")],
    )]
    #[case::forwarded_bearer_wins_over_an_oauth_api_key(
        &[("authorization", OAUTH_BEARER)],
        Some("sk-ant-oat01-deployment"),
        OAUTH_BEARER,
        &[],
    )]
    #[case::api_key_authenticates_as_a_bearer(&[], Some(OAUTH_TOKEN), OAUTH_BEARER, &[])]
    #[case::api_key_removes_a_forwarded_x_api_key(
        &[("x-api-key", OAUTH_TOKEN)],
        Some(OAUTH_TOKEN),
        OAUTH_BEARER,
        &[],
    )]
    #[case::api_key_replaces_a_forwarded_non_oauth_bearer(
        &[("Authorization", "Bearer some-proxy-token")],
        Some(OAUTH_TOKEN),
        OAUTH_BEARER,
        &[],
    )]
    fn oauth_token_is_the_whole_credential(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        #[case] expected_bearer: &str,
        #[case] kept: &[(&str, &str)],
        full_env: Env,
    ) {
        let expected = kept
            .iter()
            .copied()
            .chain([
                ("authorization", expected_bearer),
                ("anthropic-beta", ANTHROPIC_OAUTH_BETA_HEADER),
                BROWSER_ACCESS,
            ])
            .collect::<Vec<_>>();
        assert_eq!(
            authenticate_with(forwarded, api_key, full_env).unwrap(),
            headers(&expected)
        );
    }

    #[rstest]
    #[case::forwarded_bearer_merges_a_differently_cased_beta_header(
        &[("Anthropic-Beta", "web-search-2025-03-05"), ("authorization", OAUTH_BEARER)],
        None,
    )]
    #[case::forwarded_bearer_dedupes_an_existing_oauth_beta(
        &[("anthropic-beta", "web-search-2025-03-05, oauth-2025-04-20"), ("authorization", OAUTH_BEARER)],
        None,
    )]
    #[case::api_key_merges_the_existing_beta_header(
        &[("anthropic-beta", " web-search-2025-03-05 ,")],
        Some(OAUTH_TOKEN),
    )]
    #[case::forwarded_bearer_unions_every_beta_header_casing(
        &[("anthropic-beta", "oauth-2025-04-20"), ("ANTHROPIC-BETA", "web-search-2025-03-05"), ("authorization", OAUTH_BEARER)],
        None,
    )]
    fn oauth_beta_merges_into_existing_betas(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        no_env: Env,
    ) {
        assert_eq!(
            authenticate_with(forwarded, api_key, no_env).unwrap(),
            headers(&[
                ("authorization", OAUTH_BEARER),
                (
                    "anthropic-beta",
                    &betas(&[ANTHROPIC_OAUTH_BETA_HEADER, "web-search-2025-03-05"])
                ),
                BROWSER_ACCESS,
            ])
        );
    }

    #[rstest]
    #[case::x_api_key_over_the_deployment_key(&[("x-api-key", "caller-key")], Some("sk-other"))]
    #[case::uppercase_x_api_key(&[("X-API-KEY", "caller-key")], None)]
    #[case::non_oauth_bearer(&[("Authorization", "Bearer some-proxy-token")], None)]
    #[case::non_oauth_bearer_over_a_regular_api_key(
        &[("authorization", "Bearer sk-ant-api03-forwarded")],
        Some(REGULAR_KEY),
    )]
    #[case::oauth_token_without_the_bearer_scheme(&[("authorization", OAUTH_TOKEN)], None)]
    #[case::oauth_token_behind_a_lowercase_bearer_scheme(
        &[("authorization", "bearer sk-ant-oat01-token")],
        None,
    )]
    fn forwarded_auth_header_is_kept_untouched(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        full_env: Env,
    ) {
        assert_eq!(
            authenticate_with(forwarded, api_key, full_env).unwrap(),
            headers(forwarded)
        );
    }

    #[rstest]
    #[case::api_key_param(Some("sk-param"), &[], ("x-api-key", "sk-param"))]
    #[case::api_key_param_over_env_key_and_auth_token(
        Some("sk-param"),
        &[("ANTHROPIC_API_KEY", "sk-env"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        ("x-api-key", "sk-param"),
    )]
    #[case::env_key_without_a_param(None, &[("ANTHROPIC_API_KEY", "sk-env")], ("x-api-key", "sk-env"))]
    #[case::env_key_when_the_param_is_empty(Some(""), &[("ANTHROPIC_API_KEY", "sk-env")], ("x-api-key", "sk-env"))]
    #[case::env_key_when_the_param_is_whitespace(
        Some("  "),
        &[("ANTHROPIC_API_KEY", "sk-env")],
        ("x-api-key", "sk-env"),
    )]
    #[case::env_key_over_auth_token(
        None,
        &[("ANTHROPIC_API_KEY", "sk-env"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        ("x-api-key", "sk-env"),
    )]
    #[case::auth_token_as_a_bearer(
        None,
        &[("ANTHROPIC_AUTH_TOKEN", "env-token")],
        ("authorization", "Bearer env-token"),
    )]
    #[case::auth_token_when_the_env_key_is_whitespace(
        None,
        &[("ANTHROPIC_API_KEY", " \t"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        ("authorization", "Bearer env-token"),
    )]
    #[case::oauth_env_key_as_a_plain_bearer(
        None,
        &[("ANTHROPIC_API_KEY", "sk-ant-oat01-env")],
        ("authorization", "Bearer sk-ant-oat01-env"),
    )]
    fn credential_is_resolved_after_the_existing_headers(
        #[case] api_key: Option<&str>,
        #[case] env: Env,
        #[case] expected: (&str, &str),
    ) {
        let forwarded = [("anthropic-beta", "web-search-2025-03-05")];
        assert_eq!(
            authenticate_with(&forwarded, api_key, env).unwrap(),
            headers(&[forwarded[0], expected])
        );
    }

    #[rstest]
    #[case::no_credentials(&[], None, &[])]
    #[case::empty_api_key(&[], Some(""), &[])]
    #[case::whitespace_only_env_values(
        &[],
        None,
        &[("ANTHROPIC_API_KEY", "  "), ("ANTHROPIC_AUTH_TOKEN", " \t")],
    )]
    #[case::unrelated_forwarded_headers(&[("anthropic-beta", "web-search-2025-03-05")], None, &[])]
    fn missing_credentials_are_an_auth_error(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        #[case] env: Env,
    ) {
        assert!(matches!(
            authenticate_with(forwarded, api_key, env),
            Err(litellm_auth::Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: "ANTHROPIC_API_KEY",
            })
        ));
    }

    #[rstest]
    #[case::no_features(json!({}), &[])]
    #[case::output_format(json!({"output_format": {"type": "json_schema"}}), &[beta::STRUCTURED_OUTPUT])]
    #[case::null_output_format(json!({"output_format": null}), &[])]
    #[case::output_config_format(
        json!({"output_config": {"format": {"type": "json_schema"}, "effort": "xhigh"}}),
        &[beta::STRUCTURED_OUTPUT]
    )]
    #[case::null_output_config_format(json!({"output_config": {"format": null}}), &[])]
    #[case::top_level_output_config_without_format(json!({"output_config": {"effort": "high"}}), &[])]
    #[case::fast_speed(json!({"speed": "fast"}), &[beta::FAST_MODE_2026_02_01])]
    #[case::standard_speed(json!({"speed": "standard"}), &[])]
    #[case::compaction_param(json!({"compaction": {"enabled": true}}), &[beta::COMPACT_2026_09_04])]
    #[case::empty_compaction_param(json!({"compaction": {}}), &[beta::COMPACT_2026_09_04])]
    #[case::signed_compaction_block_in_history(
        json!({"messages": [
            {"role": "assistant", "content": [{"type": "compaction", "content": "summary", "signature": "sig"}]},
            {"role": "user", "content": "Continue"},
        ]}),
        &[beta::COMPACT_2026_09_04]
    )]
    #[case::unsigned_compaction_block_in_history(
        json!({"messages": [
            {"role": "assistant", "content": [{"type": "compaction", "content": "summary", "signature": ""}]},
            {"role": "user", "content": "Continue"},
        ]}),
        &[]
    )]
    #[case::advisor_tool(
        json!({"tools": [{"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-6"}]}),
        &[beta::ADVISOR_TOOL_2026_03_01]
    )]
    #[case::no_tools(json!({"tools": []}), &[])]
    #[case::regex_tool_search(
        json!({"tools": [{"type": "tool_search_tool_regex_20251119"}]}),
        &[beta::ADVANCED_TOOL_USE_2025_11_20]
    )]
    #[case::bm25_tool_search(
        json!({"tools": [{"type": "tool_search_tool_bm25_20251119"}]}),
        &[beta::ADVANCED_TOOL_USE_2025_11_20]
    )]
    #[case::unrelated_server_tool(json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}), &[])]
    #[case::only_compact_edits(
        json!({"context_management": {"edits": [{"type": "compact_20260112"}]}}),
        &[beta::COMPACT_2026_01_12]
    )]
    #[case::only_other_edits(
        json!({"context_management": {"edits": [{"type": "clear_tool_uses_20250919", "keep": {"type": "tool_uses", "value": 3}}]}}),
        &[beta::CONTEXT_MANAGEMENT_2025_06_27]
    )]
    #[case::compact_and_other_edits(
        json!({"context_management": {"edits": [{"type": "compact_20260112"}, {"type": "clear_tool_uses_20250919"}]}}),
        &[beta::COMPACT_2026_01_12, beta::CONTEXT_MANAGEMENT_2025_06_27]
    )]
    #[case::edit_without_a_type(json!({"context_management": {"edits": [{}]}}), &[beta::CONTEXT_MANAGEMENT_2025_06_27])]
    #[case::empty_edits(json!({"context_management": {"edits": []}}), &[])]
    #[case::context_management_without_edits(json!({"context_management": {}}), &[])]
    #[case::per_message_output_config(
        json!({"messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}]}),
        &[beta::PER_TURN_CONTROL_2026_07_01]
    )]
    #[case::per_message_null_output_config(
        json!({"messages": [{"role": "user", "content": "hi", "output_config": null}]}),
        &[beta::PER_TURN_CONTROL_2026_07_01]
    )]
    fn feature_betas_follow_the_request(#[case] fields: Value, #[case] expected: &[&str]) {
        assert_eq!(feature_betas(&request(fields)), expected);
    }

    #[rstest]
    #[case::no_betas(&[("x-api-key", "k"), ("anthropic-version", "2023-06-01")], json!({}))]
    #[case::blank_beta_header(&[("Anthropic-Beta", " , "), ("x-api-key", "k")], json!({}))]
    fn headers_without_any_beta_value_are_untouched(
        #[case] input: &[(&str, &str)],
        #[case] fields: Value,
    ) {
        assert_eq!(
            with_feature_betas(headers(input), &request(fields)),
            headers(input)
        );
    }

    #[rstest]
    #[case::feature_beta_is_appended(
        &[("x-api-key", "k")],
        json!({"speed": "fast"}),
        &[("x-api-key", "k"), ("anthropic-beta", beta::FAST_MODE_2026_02_01)],
    )]
    #[case::existing_betas_are_normalized_without_features(
        &[("Anthropic-Beta", "web-search-2025-03-05, interleaved-thinking-2025-05-14 ,web-search-2025-03-05"), ("x-api-key", "k")],
        json!({}),
        &[("x-api-key", "k"), ("anthropic-beta", "interleaved-thinking-2025-05-14,web-search-2025-03-05")],
    )]
    #[case::existing_advisor_beta_is_kept_without_an_advisor_tool(
        &[("anthropic-beta", beta::ADVISOR_TOOL_2026_03_01)],
        json!({"tools": []}),
        &[("anthropic-beta", beta::ADVISOR_TOOL_2026_03_01)],
    )]
    #[case::feature_already_sent_is_not_duplicated(
        &[("anthropic-beta", beta::FAST_MODE_2026_02_01)],
        json!({"speed": "fast"}),
        &[("anthropic-beta", beta::FAST_MODE_2026_02_01)],
    )]
    fn feature_betas_merge_into_the_headers(
        #[case] input: &[(&str, &str)],
        #[case] fields: Value,
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(
            with_feature_betas(headers(input), &request(fields)),
            headers(expected)
        );
    }

    #[test]
    fn differently_cased_beta_header_is_replaced_by_one_sorted_header() {
        let merged = with_feature_betas(
            headers(&[("Anthropic-Beta", "interleaved-thinking-2025-05-14")]),
            &request(
                json!({"messages": [{"role": "system", "content": "env", "output_config": {"effort": "low"}}]}),
            ),
        );
        assert_eq!(
            merged,
            headers(&[(
                "anthropic-beta",
                &betas(&[
                    "interleaved-thinking-2025-05-14",
                    beta::PER_TURN_CONTROL_2026_07_01
                ])
            )])
        );
    }

    #[test]
    fn every_beta_header_casing_is_unioned_into_one_header() {
        let merged = with_feature_betas(
            headers(&[
                ("anthropic-beta", "interleaved-thinking-2025-05-14"),
                ("Anthropic-Beta", "web-search-2025-03-05"),
            ]),
            &request(json!({"speed": "fast"})),
        );
        assert_eq!(
            merged,
            headers(&[(
                "anthropic-beta",
                &betas(&[
                    beta::FAST_MODE_2026_02_01,
                    "interleaved-thinking-2025-05-14",
                    "web-search-2025-03-05"
                ])
            )])
        );
    }

    #[test]
    fn unknown_client_betas_survive_alongside_the_added_one() {
        let client_betas = [
            "claude-code-20250219",
            "interleaved-thinking-2025-05-14",
            beta::CONTEXT_MANAGEMENT_2025_06_27,
            beta::PER_TURN_CONTROL_2026_07_01,
            "effort-2025-11-24",
        ];
        let merged = with_feature_betas(
            headers(&[("anthropic-beta", &betas(&client_betas))]),
            &request(
                json!({"messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}]}),
            ),
        );
        assert_eq!(
            merged,
            headers(&[(
                "anthropic-beta",
                &betas(&[
                    "claude-code-20250219",
                    beta::CONTEXT_MANAGEMENT_2025_06_27,
                    "effort-2025-11-24",
                    "interleaved-thinking-2025-05-14",
                    beta::PER_TURN_CONTROL_2026_07_01,
                ])
            )])
        );
    }

    #[test]
    fn every_feature_merges_with_the_oauth_beta_sorted_and_last() {
        let oauth_headers = authenticate_with(&[], Some(OAUTH_TOKEN), &[]).unwrap();
        let all_features = request(json!({
            "compaction": {"enabled": true},
            "output_format": {"type": "json_schema"},
            "speed": "fast",
            "tools": [{"type": "advisor_20260301"}, {"type": "tool_search_tool_bm25_20251119"}],
            "context_management": {"edits": [{"type": "compact_20260112"}, {"type": "clear_thinking_20251015"}]},
            "messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}],
        }));
        assert_eq!(
            with_feature_betas(oauth_headers, &all_features),
            headers(&[
                ("authorization", OAUTH_BEARER),
                BROWSER_ACCESS,
                (
                    "anthropic-beta",
                    &betas(&[
                        beta::ADVANCED_TOOL_USE_2025_11_20,
                        beta::ADVISOR_TOOL_2026_03_01,
                        beta::COMPACT_2026_01_12,
                        beta::COMPACT_2026_09_04,
                        beta::CONTEXT_MANAGEMENT_2025_06_27,
                        beta::FAST_MODE_2026_02_01,
                        ANTHROPIC_OAUTH_BETA_HEADER,
                        beta::PER_TURN_CONTROL_2026_07_01,
                        beta::STRUCTURED_OUTPUT,
                    ])
                ),
            ])
        );
    }
}
