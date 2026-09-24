//! Request headers for the direct Anthropic Messages API: credential resolution, including
//! OAuth tokens, and the `anthropic-beta` values a request's features call for.

use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;
use serde_json::Value;

use crate::anthropic::{
    ANTHROPIC_OAUTH_TOKEN_PREFIX,
    common_utils::{
        ANTHROPIC_OAUTH_BETA_HEADER, beta, has_advisor_tool, is_anthropic_oauth_key,
        is_tool_search_used, join_beta_values, requires_native_compaction_beta, split_beta_values,
    },
};

const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";
const BETA_HEADER: &str = "anthropic-beta";
const AUTHORIZATION: &str = "authorization";
const API_KEY_HEADER: &str = "x-api-key";
const DIRECT_BROWSER_ACCESS_HEADER: &str = "anthropic-dangerous-direct-browser-access";

pub type Headers = Vec<(String, String)>;

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

fn with_oauth_bearer(headers: Headers, bearer: String) -> Headers {
    let beta = merge_oauth_beta(header_value(&headers, BETA_HEADER));
    without(headers, &[API_KEY_HEADER, AUTHORIZATION, BETA_HEADER])
        .into_iter()
        .chain([
            (AUTHORIZATION.to_string(), bearer),
            (BETA_HEADER.to_string(), beta),
            (DIRECT_BROWSER_ACCESS_HEADER.to_string(), "true".to_string()),
        ])
        .collect()
}

fn merge_oauth_beta(existing: Option<&str>) -> String {
    join_beta_values(
        split_beta_values(existing).chain(std::iter::once(ANTHROPIC_OAUTH_BETA_HEADER.to_string())),
    )
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

/// Python's `validate_anthropic_messages_environment` credential steps: an OAuth token in
/// the forwarded `authorization` header or in `api_key` is the whole credential; otherwise a
/// forwarded auth header is kept, else the key (`ANTHROPIC_API_KEY`) or the bearer token
/// (`ANTHROPIC_AUTH_TOKEN`) is resolved.
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

/// The `anthropic-beta` values the request's features need, as Python's
/// `_update_headers_with_anthropic_beta` derives them for the direct API.
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

/// Merge the request's feature betas into the outgoing headers. Headers without any beta
/// value are returned untouched.
pub fn with_feature_betas(headers: Headers, request: &AnthropicMessagesRequest) -> Headers {
    let existing = split_beta_values(header_value(&headers, BETA_HEADER)).collect::<Vec<_>>();
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
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn request(fields: Value) -> AnthropicMessagesRequest {
        let mut body =
            json!({"model": "claude", "messages": [{"role": "user", "content": "Hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(body).unwrap()
    }

    fn header(name: &str, value: &str) -> (String, String) {
        (name.to_string(), value.to_string())
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn forwarded_oauth_bearer_replaces_the_key_and_adds_oauth_headers() {
        let headers = authenticate(
            vec![
                header("X-Api-Key", "sk-ant-api03-deployment"),
                header("Authorization", "Bearer sk-ant-oat01-token"),
                header("anthropic-beta", "web-search-2025-03-05"),
            ],
            Some("sk-ant-api03-deployment"),
            &no_env,
        )
        .unwrap();
        assert_eq!(header_value(&headers, "x-api-key"), None);
        assert_eq!(
            header_value(&headers, "authorization"),
            Some("Bearer sk-ant-oat01-token")
        );
        assert_eq!(
            header_value(&headers, "anthropic-beta"),
            Some("oauth-2025-04-20,web-search-2025-03-05")
        );
        assert_eq!(
            header_value(&headers, DIRECT_BROWSER_ACCESS_HEADER),
            Some("true")
        );
    }

    #[test]
    fn oauth_api_key_authenticates_as_a_bearer() {
        let headers = authenticate(vec![], Some("sk-ant-oat01-token"), &no_env).unwrap();
        assert_eq!(
            header_value(&headers, "authorization"),
            Some("Bearer sk-ant-oat01-token")
        );
        assert_eq!(
            header_value(&headers, "anthropic-beta"),
            Some("oauth-2025-04-20")
        );
        assert_eq!(header_value(&headers, "x-api-key"), None);
    }

    #[test]
    fn forwarded_auth_headers_are_kept_without_resolving_a_key() {
        let forwarded = vec![header("Authorization", "Bearer some-proxy-token")];
        let headers = authenticate(forwarded.clone(), None, &no_env).unwrap();
        assert_eq!(headers, forwarded);
        let keyed = vec![header("x-api-key", "caller-key")];
        assert_eq!(
            authenticate(keyed.clone(), Some("sk-other"), &no_env).unwrap(),
            keyed
        );
    }

    #[rstest]
    #[case(Some("sk-param"), &[], "x-api-key", "sk-param")]
    #[case(Some("  "), &[("ANTHROPIC_API_KEY", "sk-env")], "x-api-key", "sk-env")]
    #[case(None, &[("ANTHROPIC_AUTH_TOKEN", "tok")], "authorization", "Bearer tok")]
    #[case(None, &[("ANTHROPIC_API_KEY", "sk-ant-oat01-env")], "authorization", "Bearer sk-ant-oat01-env")]
    fn credential_resolution_order_matches_python(
        #[case] api_key: Option<&str>,
        #[case] env: &[(&str, &str)],
        #[case] expected_header: &str,
        #[case] expected_value: &str,
    ) {
        let lookup = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        let headers = authenticate(vec![], api_key, &lookup).unwrap();
        assert_eq!(headers, vec![header(expected_header, expected_value)]);
    }

    #[test]
    fn missing_credentials_are_an_auth_error() {
        assert!(matches!(
            authenticate(vec![], None, &no_env),
            Err(litellm_auth::Error::MissingApiKey { .. })
        ));
    }

    #[rstest]
    #[case(json!({}), &[])]
    #[case(json!({"output_format": {"type": "json_schema"}}), &[beta::STRUCTURED_OUTPUT])]
    #[case(json!({"output_config": {"format": {"type": "json_schema"}}}), &[beta::STRUCTURED_OUTPUT])]
    #[case(json!({"output_config": {"effort": "high"}}), &[])]
    #[case(json!({"speed": "fast"}), &[beta::FAST_MODE_2026_02_01])]
    #[case(json!({"speed": "standard"}), &[])]
    #[case(json!({"compaction": {"enabled": true}}), &[beta::COMPACT_2026_09_04])]
    #[case(json!({"tools": [{"type": "advisor_20260301"}]}), &[beta::ADVISOR_TOOL_2026_03_01])]
    #[case(json!({"tools": [{"type": "tool_search_tool_regex_20251119"}]}), &[beta::ADVANCED_TOOL_USE_2025_11_20])]
    #[case(
        json!({"context_management": {"edits": [{"type": "compact_20260112"}, {"type": "clear_tool_uses_20250919"}]}}),
        &[beta::COMPACT_2026_01_12, beta::CONTEXT_MANAGEMENT_2025_06_27]
    )]
    #[case(
        json!({"messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}]}),
        &[beta::PER_TURN_CONTROL_2026_07_01]
    )]
    fn feature_betas_follow_the_request(#[case] fields: Value, #[case] expected: &[&str]) {
        assert_eq!(feature_betas(&request(fields)), expected);
    }

    #[test]
    fn feature_betas_merge_into_the_existing_header_sorted() {
        let headers = with_feature_betas(
            vec![header(
                "Anthropic-Beta",
                "web-search-2025-03-05, compact-2026-09-04",
            )],
            &request(json!({"speed": "fast"})),
        );
        assert_eq!(
            headers,
            vec![header(
                "anthropic-beta",
                "compact-2026-09-04,fast-mode-2026-02-01,web-search-2025-03-05"
            )]
        );
        let untouched = vec![header("x-api-key", "k")];
        assert_eq!(
            with_feature_betas(untouched.clone(), &request(json!({}))),
            untouched
        );
    }
}
