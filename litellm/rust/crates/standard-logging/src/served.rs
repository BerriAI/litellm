//! What the payload reads of the deployment and the provider response that served the call: the
//! logged model name, the `api_base`, a cache hit's id, and the Claude subscription fields of an
//! Anthropic call made with the caller's OAuth token. Each mirrors the Python helper named on it.

use std::collections::HashMap;

use litellm_python_compat::repr::float_repr;

use crate::ClaudeSubscriptionUsage;

/// `ANTHROPIC_OAUTH_FORWARD_PROVIDERS`.
const ANTHROPIC_OAUTH_FORWARD_PROVIDERS: [&str; 1] = ["anthropic"];
const ANTHROPIC: &str = "anthropic";
const BEDROCK: &str = "bedrock";
/// The prefix `_hidden_params["additional_headers"]` names a provider header under.
const PROVIDER_HEADER_PREFIX: &str = "llm_provider-";
const UNIFIED: &str = "anthropic-ratelimit-unified-";

/// `reconstruct_model_name(model, custom_llm_provider, metadata)`: the deployment's model when it
/// names its provider, else `model`, which a Bedrock call prefixes with its provider.
pub fn reconstruct_model_name(
    model: &str,
    custom_llm_provider: Option<&str>,
    deployment: Option<&str>,
) -> String {
    if let Some(deployment) = deployment.filter(|deployment| deployment.contains('/')) {
        return deployment.to_owned();
    }
    if custom_llm_provider == Some(BEDROCK) && !model.is_empty() && !model.contains('/') {
        return format!("{BEDROCK}/{model}");
    }
    model.to_owned()
}

/// `mask_api_base_credentials(api_base)`: a `key=` query value is cut down to its last four
/// characters.
pub fn mask_api_base_credentials(api_base: &str) -> String {
    let Some(at) = api_base.find("key=") else {
        return api_base.to_owned();
    };
    let tail_start = api_base
        .char_indices()
        .rev()
        .nth(3)
        .map_or(0, |(start, _)| start);
    format!("{}*****{}", &api_base[..at + 4], &api_base[tail_start..])
}

/// `StandardLoggingPayloadSetup.strip_trailing_slash(api_base)`: a run of trailing slashes ending
/// in two is removed whole, a single one alone.
pub fn strip_trailing_slash(api_base: &str) -> &str {
    if api_base.ends_with("//") {
        api_base.trim_end_matches('/')
    } else {
        api_base.strip_suffix('/').unwrap_or(api_base)
    }
}

/// The payload's `api_base`: `strip_trailing_slash(litellm_params["api_base"]) or ""`, where the
/// logging object recorded the base the call was sent to through `_get_masked_api_base`.
pub fn api_base(sent_to: Option<&str>) -> String {
    strip_trailing_slash(&mask_api_base_credentials(sent_to.unwrap_or_default())).to_owned()
}

/// The id of a call served from the cache, `f"{id}_cache_hit{time.time()}"`, so it does not repeat
/// the id of the call that filled the cache. `now` is seconds since the Unix epoch.
pub fn cache_hit_id(id: &str, now: f64) -> String {
    format!("{id}_cache_hit{}", float_repr(now))
}

/// `resolve_used_client_oauth_token(client_sent_oauth_token, custom_llm_provider)`: whether the
/// caller's OAuth token reached a provider that takes it. `None` when the proxy stamped nothing.
pub fn used_client_oauth_token(
    client_sent_oauth_token: Option<bool>,
    custom_llm_provider: Option<&str>,
) -> Option<bool> {
    client_sent_oauth_token.map(|sent| {
        sent && custom_llm_provider
            .is_some_and(|provider| ANTHROPIC_OAUTH_FORWARD_PROVIDERS.contains(&provider))
    })
}

/// `claude_subscription_usage(call_details, response)`: the unified rate limit status of an
/// Anthropic call made with the caller's OAuth token, from the provider's response headers.
/// `client_sent_oauth_token` is the proxy's stamp, before `used_client_oauth_token` resolves it.
pub fn claude_subscription_usage(
    custom_llm_provider: Option<&str>,
    client_sent_oauth_token: Option<bool>,
    provider_headers: &[(String, String)],
) -> Option<ClaudeSubscriptionUsage> {
    if custom_llm_provider != Some(ANTHROPIC) || client_sent_oauth_token != Some(true) {
        return None;
    }
    let headers: HashMap<String, &str> = provider_headers
        .iter()
        .map(|(name, value)| {
            let name = name.to_lowercase();
            let name = match name.strip_prefix(PROVIDER_HEADER_PREFIX) {
                Some(unprefixed) => unprefixed.to_owned(),
                None => name,
            };
            (name, value.as_str())
        })
        .collect();
    let unified = |name: &str| headers.get(&format!("{UNIFIED}{name}")).copied();
    let status = unified("status");
    let overage_status = unified("overage-status");
    let is_overage = unified("overage-in-use") == Some("true")
        || (status == Some("rejected")
            && matches!(overage_status, Some("allowed" | "allowed_warning")));
    if status.is_none() && !is_overage {
        return None;
    }
    Some(ClaudeSubscriptionUsage {
        is_overage,
        status: status.map(str::to_owned),
        overage_status: overage_status.map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::the_deployment_names_its_provider(
        "claude-x",
        Some("anthropic"),
        Some("anthropic/claude-x"),
        "anthropic/claude-x"
    )]
    #[case::a_bare_deployment("claude-x", Some("anthropic"), Some("claude-x"), "claude-x")]
    #[case::bedrock_gets_its_prefix(
        "us.anthropic.claude-x",
        Some("bedrock"),
        None,
        "bedrock/us.anthropic.claude-x"
    )]
    #[case::bedrock_already_prefixed(
        "converse/claude-x",
        Some("bedrock"),
        None,
        "converse/claude-x"
    )]
    #[case::no_model("", Some("bedrock"), None, "")]
    fn the_logged_model_names_its_provider_where_python_does(
        #[case] model: &str,
        #[case] provider: Option<&str>,
        #[case] deployment: Option<&str>,
        #[case] logged: &str,
    ) {
        assert_eq!(reconstruct_model_name(model, provider, deployment), logged);
    }

    #[rstest]
    #[case::none(None, "")]
    #[case::one_slash(Some("https://api.anthropic.com/"), "https://api.anthropic.com")]
    #[case::many_slashes(Some("https://api.anthropic.com///"), "https://api.anthropic.com")]
    #[case::no_slash(
        Some("https://api.anthropic.com/v1/messages"),
        "https://api.anthropic.com/v1/messages"
    )]
    #[case::a_key(
        Some("https://g.test/v1?key=credential-value"),
        "https://g.test/v1?key=*****alue"
    )]
    fn the_api_base_is_masked_then_loses_its_trailing_slash(
        #[case] sent_to: Option<&str>,
        #[case] logged: &str,
    ) {
        assert_eq!(api_base(sent_to), logged);
    }

    #[rstest]
    fn a_cache_hit_id_names_when_it_was_served() {
        assert_eq!(
            cache_hit_id("msg_1", 1_700_000_000.5),
            "msg_1_cache_hit1700000000.5"
        );
    }

    #[rstest]
    #[case::unstamped(None, Some("anthropic"), None)]
    #[case::not_sent(Some(false), Some("anthropic"), Some(false))]
    #[case::sent_to_anthropic(Some(true), Some("anthropic"), Some(true))]
    #[case::sent_elsewhere(Some(true), Some("bedrock"), Some(false))]
    fn the_oauth_token_was_used_only_by_anthropic(
        #[case] stamped: Option<bool>,
        #[case] provider: Option<&str>,
        #[case] used: Option<bool>,
    ) {
        assert_eq!(used_client_oauth_token(stamped, provider), used);
    }

    fn unified(sent: &[(&str, &str)]) -> Vec<(String, String)> {
        sent.iter()
            .map(|(name, value)| {
                (
                    format!("Anthropic-Ratelimit-Unified-{name}"),
                    (*value).to_owned(),
                )
            })
            .collect()
    }

    #[rstest]
    #[case::allowed(&[("status", "allowed")], Some((false, Some("allowed"), None)))]
    #[case::overage_in_use(&[("overage-in-use", "true")], Some((true, None, None)))]
    #[case::rejected_into_overage(&[("status", "rejected"), ("overage-status", "allowed_warning")], Some((true, Some("rejected"), Some("allowed_warning"))))]
    #[case::rejected(&[("status", "rejected"), ("overage-status", "rejected")], Some((false, Some("rejected"), Some("rejected"))))]
    #[case::no_unified_headers(&[("overage-status", "allowed")], None)]
    fn subscription_usage_reads_the_unified_rate_limit(
        #[case] sent: &[(&str, &str)],
        #[case] usage: Option<(bool, Option<&str>, Option<&str>)>,
    ) {
        assert_eq!(
            claude_subscription_usage(Some("anthropic"), Some(true), &unified(sent)),
            usage.map(
                |(is_overage, status, overage_status)| ClaudeSubscriptionUsage {
                    is_overage,
                    status: status.map(str::to_owned),
                    overage_status: overage_status.map(str::to_owned),
                }
            )
        );
    }

    #[rstest]
    #[case::not_anthropic(Some("bedrock"), Some(true))]
    #[case::no_oauth_token(Some("anthropic"), Some(false))]
    fn subscription_usage_is_only_for_an_oauth_call_to_anthropic(
        #[case] provider: Option<&str>,
        #[case] stamped: Option<bool>,
    ) {
        assert_eq!(
            claude_subscription_usage(provider, stamped, &unified(&[("status", "allowed")])),
            None
        );
    }
}
