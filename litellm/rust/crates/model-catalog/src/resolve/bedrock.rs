//! Bedrock model id spellings: routing prefixes, region paths, ARNs and throughput suffixes, as main's
//! `litellm/llms/bedrock/common_utils.py` reads them. Catalog lookups and the Bedrock routes share these

use litellm_auth_aws::is_bedrock_region;

const ROUTE_PREFIXES: [&str; 3] = ["bedrock/converse/", "bedrock/", "converse/"];
const ROUTING_PREFIXES: [&str; 8] = [
    "bedrock/",
    "chat_completions/",
    "converse/",
    "invoke/",
    "openai/",
    "mantle/",
    "nova-2/",
    "nova/",
];
const CROSS_REGION_INFERENCE_PREFIXES: [&str; 8] =
    ["global", "us", "eu", "apac", "jp", "au", "ca", "us-gov"];

/// Port of Python's `get_bedrock_base_model`
pub fn bedrock_base_model(model: &str) -> &str {
    let unrouted = ROUTE_PREFIXES
        .into_iter()
        .find_map(|prefix| model.strip_prefix(prefix))
        .unwrap_or(model);
    if unrouted.starts_with("nova-2/") {
        return "amazon.nova-2-custom";
    }
    if unrouted.starts_with("nova/") {
        return "amazon.nova-custom";
    }
    let named = without_throughput_suffix(bedrock_model_name_from_arn(
        strip_bedrock_routing_prefix(model),
    ));
    if let Some((region, rest)) = named.split_once('.')
        && CROSS_REGION_INFERENCE_PREFIXES.contains(&region)
    {
        return rest;
    }
    match named.split_once('/') {
        Some((region, rest)) if is_bedrock_region(region) => rest,
        _ => named,
    }
}

/// Port of Python's `strip_bedrock_routing_prefix`.
pub fn strip_bedrock_routing_prefix(model: &str) -> &str {
    ROUTING_PREFIXES.into_iter().fold(model, |model, prefix| {
        model.strip_prefix(prefix).unwrap_or(model)
    })
}

pub fn split_bedrock_region_path(model: &str) -> (Option<&str>, &str) {
    let stripped = strip_bedrock_routing_prefix(model);
    match stripped.split_once('/') {
        Some((region, model_id)) if is_bedrock_region(region) => (Some(region), model_id),
        _ => (None, stripped),
    }
}

/// Port of Python's `_is_bedrock_cross_region_profile_of`: `profile` is a cross-region profile such as
/// `us.anthropic.claude-x` of `served_model`
pub fn is_bedrock_cross_region_profile_of(profile: &str, served_model: &str) -> bool {
    let (region, profiled_model) = strip_bedrock_routing_prefix(profile)
        .split_once('.')
        .unwrap_or((strip_bedrock_routing_prefix(profile), ""));
    CROSS_REGION_INFERENCE_PREFIXES.contains(&region)
        && strip_bedrock_routing_prefix(served_model) == profiled_model
}

/// Port of Python's `extract_model_name_from_bedrock_arn`
pub fn bedrock_model_name_from_arn(model: &str) -> &str {
    if !model.to_lowercase().contains("arn") {
        return model;
    }
    model.rsplit('/').next().unwrap_or(model)
}

fn without_throughput_suffix(model: &str) -> &str {
    let unthrottled = model
        .strip_suffix('k')
        .and_then(|rest| rest.rsplit_once(':'))
        .filter(|(head, units)| {
            is_digits(units)
                && head
                    .rsplit_once(':')
                    .is_some_and(|(_, minor)| is_digits(minor))
        })
        .map_or(model, |(head, _)| head);
    unthrottled
        .strip_suffix(']')
        .and_then(|rest| rest.rsplit_once('['))
        .filter(|(_, tag)| {
            !tag.is_empty()
                && tag
                    .chars()
                    .all(|char| char.is_alphanumeric() || char == '_')
        })
        .map_or(unthrottled, |(head, _)| head)
}

fn is_digits(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::cross_region("us.anthropic.claude-x-v1:0", "anthropic.claude-x-v1:0")]
    #[case::canada_cross_region("ca.anthropic.claude-x-v1:0", "anthropic.claude-x-v1:0")]
    #[case::converse_route("bedrock/converse/anthropic.claude-x", "anthropic.claude-x")]
    #[case::mantle_route(
        "bedrock/mantle/anthropic.claude-sonnet-5",
        "anthropic.claude-sonnet-5"
    )]
    #[case::invoke_route_cross_region("invoke/global.anthropic.claude-x", "anthropic.claude-x")]
    #[case::throughput_suffix("anthropic.claude-x-v2:0:51k", "anthropic.claude-x-v2:0")]
    #[case::version_pair_without_throughput("anthropic.claude-x:1:2", "anthropic.claude-x:1:2")]
    #[case::context_tag("anthropic.claude-x[1m]", "anthropic.claude-x")]
    #[case::region_path("us-east-1/anthropic.claude-x", "anthropic.claude-x")]
    #[case::region_path_keeps_cross_region(
        "bedrock/us-east-1/us.anthropic.claude-x",
        "us.anthropic.claude-x"
    )]
    #[case::profile_arn(
        "arn:aws:bedrock:us-east-1:123:application-inference-profile/us.anthropic.claude-x",
        "anthropic.claude-x"
    )]
    #[case::nova_2_custom(
        "bedrock/nova-2/arn:aws:bedrock:us-east-1:1:custom-model/x",
        "amazon.nova-2-custom"
    )]
    #[case::nova_custom("bedrock/nova/arn:aws:x", "amazon.nova-custom")]
    fn base_model_matches_the_python_resolution(#[case] model: &str, #[case] expected: &str) {
        assert_eq!(bedrock_base_model(model), expected);
    }

    #[rstest]
    #[case::us_profile("us.anthropic.claude-x", "anthropic.claude-x", true)]
    #[case::routed_profile("invoke/global.anthropic.claude-x", "anthropic.claude-x", true)]
    #[case::other_model("us.anthropic.claude-x", "anthropic.claude-y", false)]
    #[case::not_a_geo("anthropic.claude-x", "claude-x", false)]
    #[case::no_dot("claude-x", "claude-x", false)]
    fn a_cross_region_profile_names_its_served_model(
        #[case] profile: &str,
        #[case] served: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(
            is_bedrock_cross_region_profile_of(profile, served),
            expected
        );
    }
}
