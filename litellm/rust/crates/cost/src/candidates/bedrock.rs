//! The Bedrock spellings the candidate walk reads, as main's `litellm/llms/bedrock/common_utils.py` reads them. The
//! catalog's name forms and the Bedrock routes read these too, through model-catalog's re-export, and router-core
//! reads the base model here directly

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

/// The geo prefixes of a cross-region inference profile id such as `us.anthropic.claude-x`
pub const CROSS_REGION_INFERENCE_PREFIXES: [&str; 8] =
    ["global", "us", "eu", "apac", "jp", "au", "ca", "us-gov"];

/// Port of Python's `strip_bedrock_routing_prefix`.
pub fn strip_bedrock_routing_prefix(model: &str) -> &str {
    ROUTING_PREFIXES.into_iter().fold(model, |model, prefix| {
        model.strip_prefix(prefix).unwrap_or(model)
    })
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

const ROUTE_PREFIXES: [&str; 3] = ["bedrock/converse/", "bedrock/", "converse/"];

/// Whether a path segment of a model id reads as an AWS region, such as `us-east-1`
pub fn is_bedrock_region(value: &str) -> bool {
    value.len() > 3
        && value.contains('-')
        && value
            .chars()
            .all(|char| char.is_ascii_alphanumeric() || char == '-')
}

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

    #[rstest]
    #[case::plain_id("anthropic.claude-x-v1:0", "anthropic.claude-x-v1:0")]
    #[case::bedrock_route("bedrock/anthropic.claude-x-v1:0", "anthropic.claude-x-v1:0")]
    #[case::converse_route("bedrock/converse/anthropic.claude-x", "anthropic.claude-x")]
    #[case::mantle_route(
        "bedrock/mantle/anthropic.claude-sonnet-5",
        "anthropic.claude-sonnet-5"
    )]
    #[case::cross_region("us.anthropic.claude-x-v1:0", "anthropic.claude-x-v1:0")]
    #[case::canada_cross_region("ca.anthropic.claude-x-v1:0", "anthropic.claude-x-v1:0")]
    #[case::gov_cross_region("us-gov.anthropic.claude-x-v1:0", "anthropic.claude-x-v1:0")]
    #[case::routed_cross_region(
        "bedrock/converse/us.anthropic.claude-x-v1:0",
        "anthropic.claude-x-v1:0"
    )]
    #[case::invoke_route_cross_region("invoke/global.anthropic.claude-x", "anthropic.claude-x")]
    #[case::not_a_geo("usa.anthropic.claude-x", "usa.anthropic.claude-x")]
    #[case::throughput_suffix("anthropic.claude-x-v2:0:51k", "anthropic.claude-x-v2:0")]
    #[case::cross_region_throughput_suffix(
        "us.anthropic.claude-x-v2:0:51k",
        "anthropic.claude-x-v2:0"
    )]
    #[case::version_pair_without_throughput("anthropic.claude-x:1:2", "anthropic.claude-x:1:2")]
    #[case::context_tag("anthropic.claude-x[1m]", "anthropic.claude-x")]
    #[case::context_tag_after_throughput(
        "anthropic.claude-x-v2:0:51k[1m]",
        "anthropic.claude-x-v2:0:51k"
    )]
    #[case::region_path("us-east-1/anthropic.claude-x", "anthropic.claude-x")]
    #[case::routed_region_path(
        "bedrock/us-gov-west-1/openai.gpt-oss-20b-1:0",
        "openai.gpt-oss-20b-1:0"
    )]
    #[case::region_path_keeps_cross_region(
        "bedrock/us-east-1/us.anthropic.claude-x",
        "us.anthropic.claude-x"
    )]
    #[case::profile_arn(
        "arn:aws:bedrock:us-east-1:123:application-inference-profile/us.anthropic.claude-x",
        "anthropic.claude-x"
    )]
    #[case::routed_opaque_arn(
        "bedrock/converse/arn:aws:bedrock:us-east-1:123:provisioned-model/abc123",
        "abc123"
    )]
    #[case::foundation_model_arn(
        "arn:aws:bedrock:us-east-1:123:foundation-model/anthropic.claude-x-v1:0",
        "anthropic.claude-x-v1:0"
    )]
    #[case::nova_2_custom(
        "bedrock/nova-2/arn:aws:bedrock:us-east-1:1:custom-model/x",
        "amazon.nova-2-custom"
    )]
    #[case::nova_custom("bedrock/nova/arn:aws:x", "amazon.nova-custom")]
    #[case::unrouted_nova_custom("nova/arn:aws:x", "amazon.nova-custom")]
    #[case::a_nova_foundation_model("us.amazon.nova-pro-v1:0", "amazon.nova-pro-v1:0")]
    fn base_model_matches_the_python_resolution(#[case] model: &str, #[case] expected: &str) {
        assert_eq!(bedrock_base_model(model), expected);
    }

    #[rstest]
    #[case::us("us")]
    #[case::global("global")]
    fn a_bare_geo_word_is_its_own_base_model_where_python_raises(#[case] model: &str) {
        assert_eq!(bedrock_base_model(model), model);
    }
}
