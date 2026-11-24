//! The Bedrock spellings the candidate walk reads, as main's `litellm/llms/bedrock/common_utils.py` reads them. The
//! catalog's name forms and the Bedrock routes read these too, through model-catalog's re-export

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
}
