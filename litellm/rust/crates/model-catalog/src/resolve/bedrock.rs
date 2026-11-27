//! Bedrock model id spellings: routing prefixes, region paths, ARNs and throughput suffixes, as main's
//! `litellm/llms/bedrock/common_utils.py` reads them. Catalog lookups and the Bedrock routes share these

pub use litellm_cost::candidates::bedrock::{
    bedrock_base_model, bedrock_model_name_from_arn, is_bedrock_cross_region_profile_of,
    is_bedrock_region, strip_bedrock_routing_prefix,
};

pub fn split_bedrock_region_path(model: &str) -> (Option<&str>, &str) {
    let stripped = strip_bedrock_routing_prefix(model);
    match stripped.split_once('/') {
        Some((region, model_id)) if is_bedrock_region(region) => (Some(region), model_id),
        _ => (None, stripped),
    }
}
