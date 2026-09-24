use std::sync::LazyLock;

use regex::Regex;

const ROUTING_PREFIXES: [&str; 7] = [
    "bedrock/",
    "converse/",
    "invoke/",
    "openai/",
    "mantle/",
    "nova-2/",
    "nova/",
];

const CROSS_REGION_INFERENCE_REGIONS: [&str; 7] =
    ["global", "us", "eu", "apac", "jp", "au", "us-gov"];

const ALL_REGIONS: [&str; 23] = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "us-gov-east-1",
    "us-gov-west-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-central-1",
    "eu-central-2",
    "eu-south-1",
    "eu-south-2",
    "eu-north-1",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ca-central-1",
    "sa-east-1",
];

static THROUGHPUT_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(:\d+):\d+k$").expect("valid throughput pattern"));
static CONTEXT_WINDOW_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\w+\]$").expect("valid context window pattern"));

pub fn is_bedrock_region(region: &str) -> bool {
    ALL_REGIONS.contains(&region)
}

pub fn strip_bedrock_routing_prefix(model: &str) -> &str {
    ROUTING_PREFIXES.iter().fold(model, |model, prefix| {
        if model.starts_with(prefix) {
            model.split_once('/').map_or(model, |(_, rest)| rest)
        } else {
            model
        }
    })
}

pub fn strip_bedrock_throughput_suffix(model: &str) -> String {
    let without_throughput = THROUGHPUT_SUFFIX.replace(model, "${1}");
    CONTEXT_WINDOW_SUFFIX
        .replace(&without_throughput, "")
        .into_owned()
}

pub fn extract_model_name_from_bedrock_arn(model: &str) -> &str {
    if model.to_lowercase().contains("arn") {
        model.rsplit('/').next().unwrap_or(model)
    } else {
        model
    }
}

pub fn get_bedrock_base_model(model: &str) -> String {
    let spec = ["bedrock/converse/", "bedrock/", "converse/"]
        .into_iter()
        .find_map(|prefix| model.strip_prefix(prefix))
        .unwrap_or(model);
    if spec.starts_with("nova-2/") {
        return "amazon.nova-2-custom".to_owned();
    }
    if spec.starts_with("nova/") {
        return "amazon.nova-custom".to_owned();
    }
    let model = strip_bedrock_throughput_suffix(extract_model_name_from_bedrock_arn(
        strip_bedrock_routing_prefix(model),
    ));
    if let Some((region, rest)) = model.split_once('.')
        && CROSS_REGION_INFERENCE_REGIONS.contains(&region)
    {
        return rest.to_owned();
    }
    if let Some((region, rest)) = model.split_once('/')
        && is_bedrock_region(region)
    {
        return rest.to_owned();
    }
    model
}
