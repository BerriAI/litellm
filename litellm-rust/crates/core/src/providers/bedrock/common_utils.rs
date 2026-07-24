use serde_json::{Map, Value};

use super::aws_base::AwsAuthConfig;
use super::constants::{AWS_REGION, AWS_REGION_NAME, DEFAULT_BEDROCK_REGION};

pub fn bedrock_model_id_and_region(model: &str) -> (String, Option<String>) {
    let mut stripped = model;
    for prefix in [
        "bedrock/converse/",
        "bedrock/messages/",
        "bedrock/",
        "converse/",
    ] {
        if let Some(value) = stripped.strip_prefix(prefix) {
            stripped = value;
            break;
        }
    }
    let mut region = None;
    if let Some((candidate, remainder)) = stripped.split_once('/')
        && is_bedrock_region(candidate)
    {
        region = Some(candidate.to_string());
        stripped = remainder;
    }
    for prefix in ["nova-2/", "nova/"] {
        if let Some(value) = stripped.strip_prefix(prefix) {
            stripped = value;
            break;
        }
    }
    if region.is_none() {
        region = stripped
            .strip_prefix("arn:")
            .and_then(|value| value.split(':').nth(2))
            .filter(|value| !value.is_empty())
            .map(str::to_string);
    }
    (stripped.to_string(), region)
}

pub fn is_bedrock_region(value: &str) -> bool {
    value.len() > 3
        && value.contains('-')
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '-')
}

pub fn resolve_bedrock_region(
    model_region: Option<&str>,
    optional_params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    optional_params
        .get("aws_region_name")
        .and_then(Value::as_str)
        .filter(|region| !region.trim().is_empty())
        .map(str::to_string)
        .or_else(|| model_region.map(str::to_string))
        .or_else(|| env_lookup(AWS_REGION_NAME))
        .or_else(|| env_lookup(AWS_REGION))
        .unwrap_or_else(|| DEFAULT_BEDROCK_REGION.to_string())
}

pub fn aws_auth_config(
    optional_params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
    region: Option<&str>,
) -> AwsAuthConfig {
    let value = |key: &str| {
        optional_params
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let env = |key: &str| env_lookup(key);
    AwsAuthConfig {
        access_key_id: value("aws_access_key_id").or_else(|| env("AWS_ACCESS_KEY_ID")),
        secret_access_key: value("aws_secret_access_key").or_else(|| env("AWS_SECRET_ACCESS_KEY")),
        session_token: value("aws_session_token").or_else(|| env("AWS_SESSION_TOKEN")),
        region_name: region
            .map(str::to_string)
            .or_else(|| value("aws_region_name"))
            .or_else(|| env(AWS_REGION_NAME)),
        session_name: value("aws_session_name").or_else(|| env("AWS_SESSION_NAME")),
        profile_name: value("aws_profile_name").or_else(|| env("AWS_PROFILE_NAME")),
        role_name: value("aws_role_name").or_else(|| env("AWS_ROLE_NAME")),
        web_identity_token: value("aws_web_identity_token")
            .or_else(|| env("AWS_WEB_IDENTITY_TOKEN")),
        sts_endpoint: value("aws_sts_endpoint").or_else(|| env("AWS_STS_ENDPOINT")),
        external_id: value("aws_external_id").or_else(|| env("AWS_EXTERNAL_ID")),
    }
}
