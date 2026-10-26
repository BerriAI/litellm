use litellm_layered::Layered;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use veil::Redact;

#[derive(Layered)]
pub struct AwsKeys {
    #[layer(
        skip_blank,
        litellm_param = "aws_access_key_id",
        env = "AWS_ACCESS_KEY_ID"
    )]
    pub access_key_id: String,
    #[layer(
        skip_blank,
        litellm_param = "aws_secret_access_key",
        env = "AWS_SECRET_ACCESS_KEY",
        secret
    )]
    pub secret_access_key: String,
    #[layer(
        skip_blank,
        litellm_param = "aws_session_token",
        env = "AWS_SESSION_TOKEN",
        secret
    )]
    pub session_token: String,
    #[layer(skip_blank, litellm_param = "aws_region_name", env = ["AWS_REGION_NAME", "AWS_REGION"])]
    pub region: String,
    #[layer(
        skip_blank,
        litellm_param = "aws_session_name",
        env = "AWS_SESSION_NAME"
    )]
    pub session_name: String,
    #[layer(
        skip_blank,
        litellm_param = "aws_profile_name",
        env = "AWS_PROFILE_NAME"
    )]
    pub profile_name: String,
    #[layer(skip_blank, litellm_param = "aws_role_name", env = "AWS_ROLE_NAME")]
    pub role_name: String,
    #[layer(
        skip_blank,
        litellm_param = "aws_web_identity_token",
        env = "AWS_WEB_IDENTITY_TOKEN",
        secret
    )]
    pub web_identity_token: String,
    #[layer(
        skip_blank,
        litellm_param = "aws_sts_endpoint",
        env = "AWS_STS_ENDPOINT",
        label = "STS endpoint"
    )]
    pub sts_endpoint: String,
    #[layer(
        skip_blank,
        litellm_param = "aws_external_id",
        env = "AWS_EXTERNAL_ID",
        secret
    )]
    pub external_id: String,
    #[layer(
        skip_blank,
        litellm_param = "aws_bedrock_runtime_endpoint",
        env = "AWS_BEDROCK_RUNTIME_ENDPOINT",
        label = "Bedrock runtime endpoint"
    )]
    pub bedrock_runtime_endpoint: String,
}

/// The `aws_*` fields of Python's `GenericLiteLLMParams`.
#[derive(Redact, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AwsParams {
    #[serde(default)]
    pub aws_access_key_id: Option<String>,
    #[serde(default)]
    #[redact(with = "[REDACTED]")]
    pub aws_secret_access_key: Option<String>,
    #[serde(default)]
    #[redact(with = "[REDACTED]")]
    pub aws_session_token: Option<String>,
    #[serde(default)]
    pub aws_region_name: Option<String>,
    #[serde(default)]
    pub aws_session_name: Option<String>,
    #[serde(default)]
    pub aws_profile_name: Option<String>,
    #[serde(default)]
    pub aws_role_name: Option<String>,
    #[serde(default)]
    #[redact(with = "[REDACTED]")]
    pub aws_web_identity_token: Option<String>,
    #[serde(default)]
    pub aws_sts_endpoint: Option<String>,
    #[serde(default)]
    #[redact(with = "[REDACTED]")]
    pub aws_external_id: Option<String>,
    #[serde(default)]
    pub aws_bedrock_runtime_endpoint: Option<String>,
}

impl AwsParams {
    pub fn keys(&self) -> AwsKeysPartial {
        AwsKeysPartial::from_litellm_params(&super::as_litellm_params(self))
    }

    /// Reads the string-valued `aws_*` keys of an untyped params map, ignoring anything else.
    pub fn from_optional_params(optional_params: &Map<String, Value>) -> Self {
        let value = |key: &str| {
            optional_params
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        Self {
            aws_access_key_id: value("aws_access_key_id"),
            aws_secret_access_key: value("aws_secret_access_key"),
            aws_session_token: value("aws_session_token"),
            aws_region_name: value("aws_region_name"),
            aws_session_name: value("aws_session_name"),
            aws_profile_name: value("aws_profile_name"),
            aws_role_name: value("aws_role_name"),
            aws_web_identity_token: value("aws_web_identity_token"),
            aws_sts_endpoint: value("aws_sts_endpoint"),
            aws_external_id: value("aws_external_id"),
            aws_bedrock_runtime_endpoint: value("aws_bedrock_runtime_endpoint"),
        }
    }
}
