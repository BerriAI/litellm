use std::time::Duration;

use serde_json::{Map, Value};

#[derive(Clone, Debug, Default)]
pub struct BedrockOptions {
    pub aws_access_key_id: Option<String>,
    pub aws_secret_access_key: Option<String>,
    pub aws_session_token: Option<String>,
    pub aws_region_name: Option<String>,
    pub aws_session_name: Option<String>,
    pub aws_profile_name: Option<String>,
    pub aws_role_name: Option<String>,
    pub aws_web_identity_token: Option<String>,
    pub aws_sts_endpoint: Option<String>,
    pub aws_external_id: Option<String>,
    pub aws_bedrock_runtime_endpoint: Option<String>,
    pub request_metadata_fields: Vec<String>,
    pub request_metadata: Option<std::collections::BTreeMap<String, String>>,
}

impl BedrockOptions {
    pub fn into_map(&self) -> Map<String, Value> {
        [
            ("aws_access_key_id", self.aws_access_key_id.clone()),
            ("aws_secret_access_key", self.aws_secret_access_key.clone()),
            ("aws_session_token", self.aws_session_token.clone()),
            ("aws_region_name", self.aws_region_name.clone()),
            ("aws_session_name", self.aws_session_name.clone()),
            ("aws_profile_name", self.aws_profile_name.clone()),
            ("aws_role_name", self.aws_role_name.clone()),
            (
                "aws_web_identity_token",
                self.aws_web_identity_token.clone(),
            ),
            ("aws_sts_endpoint", self.aws_sts_endpoint.clone()),
            ("aws_external_id", self.aws_external_id.clone()),
            (
                "aws_bedrock_runtime_endpoint",
                self.aws_bedrock_runtime_endpoint.clone(),
            ),
        ]
        .into_iter()
        .filter_map(|(name, value)| value.map(|value| (name.to_string(), Value::String(value))))
        .collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct AnthropicOptions {
    pub user_id: Option<String>,
    pub has_user_id: bool,
}

#[derive(Clone, Debug, Default)]
pub struct VertexOptions {
    pub project: Option<String>,
    pub location: Option<String>,
}

impl VertexOptions {
    pub fn into_map(&self) -> Map<String, Value> {
        [
            ("vertex_project", self.project.clone()),
            ("vertex_location", self.location.clone()),
        ]
        .into_iter()
        .filter_map(|(name, value)| value.map(|value| (name.to_string(), Value::String(value))))
        .collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct RequestOptions {
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub extra_query: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
    pub bedrock: Option<BedrockOptions>,
    pub anthropic: Option<AnthropicOptions>,
    pub vertex: Option<VertexOptions>,
}
