#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid request: extra_body must be an object")]
    ExtraBody,
    #[error("invalid request: body must be a JSON object")]
    Body,
}

use std::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpaqueParams(Map<String, Value>);

pub fn is_control_param(name: &str) -> bool {
    matches!(
        name,
        "api_key"
            | "api_base"
            | "custom_llm_provider"
            | "extra_headers"
            | "timeout"
            | "timeout_seconds"
            | "request_timeout"
            | "max_retries"
            | "req_format"
            | "max_response_bytes"
            | "litellm_call_id"
            | "litellm_logging_obj"
            | "litellm_metadata"
            | "proxy_server_request"
            | "callbacks"
            | "success_callback"
            | "failure_callback"
            | "guardrails"
            | "azure_ad_token"
            | "azure_ad_token_provider"
            | "tenant_id"
            | "client_id"
            | "client_secret"
            | "azure_scope"
            | "azure_authority_host"
            | "azure_credential"
            | "azure_federated_token_file"
            | "enable_azure_ad_token_refresh"
            | "vertex_credentials"
            | "vertex_ai_credentials"
            | "vertex_project"
            | "vertex_ai_project"
            | "vertex_location"
            | "vertex_ai_location"
            | "aws_access_key_id"
            | "aws_secret_access_key"
            | "aws_session_token"
            | "aws_region_name"
            | "aws_session_name"
            | "aws_profile_name"
            | "aws_role_name"
            | "aws_web_identity_token"
            | "aws_sts_endpoint"
            | "aws_external_id"
            | "aws_bedrock_runtime_endpoint"
    )
}

impl Deref for OpaqueParams {
    type Target = Map<String, Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for OpaqueParams {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Map<String, Value>> for OpaqueParams {
    fn from(value: Map<String, Value>) -> Self {
        Self(value)
    }
}

impl From<OpaqueParams> for Map<String, Value> {
    fn from(value: OpaqueParams) -> Self {
        value.0
    }
}

impl FromIterator<(String, Value)> for OpaqueParams {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for OpaqueParams {
    type Item = (String, Value);
    type IntoIter = serde_json::map::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::OpaqueParams;

    #[test]
    fn outer_value_must_be_an_object() {
        assert!(serde_json::from_value::<OpaqueParams>(json!(["value"])).is_err());
    }
}
