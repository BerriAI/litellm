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

mod owned;

pub fn is_litellm_owned(name: &str) -> bool {
    is_control_param(name)
        || name.starts_with(owned::INTERNAL_PREFIX)
        || owned::PYTHON_OWNED.binary_search(&name).is_ok()
}

pub fn is_secret_param(name: &str) -> bool {
    matches!(
        name,
        "azure_ad_token"
            | "client_secret"
            | "azure_federated_token_file"
            | "vertex_credentials"
            | "vertex_ai_credentials"
            | "aws_secret_access_key"
            | "aws_session_token"
            | "aws_web_identity_token"
    )
}

fn is_control_param(name: &str) -> bool {
    matches!(
        name,
        "model"
            | "extra_body"
            | "base_url"
            | "default_headers"
            | "organization"
            | "deployment_id"
            | "callbacks"
            | "success_callback"
            | "failure_callback"
            | "drop_params"
            | "additional_drop_params"
            | "api_key"
            | "api_base"
            | "custom_llm_provider"
            | "extra_headers"
            | "timeout"
            | "timeout_seconds"
            | "request_timeout"
            | "max_retries"
            | "req_format"
            | "max_response_bytes"
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
    use rstest::rstest;
    use serde_json::json;

    use super::{OpaqueParams, is_litellm_owned, is_secret_param, owned::PYTHON_OWNED};

    #[rstest]
    #[case::rust_control("drop_params")]
    #[case::rust_credential("aws_secret_access_key")]
    #[case::python_connection("api_key")]
    #[case::python_logging("litellm_call_id")]
    #[case::python_metadata("metadata")]
    #[case::python_callback_credential("langfuse_secret_key")]
    #[case::python_pricing("input_cost_per_token")]
    #[case::internal_prefix("_litellm_anything_new")]
    fn owned_names(#[case] name: &str) {
        assert!(is_litellm_owned(name));
    }

    #[rstest]
    #[case::openai_param("temperature")]
    #[case::provider_field("top_k")]
    #[case::unknown_field("future_provider_option")]
    #[case::prefix_without_leading_underscore("litellm_")]
    fn provider_names(#[case] name: &str) {
        assert!(!is_litellm_owned(name));
    }

    #[test]
    fn every_generated_python_name_is_owned() {
        assert!(PYTHON_OWNED.iter().all(|name| is_litellm_owned(name)));
    }

    #[test]
    fn every_secret_is_owned() {
        let secrets = [
            "azure_ad_token",
            "client_secret",
            "azure_federated_token_file",
            "vertex_credentials",
            "vertex_ai_credentials",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_web_identity_token",
        ];
        assert!(
            secrets
                .iter()
                .all(|name| is_secret_param(name) && is_litellm_owned(name))
        );
        assert!(!is_secret_param("aws_region_name"));
    }

    #[test]
    fn outer_value_must_be_an_object() {
        assert!(serde_json::from_value::<OpaqueParams>(json!(["value"])).is_err());
    }
}
