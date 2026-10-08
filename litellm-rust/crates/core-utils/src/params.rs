#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid request: extra_body must be an object")]
    ExtraBody,
    #[error("invalid request: body must be a JSON object")]
    Body,
}

use std::ops::{Deref, DerefMut};

use litellm_auth_types::is_connection_name;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpaqueParams(Map<String, Value>);

mod owned;

pub fn is_litellm_owned(name: &str) -> bool {
    is_control_param(name)
        || is_connection_name(name)
        || name.starts_with(owned::INTERNAL_PREFIX)
        || owned::PYTHON_OWNED.binary_search(&name).is_ok()
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
            | "azure_ad_token_provider"
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

    use super::{OpaqueParams, is_litellm_owned, owned::PYTHON_OWNED};

    #[rstest]
    #[case::rust_control("drop_params")]
    #[case::credential("aws_secret_access_key")]
    #[case::credential_alias("vertex_ai_location")]
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
    fn outer_value_must_be_an_object() {
        assert!(serde_json::from_value::<OpaqueParams>(json!(["value"])).is_err());
    }
}
