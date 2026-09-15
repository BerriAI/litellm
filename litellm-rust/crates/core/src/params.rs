use std::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpaqueFields(Map<String, Value>);

impl Deref for OpaqueFields {
    type Target = Map<String, Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for OpaqueFields {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Map<String, Value>> for OpaqueFields {
    fn from(fields: Map<String, Value>) -> Self {
        Self(fields)
    }
}

impl From<OpaqueFields> for Map<String, Value> {
    fn from(fields: OpaqueFields) -> Self {
        fields.0
    }
}

impl FromIterator<(String, Value)> for OpaqueFields {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(fields: T) -> Self {
        Self(fields.into_iter().collect())
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ParsedProviderParams<T> {
    #[serde(flatten)]
    pub known: T,
    #[serde(default, flatten)]
    pub extra_params: OpaqueFields,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("extra_body must be an object")]
    ExtraBody,
    #[error("body must be a JSON object")]
    Body,
}

impl From<Error> for crate::Error {
    fn from(error: Error) -> Self {
        Self::InvalidRequest(error.to_string())
    }
}

pub fn is_control_param(name: &str) -> bool {
    matches!(
        name,
        "api_key"
            | "api_base"
            | "custom_llm_provider"
            | "extra_headers"
            | "extra_query"
            | "timeout"
            | "timeout_seconds"
            | "request_timeout"
            | "max_retries"
            | "req_format"
            | "max_response_bytes"
            | "input_sources"
            | "litellm_call_id"
            | "litellm_logging_obj"
            | "litellm_metadata"
            | "proxy_server_request"
            | "callbacks"
            | "success_callback"
            | "failure_callback"
            | "guardrails"
            | "drop_params"
            | "additional_drop_params"
            | "allowed_openai_params"
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

pub fn body_overrides(params: &Map<String, Value>) -> Result<Option<&Map<String, Value>>, Error> {
    match params.get("extra_body") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(fields)) => Ok(Some(fields)),
        Some(_) => Err(Error::ExtraBody),
    }
}

pub fn provider_fields(params: &Map<String, Value>) -> OpaqueFields {
    params
        .iter()
        .filter(|(name, _)| !is_control_param(name))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

pub fn compose_body<B: Serialize>(
    body: &B,
    params: &Map<String, Value>,
    consumed: &[&str],
) -> Result<Value, Error> {
    let overrides = body_overrides(params)?;
    let Value::Object(fields) = serde_json::to_value(body).map_err(|_| Error::Body)? else {
        return Err(Error::Body);
    };
    Ok(Value::Object(
        fields
            .into_iter()
            .chain(
                params
                    .iter()
                    .filter(|(name, _)| !consumed.contains(&name.as_str()))
                    .chain(overrides.into_iter().flat_map(|fields| fields.iter()))
                    .filter(|(name, _)| name.as_str() != "extra_body" && !is_control_param(name))
                    .map(|(name, value)| (name.clone(), value.clone())),
            )
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    #[case::null(json!(null))]
    #[case::false_value(json!(false))]
    #[case::zero(json!(0))]
    #[case::empty_string(json!(""))]
    #[case::empty_array(json!([]))]
    #[case::empty_object(json!({}))]
    #[case::nested_names(json!({"timeout":null,"extra_body":{"api_key":"data"}}))]
    fn unknown_values_survive_composition(#[case] value: Value) {
        let params = json!({"future":value});
        assert_eq!(
            compose_body(&json!({}), params.as_object().unwrap(), &[]).unwrap(),
            params
        );
    }

    #[test]
    fn opaque_fields_do_not_apply_request_policy() {
        let value =
            json!({"extra_body":{"api_key":"data"},"timeout":null,"future":[false,0,"",[],{}]});
        let fields: OpaqueFields = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(fields).unwrap(), value);
    }

    #[test]
    fn overrides_replace_objects_without_dropping_nulls_or_nested_names() {
        let params = json!({
            "future":{"extra_body":{"timeout":null}},
            "api_key":"secret",
            "extra_body":{"settings":{"b":2},"explicit_null":null,"aws_secret_access_key":"secret"}
        });
        assert_eq!(
            compose_body(
                &json!({"settings":{"a":1}}),
                params.as_object().unwrap(),
                &[]
            )
            .unwrap(),
            json!({"settings":{"b":2},"explicit_null":null,"future":{"extra_body":{"timeout":null}}})
        );
    }

    #[test]
    fn consumed_options_are_not_remapped_but_overrides_are_applied() {
        let params = json!({"temperature":0.2,"future":true,"extra_body":{"inferenceConfig":{"temperature":0.7}}});
        assert_eq!(
            compose_body(
                &json!({"inferenceConfig":{"temperature":0.2,"maxTokens":10}}),
                params.as_object().unwrap(),
                &["temperature"]
            )
            .unwrap(),
            json!({"inferenceConfig":{"temperature":0.7},"future":true})
        );
    }

    #[rstest]
    #[case::absent(json!({}))]
    #[case::null(json!({"extra_body":null}))]
    #[case::empty(json!({"extra_body":{}}))]
    fn empty_overrides(#[case] params: Value) {
        assert_eq!(
            compose_body(&json!({}), params.as_object().unwrap(), &[]).unwrap(),
            json!({})
        );
    }

    #[rstest]
    #[case::boolean(json!(false))]
    #[case::number(json!(1))]
    #[case::array(json!([]))]
    #[case::string(json!("body"))]
    fn invalid_overrides(#[case] value: Value) {
        let params = json!({"extra_body":value});
        assert_eq!(
            compose_body(&json!({}), params.as_object().unwrap(), &[]),
            Err(Error::ExtraBody)
        );
    }

    #[rstest]
    fn controls_are_not_body_extensions(
        #[values(
            "api_key",
            "timeout",
            "callbacks",
            "aws_secret_access_key",
            "vertex_credentials",
            "azure_ad_token"
        )]
        name: &str,
        #[values(false, true)] explicit_override: bool,
    ) {
        let fields = Map::from_iter([(name.into(), json!("secret-or-control"))]);
        let params = if explicit_override {
            json!({"extra_body":fields})
        } else {
            Value::Object(fields)
        };
        assert_eq!(
            compose_body(&json!({}), params.as_object().unwrap(), &[]).unwrap(),
            json!({})
        );
    }
}
