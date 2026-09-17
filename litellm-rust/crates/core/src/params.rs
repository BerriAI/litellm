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

impl OpaqueParams {
    pub fn into_inner(self) -> Map<String, Value> {
        self.0
    }

    pub fn without(&self, names: &[&str]) -> Self {
        self.iter()
            .filter(|(name, _)| !names.contains(&name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    pub fn provider_params(&self) -> Self {
        self.iter()
            .filter(|(name, _)| !is_control_param(name))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    pub fn into_provider_body(self) -> Result<Map<String, Value>, Error> {
        let mut fields = self.0;
        let overrides = match fields.remove("extra_body") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(fields)) => fields,
            Some(_) => {
                return Err(Error::ExtraBody);
            }
        };
        Ok(fields
            .into_iter()
            .chain(overrides)
            .filter(|(name, _)| name != "extra_body" && !is_control_param(name))
            .collect())
    }
}

#[cfg(test)]
fn merge_extra_params<B: Serialize>(body: &B, extra_params: OpaqueParams) -> Result<Value, Error> {
    let Value::Object(fields) = serde_json::to_value(body).map_err(|_| Error::Body)? else {
        return Err(Error::Body);
    };
    Ok(Value::Object(
        fields
            .into_iter()
            .chain(
                extra_params
                    .into_provider_body()?
                    .into_iter()
                    .filter(|(name, _)| name != "model"),
            )
            .collect(),
    ))
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

    use super::*;

    #[test]
    fn extras_merge_shallowly_and_preserve_values_without_leaking_controls() {
        let extras: OpaqueParams = serde_json::from_value(json!({
            "future": {"nested": [false, 0, null]},
            "explicit_null": null,
            "azure_ad_token": "secret",
            "req_format": "native",
            "extra_body": {
                "future": {"replacement": true},
                "temperature": 0.5,
                "model": "override",
                "aws_secret_access_key": "secret"
            }
        }))
        .unwrap();
        let body =
            merge_extra_params(&json!({"model":"resolved", "temperature":0.1}), extras).unwrap();
        assert_eq!(
            body,
            json!({
                "model":"resolved", "temperature":0.5,
                "future":{"replacement":true}, "explicit_null":null
            })
        );
    }

    #[test]
    fn invalid_extra_body_is_rejected_and_null_is_empty() {
        for value in [json!(false), json!([]), json!("value"), json!(1)] {
            let params: OpaqueParams = serde_json::from_value(json!({"extra_body":value})).unwrap();
            assert!(params.into_provider_body().is_err());
        }
        let params: OpaqueParams =
            serde_json::from_value(json!({"extra_body":null,"future":null})).unwrap();
        assert_eq!(
            Value::Object(params.into_provider_body().unwrap()),
            json!({"future":null})
        );
    }

    #[test]
    fn provider_params_preserve_opaque_values() {
        let params: OpaqueParams = serde_json::from_value(json!({
            "object": {"future": [1, null]},
            "null": null,
            "azure_ad_token": "secret"
        }))
        .unwrap();

        let retained = params.provider_params();

        assert_eq!(
            serde_json::to_value(retained).unwrap(),
            json!({"object": {"future": [1, null]}, "null": null})
        );
    }

    #[test]
    fn outer_value_must_be_an_object() {
        assert!(serde_json::from_value::<OpaqueParams>(json!(["value"])).is_err());
    }
}
