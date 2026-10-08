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

mod owned;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpaqueParams(Map<String, Value>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlClass {
    Destination,
    Credential,
    Identity,
    Audience,
    Header,
    Transport,
    Control,
}

impl ControlClass {
    pub fn steers_exchange(self) -> bool {
        matches!(self, Self::Destination | Self::Identity | Self::Audience)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialFamily {
    Aws,
    Azure,
    Vertex,
}

#[derive(Debug, Serialize)]
pub struct ControlParam {
    pub name: &'static str,
    pub class: ControlClass,
    pub family: Option<CredentialFamily>,
    pub redacted: bool,
}

impl ControlParam {
    const fn new(name: &'static str, class: ControlClass) -> Self {
        Self {
            name,
            class,
            family: None,
            redacted: false,
        }
    }

    const fn aws(self) -> Self {
        Self {
            family: Some(CredentialFamily::Aws),
            ..self
        }
    }

    const fn azure(self) -> Self {
        Self {
            family: Some(CredentialFamily::Azure),
            ..self
        }
    }

    const fn vertex(self) -> Self {
        Self {
            family: Some(CredentialFamily::Vertex),
            ..self
        }
    }

    const fn redacted(self) -> Self {
        Self {
            redacted: true,
            ..self
        }
    }
}

use ControlClass::{Audience, Control, Credential, Destination, Header, Identity, Transport};

const CONTROL_PARAMS: &[ControlParam] = &[
    ControlParam::new("model", Control),
    ControlParam::new("custom_llm_provider", Control),
    ControlParam::new("deployment_id", Control),
    ControlParam::new("organization", Control),
    ControlParam::new("extra_body", Control),
    ControlParam::new("callbacks", Control),
    ControlParam::new("success_callback", Control),
    ControlParam::new("failure_callback", Control),
    ControlParam::new("drop_params", Control),
    ControlParam::new("additional_drop_params", Control),
    ControlParam::new("req_format", Control),
    ControlParam::new("max_response_bytes", Control),
    ControlParam::new("api_base", Destination),
    ControlParam::new("base_url", Destination),
    ControlParam::new("api_key", Credential).redacted(),
    ControlParam::new("extra_headers", Header),
    ControlParam::new("default_headers", Header),
    ControlParam::new("timeout", Transport),
    ControlParam::new("timeout_seconds", Transport),
    ControlParam::new("request_timeout", Transport),
    ControlParam::new("max_retries", Transport),
    ControlParam::new("azure_ad_token", Identity)
        .azure()
        .redacted(),
    ControlParam::new("azure_ad_token_provider", Control),
    ControlParam::new("tenant_id", Identity).azure(),
    ControlParam::new("client_id", Identity).azure(),
    ControlParam::new("client_secret", Credential)
        .azure()
        .redacted(),
    ControlParam::new("azure_scope", Audience).azure(),
    ControlParam::new("azure_authority_host", Destination).azure(),
    ControlParam::new("azure_credential", Identity).azure(),
    ControlParam::new("azure_federated_token_file", Identity)
        .azure()
        .redacted(),
    ControlParam::new("enable_azure_ad_token_refresh", Control).azure(),
    ControlParam::new("vertex_credentials", Identity)
        .vertex()
        .redacted(),
    ControlParam::new("vertex_ai_credentials", Identity)
        .vertex()
        .redacted(),
    ControlParam::new("vertex_project", Audience).vertex(),
    ControlParam::new("vertex_ai_project", Audience).vertex(),
    ControlParam::new("vertex_location", Destination).vertex(),
    ControlParam::new("vertex_ai_location", Destination).vertex(),
    ControlParam::new("aws_access_key_id", Credential).aws(),
    ControlParam::new("aws_secret_access_key", Credential)
        .aws()
        .redacted(),
    ControlParam::new("aws_session_token", Credential)
        .aws()
        .redacted(),
    ControlParam::new("aws_region_name", Control).aws(),
    ControlParam::new("aws_session_name", Identity).aws(),
    ControlParam::new("aws_profile_name", Identity).aws(),
    ControlParam::new("aws_role_name", Identity).aws(),
    ControlParam::new("aws_web_identity_token", Identity)
        .aws()
        .redacted(),
    ControlParam::new("aws_sts_endpoint", Destination).aws(),
    ControlParam::new("aws_external_id", Identity).aws(),
    ControlParam::new("aws_bedrock_runtime_endpoint", Destination).aws(),
];

pub fn control_params() -> &'static [ControlParam] {
    CONTROL_PARAMS
}

pub fn control_param(name: &str) -> Option<&'static ControlParam> {
    CONTROL_PARAMS.iter().find(|param| param.name == name)
}

pub fn control_class(name: &str) -> Option<ControlClass> {
    control_param(name).map(|param| param.class)
}

pub fn is_control_param(name: &str) -> bool {
    control_param(name).is_some()
}

pub fn is_secret_param(name: &str) -> bool {
    control_param(name).is_some_and(|param| param.redacted)
}

pub fn is_litellm_owned(name: &str) -> bool {
    is_control_param(name)
        || name.starts_with(owned::INTERNAL_PREFIX)
        || owned::PYTHON_OWNED.binary_search(&name).is_ok()
}

pub fn family_names(family: CredentialFamily) -> impl Iterator<Item = &'static str> {
    CONTROL_PARAMS
        .iter()
        .filter(move |param| param.family == Some(family))
        .map(|param| param.name)
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
    use std::collections::BTreeSet;

    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[test]
    fn outer_value_must_be_an_object() {
        assert!(serde_json::from_value::<OpaqueParams>(json!(["value"])).is_err());
    }

    #[test]
    fn every_control_name_is_listed_once() {
        let names: BTreeSet<&str> = CONTROL_PARAMS.iter().map(|param| param.name).collect();
        assert_eq!(names.len(), CONTROL_PARAMS.len());
    }

    #[test]
    fn the_generated_owned_set_is_sorted_for_binary_search() {
        assert!(owned::PYTHON_OWNED.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[rstest]
    #[case::rust_control("base_url", true)]
    #[case::python_owned("litellm_call_id", true)]
    #[case::internal_prefix("_litellm_anything", true)]
    #[case::provider_field("temperature", false)]
    #[case::unknown_field("future_field", false)]
    fn owned_names_come_from_controls_the_prefix_and_the_python_set(
        #[case] name: &str,
        #[case] owned: bool,
    ) {
        assert_eq!(is_litellm_owned(name), owned);
    }

    #[test]
    fn redaction_covers_exactly_the_raw_secrets() {
        let redacted: BTreeSet<&str> = CONTROL_PARAMS
            .iter()
            .filter(|param| param.redacted)
            .map(|param| param.name)
            .collect();
        assert_eq!(
            redacted,
            BTreeSet::from([
                "api_key",
                "aws_secret_access_key",
                "aws_session_token",
                "aws_web_identity_token",
                "azure_ad_token",
                "azure_federated_token_file",
                "client_secret",
                "vertex_ai_credentials",
                "vertex_credentials",
            ])
        );
        assert!(is_secret_param("client_secret"));
        assert!(!is_secret_param("tenant_id"));
    }

    #[rstest]
    #[case::destination(Destination, true)]
    #[case::identity(Identity, true)]
    #[case::audience(Audience, true)]
    #[case::credential(Credential, false)]
    #[case::header(Header, false)]
    #[case::transport(Transport, false)]
    #[case::control(Control, false)]
    fn only_destination_identity_and_audience_steer_an_exchange(
        #[case] class: ControlClass,
        #[case] steers: bool,
    ) {
        assert_eq!(class.steers_exchange(), steers);
    }

    #[rstest]
    #[case::caller_endpoint("api_base", Some(Destination))]
    #[case::token_authority("azure_authority_host", Some(Destination))]
    #[case::vertex_host("vertex_location", Some(Destination))]
    #[case::host_identity("tenant_id", Some(Identity))]
    #[case::host_file("vertex_credentials", Some(Identity))]
    #[case::token_audience("azure_scope", Some(Audience))]
    #[case::billing_project("vertex_project", Some(Audience))]
    #[case::own_key("api_key", Some(Credential))]
    #[case::aws_region_stays_on_aws("aws_region_name", Some(Control))]
    #[case::provider_field("temperature", None)]
    fn names_are_classified_by_what_a_caller_could_do_with_them(
        #[case] name: &str,
        #[case] class: Option<ControlClass>,
    ) {
        assert_eq!(control_class(name), class);
    }

    #[rstest]
    #[case::azure(
        CredentialFamily::Azure,
        &[
            "azure_ad_token",
            "tenant_id",
            "client_id",
            "client_secret",
            "azure_scope",
            "azure_authority_host",
            "azure_credential",
            "azure_federated_token_file",
            "enable_azure_ad_token_refresh",
        ]
    )]
    #[case::vertex(
        CredentialFamily::Vertex,
        &[
            "vertex_credentials",
            "vertex_ai_credentials",
            "vertex_project",
            "vertex_ai_project",
            "vertex_location",
            "vertex_ai_location",
        ]
    )]
    fn family_names_are_the_provider_credential_inputs(
        #[case] family: CredentialFamily,
        #[case] expected: &[&str],
    ) {
        let names: BTreeSet<&str> = family_names(family).collect();
        assert_eq!(names, expected.iter().copied().collect());
    }
}
