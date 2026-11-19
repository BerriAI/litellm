use std::{collections::BTreeMap, fmt, ops::Deref};

use litellm_auth_types::{AwsParams, ParamSpec, VertexParams};
use litellm_secret_value::SecretValue;

use serde::Deserialize;
use serde_json::Value;
use serde_with::{OneOrMany, formats::PreferMany, serde_as};

use crate::Spelled;

/// Python's `LiteLLM_Params`: a deployment's `litellm_params`, as a config spells them and as
/// a caller passes them. Each credential family is one flattened group owned by its auth
/// crate, so a config reaches for `litellm_params.aws` or `litellm_params.vertex`, never a
/// map. Every key no field names lands in [`Self::extra`], as Python's `extra="allow"` keeps it.
///
/// A key a field names but cannot hold is an error, as it is for Python's pydantic model;
/// the coercions Python's validators apply (`max_retries` from text, `drop_params` from a flag
/// word) are left to the readers, so the shapes here are the ones the YAML carries. An
/// `organization` list is what Python's router expands into one deployment per entry.
#[serde_as]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct LitellmParams {
    pub model: String,
    pub api_key: Option<SecretValue>,
    pub api_base: Option<String>,
    pub api_version: Option<String>,
    #[serde(flatten)]
    pub aws: AwsParams,
    #[serde(flatten)]
    pub vertex: VertexParams,
    pub custom_llm_provider: Option<String>,
    pub timeout: Option<Spelled<f64>>,
    pub stream_timeout: Option<Spelled<f64>>,
    pub max_retries: Option<Spelled<f64>>,
    pub tpm: Option<Spelled<f64>>,
    pub rpm: Option<Spelled<f64>>,
    pub itpm: Option<Spelled<f64>>,
    pub otpm: Option<Spelled<f64>>,
    pub max_parallel_requests: Option<u64>,
    #[serde_as(as = "Option<OneOrMany<_, PreferMany>>")]
    pub organization: Option<Vec<String>>,
    pub drop_params: Option<Spelled<bool>>,
    pub tags: Option<Box<[String]>>,
    pub tag_regex: Option<Box<[String]>>,
    pub max_budget: Option<f64>,
    pub budget_duration: Option<String>,
    pub default_api_key_tpm_limit: Option<u64>,
    pub default_api_key_rpm_limit: Option<u64>,
    pub use_in_pass_through: Option<bool>,
    pub use_chat_completions_api: Option<bool>,
    pub litellm_credential_name: Option<String>,
    pub provider_affinity_header: Option<String>,
    /// The Bedrock Mantle workspace a call is billed to, sent as `anthropic-workspace-id`.
    pub aws_bedrock_project_id: Option<String>,
    #[serde(flatten)]
    pub claude_platform: ClaudePlatformParams,
    #[serde(flatten)]
    pub extra: ExtraParams,
}

impl LitellmParams {
    /// Every connection param spec the credential families declare, for hosts that fold a
    /// spec's module global in when a call names none of its spellings.
    pub fn specs() -> impl Iterator<Item = &'static ParamSpec> {
        AwsParams::SPECS.iter().chain(VertexParams::SPECS.iter())
    }

    /// The wire names a host projects out of a caller's kwargs: the model, the credentials
    /// and each credential family. The routing and limit fields are a deployment's
    /// settings, which only a config spells.
    pub fn fields() -> impl Iterator<Item = &'static str> {
        [
            "model",
            "api_key",
            "api_base",
            "api_version",
            "aws_bedrock_project_id",
        ]
        .into_iter()
        .chain(ClaudePlatformParams::FIELDS)
        .chain(AwsParams::fields())
        .chain(VertexParams::fields())
    }
}

/// The Claude Platform on AWS settings a `bedrock/claude_platform/` call carries: the workspace,
/// under any of the four spellings Python's config reads, and the body params the endpoint rejects.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct ClaudePlatformParams {
    pub workspace_id: Option<WorkspaceId>,
    pub aws_workspace_id: Option<WorkspaceId>,
    #[serde(rename = "anthropic-workspace-id")]
    pub anthropic_workspace_header: Option<WorkspaceId>,
    pub anthropic_workspace_id: Option<WorkspaceId>,
    pub claude_platform_unsupported_params: Option<UnsupportedParams>,
}

impl ClaudePlatformParams {
    const FIELDS: [&'static str; 5] = [
        "workspace_id",
        "aws_workspace_id",
        "anthropic-workspace-id",
        "anthropic_workspace_id",
        "claude_platform_unsupported_params",
    ];

    /// The workspace in Python's lookup order, skipping empty spellings.
    pub fn workspace(&self) -> Option<String> {
        [
            &self.workspace_id,
            &self.aws_workspace_id,
            &self.anthropic_workspace_header,
            &self.anthropic_workspace_id,
        ]
        .into_iter()
        .find_map(|workspace| workspace.as_ref()?.text())
    }

    /// The body params to drop instead of the endpoint's default set, when a list names them.
    pub fn unsupported_params(&self) -> Option<&[String]> {
        match self.claude_platform_unsupported_params.as_ref()? {
            UnsupportedParams::Names(names) => Some(names),
            _ => None,
        }
    }

    /// The Python type name of an override that is not a list, which Python warns it ignores.
    pub fn ignored_override_type(&self) -> Option<&'static str> {
        match self.claude_platform_unsupported_params.as_ref()? {
            UnsupportedParams::Names(_) | UnsupportedParams::MixedList(_) => None,
            UnsupportedParams::Flag(_) => Some("bool"),
            UnsupportedParams::Integer(_) => Some("int"),
            UnsupportedParams::Float(_) => Some("float"),
            UnsupportedParams::Text(_) => Some("str"),
            UnsupportedParams::Mapping(_) => Some("dict"),
        }
    }
}

/// A workspace id as a config writes it: text, or a bare number YAML reads as one.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum WorkspaceId {
    Text(String),
    Number(serde_json::Number),
}

impl WorkspaceId {
    fn text(&self) -> Option<String> {
        match self {
            Self::Text(text) if text.is_empty() => None,
            Self::Text(text) => Some(text.clone()),
            Self::Number(number) => Some(number.to_string()),
        }
    }
}

/// `claude_platform_unsupported_params`: a list of names, or any other value, which Python
/// ignores in favour of the default set.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum UnsupportedParams {
    Names(Vec<String>),
    MixedList(Vec<serde::de::IgnoredAny>),
    Flag(bool),
    Integer(i64),
    Float(f64),
    Text(String),
    Mapping(serde::de::IgnoredAny),
}

/// The `litellm_params` keys no field names, kept as written. Its `Debug` lists the keys
/// only, since a provider's secret may sit among the values.
#[derive(Clone, Default, PartialEq, Deserialize)]
#[serde(transparent)]
pub struct ExtraParams(BTreeMap<String, Value>);

impl Deref for ExtraParams {
    type Target = BTreeMap<String, Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl FromIterator<(String, Value)> for ExtraParams {
    fn from_iter<I: IntoIterator<Item = (String, Value)>>(entries: I) -> Self {
        Self(entries.into_iter().collect())
    }
}

impl fmt::Debug for ExtraParams {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.0.keys()).finish()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn platform(fields: Value) -> ClaudePlatformParams {
        let mut params = json!({"model": "bedrock/claude_platform/claude-test"});
        params
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value::<LitellmParams>(params)
            .unwrap()
            .claude_platform
    }

    #[rstest]
    #[case::none(json!({}), None)]
    #[case::text(json!({"workspace_id": "w"}), Some("w"))]
    #[case::number(json!({"aws_workspace_id": 42}), Some("42"))]
    #[case::header_spelling(json!({"anthropic-workspace-id": "h"}), Some("h"))]
    #[case::empty_skipped(json!({"workspace_id": "", "anthropic_workspace_id": "u"}), Some("u"))]
    fn the_workspace_is_read_from_any_spelling(
        #[case] fields: Value,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(platform(fields).workspace().as_deref(), expected);
    }

    #[rstest]
    #[case::absent(json!({}), None)]
    #[case::names(json!({"claude_platform_unsupported_params": ["a", "b"]}), Some(vec!["a", "b"]))]
    #[case::empty_list(json!({"claude_platform_unsupported_params": []}), Some(vec![]))]
    #[case::text_ignored(json!({"claude_platform_unsupported_params": "a"}), None)]
    #[case::mixed_list_ignored(json!({"claude_platform_unsupported_params": ["a", 1]}), None)]
    fn only_a_list_of_names_overrides_the_rejected_params(
        #[case] fields: Value,
        #[case] expected: Option<Vec<&str>>,
    ) {
        let params = platform(fields);
        assert_eq!(
            params
                .unsupported_params()
                .map(|names| names.iter().map(String::as_str).collect::<Vec<_>>()),
            expected
        );
    }

    #[rstest]
    #[case::list(json!({"claude_platform_unsupported_params": ["a"]}), None)]
    #[case::mixed_list(json!({"claude_platform_unsupported_params": ["a", 1]}), None)]
    #[case::text(json!({"claude_platform_unsupported_params": "a"}), Some("str"))]
    #[case::integer(json!({"claude_platform_unsupported_params": 1}), Some("int"))]
    #[case::float(json!({"claude_platform_unsupported_params": 1.5}), Some("float"))]
    #[case::flag(json!({"claude_platform_unsupported_params": true}), Some("bool"))]
    #[case::mapping(json!({"claude_platform_unsupported_params": {"a": 1}}), Some("dict"))]
    fn an_ignored_override_is_named_by_its_python_type(
        #[case] fields: Value,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(platform(fields).ignored_override_type(), expected);
    }

    #[rstest]
    fn the_workspace_keys_stay_out_of_the_extra_params() {
        let params: LitellmParams = serde_json::from_value(json!({
            "model": "m",
            "workspace_id": "w",
            "aws_workspace_id": "a",
            "aws_region_name": "us-east-1",
        }))
        .unwrap();
        assert!(params.extra.is_empty(), "{:?}", params.extra);
        assert_eq!(params.aws.aws_region_name.as_deref(), Some("us-east-1"));
    }
}
