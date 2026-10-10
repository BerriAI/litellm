use std::{collections::BTreeMap, fmt, ops::Deref};

use litellm_auth_types::{AwsParams, ParamSpec, SecretValue, VertexParams};
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
    #[serde(skip)]
    pub github_copilot_session: Option<GithubCopilotSession>,
    #[serde(flatten)]
    pub extra: ExtraParams,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GithubCopilotSession {
    pub token: SecretValue,
    pub api_base: String,
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
        ["model", "api_key", "api_base", "api_version"]
            .into_iter()
            .chain(AwsParams::fields())
            .chain(VertexParams::fields())
    }
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

impl fmt::Debug for ExtraParams {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.0.keys()).finish()
    }
}
