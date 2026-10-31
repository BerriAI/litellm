//! What the router is told: [`RouterConfig`], fixed for the router's life, and
//! [`RouterDynamicConfig`], a versioned snapshot read once per call. The router never reads
//! process globals.

use std::{collections::BTreeMap, time::Duration};

use serde::Deserialize;

use crate::{
    error::FailureKind,
    registry::{DeploymentSpec, RoutingGroupSpec},
};

/// `DEFAULT_COOLDOWN_TIME_SECONDS` in `litellm/constants.py`.
pub const DEFAULT_COOLDOWN_TIME: Duration = Duration::from_secs(5);
/// `ROUTER_MAX_FALLBACKS` in `litellm/constants.py`.
pub const DEFAULT_MAX_FALLBACKS: u32 = 5;

/// The router operation a call runs, spelled as the Python router's endpoint method.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    strum::EnumString,
    strum::Display,
    serde_with::DeserializeFromStr,
)]
pub enum Op {
    #[strum(serialize = "acompletion")]
    Completion,
    #[strum(serialize = "atext_completion")]
    TextCompletion,
    #[strum(serialize = "aembedding")]
    Embedding,
    #[strum(serialize = "aresponses")]
    Responses,
    #[strum(serialize = "aanthropic_messages")]
    Messages,
    #[strum(serialize = "aimage_generation")]
    ImageGeneration,
    #[strum(serialize = "atranscription")]
    Transcription,
    #[strum(serialize = "aspeech")]
    Speech,
    #[strum(serialize = "aocr")]
    Ocr,
    #[strum(default, transparent)]
    Other(String),
}

/// `routing_strategy`. Only simple-shuffle is native; the rest go through `Host::select`.
#[derive(
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    strum::EnumString,
    strum::Display,
    serde_with::DeserializeFromStr,
)]
pub enum RoutingStrategy {
    #[default]
    #[strum(serialize = "simple-shuffle")]
    SimpleShuffle,
    #[strum(serialize = "least-busy")]
    LeastBusy,
    #[strum(serialize = "usage-based-routing")]
    UsageBased,
    #[strum(serialize = "usage-based-routing-v2")]
    UsageBasedV2,
    #[strum(serialize = "latency-based-routing")]
    LatencyBased,
    #[strum(serialize = "cost-based-routing")]
    CostBased,
    #[strum(default, transparent)]
    Other(String),
}

impl RoutingStrategy {
    pub fn is_native(&self) -> bool {
        matches!(self, Self::SimpleShuffle)
    }
}

/// `PluginKind` in `litellm/router_api.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, strum::Display)]
pub enum PluginKind {
    #[serde(rename = "auto")]
    #[strum(serialize = "auto")]
    Auto,
    #[serde(rename = "complexity")]
    #[strum(serialize = "complexity")]
    Complexity,
    #[serde(rename = "quality")]
    #[strum(serialize = "quality")]
    Quality,
    #[serde(rename = "adaptive")]
    #[strum(serialize = "adaptive")]
    Adaptive,
    #[serde(rename = "routing")]
    #[strum(serialize = "routing")]
    Routing,
    #[serde(rename = "budget_limiter")]
    #[strum(serialize = "budget_limiter")]
    BudgetLimiter,
    #[serde(rename = "prompt_caching_pins")]
    #[strum(serialize = "prompt_caching_pins")]
    PromptCachingPins,
}

/// One registered plugin. Its behaviour lives in the host until it is ported.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct PluginSpec {
    pub kind: PluginKind,
    pub name: String,
    #[serde(default)]
    pub config: serde_json::Value,
}

/// `RetryPolicy` in `litellm/types/router.py`: retries per exception type.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[allow(non_snake_case)]
pub struct RetryPolicy {
    pub BadRequestErrorRetries: Option<u32>,
    pub AuthenticationErrorRetries: Option<u32>,
    pub TimeoutErrorRetries: Option<u32>,
    pub RateLimitErrorRetries: Option<u32>,
    pub ContentPolicyViolationErrorRetries: Option<u32>,
    pub InternalServerErrorRetries: Option<u32>,
}

impl RetryPolicy {
    pub fn retries_for(&self, kind: FailureKind) -> Option<u32> {
        match kind {
            FailureKind::BadRequest | FailureKind::ContextWindowExceeded => {
                self.BadRequestErrorRetries
            }
            FailureKind::Authentication => self.AuthenticationErrorRetries,
            FailureKind::Timeout => self.TimeoutErrorRetries,
            FailureKind::RateLimit => self.RateLimitErrorRetries,
            FailureKind::ContentPolicyViolation => self.ContentPolicyViolationErrorRetries,
            FailureKind::InternalServer => self.InternalServerErrorRetries,
            FailureKind::PermissionDenied
            | FailureKind::NotFound
            | FailureKind::ApiConnection
            | FailureKind::ServiceUnavailable
            | FailureKind::BadGateway
            | FailureKind::GuardrailIntervention
            | FailureKind::Declined
            | FailureKind::CallerCredential
            | FailureKind::Other => None,
        }
    }
}

/// `AllowedFailsPolicy` in `litellm/types/router.py`: allowed fails per minute per exception type.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[allow(non_snake_case)]
pub struct AllowedFailsPolicy {
    pub BadRequestErrorAllowedFails: Option<u32>,
    pub AuthenticationErrorAllowedFails: Option<u32>,
    pub TimeoutErrorAllowedFails: Option<u32>,
    pub RateLimitErrorAllowedFails: Option<u32>,
    pub ContentPolicyViolationErrorAllowedFails: Option<u32>,
    pub InternalServerErrorAllowedFails: Option<u32>,
    pub ServiceUnavailableErrorAllowedFails: Option<u32>,
    pub BadGatewayErrorAllowedFails: Option<u32>,
}

impl AllowedFailsPolicy {
    pub fn allowed_fails_for(&self, kind: FailureKind) -> Option<u32> {
        match kind {
            FailureKind::BadRequest | FailureKind::ContextWindowExceeded => {
                self.BadRequestErrorAllowedFails
            }
            FailureKind::Authentication => self.AuthenticationErrorAllowedFails,
            FailureKind::Timeout => self.TimeoutErrorAllowedFails,
            FailureKind::RateLimit => self.RateLimitErrorAllowedFails,
            FailureKind::ContentPolicyViolation => self.ContentPolicyViolationErrorAllowedFails,
            FailureKind::InternalServer => self.InternalServerErrorAllowedFails,
            FailureKind::ServiceUnavailable => self.ServiceUnavailableErrorAllowedFails,
            FailureKind::BadGateway => self.BadGatewayErrorAllowedFails,
            FailureKind::PermissionDenied
            | FailureKind::NotFound
            | FailureKind::ApiConnection
            | FailureKind::GuardrailIntervention
            | FailureKind::Declined
            | FailureKind::CallerCredential
            | FailureKind::Other => None,
        }
    }
}

/// A fallback list as Python spells it, `[{"gpt-4": ["gpt-3.5"]}, {"*": ["default"]}]`, kept in
/// order. `"*"` is the default entry.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(from = "Vec<BTreeMap<String, Vec<String>>>")]
pub struct Fallbacks(Vec<(String, Vec<String>)>);

impl From<Vec<BTreeMap<String, Vec<String>>>> for Fallbacks {
    fn from(entries: Vec<BTreeMap<String, Vec<String>>>) -> Self {
        Self(entries.into_iter().flatten().collect())
    }
}

impl Fallbacks {
    pub fn new(entries: impl IntoIterator<Item = (String, Vec<String>)>) -> Self {
        Self(entries.into_iter().collect())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// `_get_first_default_fallback`: the first target of the first `"*"` entry.
    pub fn default_target(&self) -> Option<&str> {
        self.0
            .iter()
            .find(|(key, targets)| key == "*" && !targets.is_empty())
            .map(|(_, targets)| targets[0].as_str())
    }

    /// `get_fallback_model_group`: the group's own entry, else an entry keyed by a
    /// provider-prefixed or provider-stripped spelling of it (the last such entry), else the
    /// last `"*"` entry before the group's own. The flag says the targets came from `"*"`.
    pub fn lookup(&self, group: &str, keys: &FallbackKeys) -> Option<(&[String], bool)> {
        let prefixed = keys.provider_prefixed.get(group).map(String::as_str);
        let mut spelled = None;
        let mut generic = None;
        for (key, targets) in &self.0 {
            if key == group {
                return Some((targets, false));
            }
            if Some(key.as_str()) == prefixed || keys.is_stripped(group, key) {
                spelled = Some(targets.as_slice());
            } else if key == "*" {
                generic = Some(targets.as_slice());
            }
        }
        spelled
            .map(|targets| (targets, false))
            .or_else(|| generic.map(|targets| (targets, true)))
    }

    /// `get_fallback_model_group_for_lookup_groups`: the first lookup group with its own
    /// chain wins; `"*"` applies only after every group missed.
    pub fn lookup_groups(&self, groups: &[&str], keys: &FallbackKeys) -> Option<Vec<String>> {
        let results: Vec<_> = groups
            .iter()
            .map(|group| self.lookup(group, keys))
            .collect();
        results
            .iter()
            .flatten()
            .find(|(_, generic)| !generic)
            .or_else(|| results.iter().flatten().next())
            .map(|(targets, _)| targets.to_vec())
    }
}

/// What fallback keys match besides the exact group name: Python's `provider_list`, for
/// `"openai/gpt-4o"` to use a `"gpt-4o"` entry, and the provider-prefixed spelling of each
/// bare group name (`_provider_prefixed_model_group`), for `"gpt-4o"` to use an
/// `"openai/gpt-4o"` entry.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct FallbackKeys {
    pub providers: Vec<String>,
    pub provider_prefixed: BTreeMap<String, String>,
}

impl FallbackKeys {
    /// `_check_stripped_model_group`.
    fn is_stripped(&self, group: &str, key: &str) -> bool {
        self.providers.iter().any(|provider| {
            let prefix = format!("{provider}/");
            group.starts_with(&prefix) && group.replace(&prefix, "") == key
        })
    }
}

/// Static config, fixed for the router's life.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct RouterConfig {
    pub model_list: Vec<DeploymentSpec>,
    pub routing_strategy: RoutingStrategy,
    pub routing_groups: Vec<RoutingGroupSpec>,
    pub plugins: Vec<PluginSpec>,
    /// Router default `cooldown_time`; `None` is `DEFAULT_COOLDOWN_TIME_SECONDS`.
    pub cooldown_time: Option<f64>,
    pub disable_cooldowns: bool,
    pub allowed_fails_policy: Option<AllowedFailsPolicy>,
    pub retry_policy: Option<RetryPolicy>,
    /// `retry_after`: the minimum backoff between retries, in seconds.
    pub retry_after: f64,
    pub timeout: Option<f64>,
    pub request_timeout: Option<f64>,
    pub drop_params: Option<bool>,
    pub enable_pre_call_checks: bool,
    pub enable_health_check_routing: bool,
    /// Deployment ids that are a provider's default and never cool down
    /// (`provider_default_deployment_ids`).
    pub provider_default_deployment_ids: Vec<String>,
    /// The operations this router must serve; with a host that cannot invoke, each must have a
    /// native executor for every configured provider or the router fails to build.
    pub served_ops: Vec<Op>,
}

/// Dynamic config: a versioned snapshot read once per call.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct RouterDynamicConfig {
    pub version: u64,
    /// `model_group_alias` as Python spells it: each alias maps to a group name, or to
    /// `{"model": <group>, "hidden": <bool>}`.
    #[serde(alias = "model_group_alias", deserialize_with = "alias_targets")]
    pub model_alias_map: BTreeMap<String, String>,
    pub fallbacks: Fallbacks,
    pub default_fallbacks: Vec<String>,
    pub context_window_fallbacks: Option<Fallbacks>,
    pub content_policy_fallbacks: Option<Fallbacks>,
    pub num_retries: Option<u32>,
    pub max_fallbacks: u32,
    /// `allowed_fails`. `Some` means it was set explicitly, which switches cooldowns from the
    /// failure-ratio rule to the allowed-fails counter (`_is_allowed_fails_set_on_router`).
    pub allowed_fails: Option<u32>,
    /// Other spellings a fallback key matches.
    pub fallback_keys: FallbackKeys,
}

/// One `model_group_alias` value: a group name, or `{"model": <group>, "hidden": <bool>}`.
#[derive(Deserialize)]
#[serde(untagged)]
enum AliasTarget {
    Group(String),
    Entry { model: String },
}

fn alias_targets<'de, D>(deserializer: D) -> Result<BTreeMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let aliases = BTreeMap::<String, AliasTarget>::deserialize(deserializer)?;
    Ok(aliases
        .into_iter()
        .map(|(alias, target)| match target {
            AliasTarget::Group(group) | AliasTarget::Entry { model: group } => (alias, group),
        })
        .collect())
}

impl Default for RouterDynamicConfig {
    fn default() -> Self {
        Self {
            version: 0,
            model_alias_map: BTreeMap::new(),
            fallbacks: Fallbacks::default(),
            default_fallbacks: Vec::new(),
            context_window_fallbacks: None,
            content_policy_fallbacks: None,
            num_retries: None,
            max_fallbacks: DEFAULT_MAX_FALLBACKS,
            allowed_fails: None,
            fallback_keys: FallbackKeys::default(),
        }
    }
}

impl RouterDynamicConfig {
    /// The regular fallback targets for the lookup groups (`get_fallback_model_group_for_lookup_groups`
    /// over `fallbacks`), then `default_fallbacks`, which Python folds into `fallbacks` as
    /// `{"*": [...]}`.
    pub fn fallback_targets(&self, groups: &[&str]) -> Option<Vec<String>> {
        self.fallbacks
            .lookup_groups(groups, &self.fallback_keys)
            .or_else(|| {
                (!self.default_fallbacks.is_empty()).then(|| self.default_fallbacks.clone())
            })
    }

    pub fn has_regular_fallbacks(&self) -> bool {
        !self.fallbacks.is_empty() || !self.default_fallbacks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::own_entry("gpt-4", Some(vec!["gpt-3.5".to_owned()]))]
    #[case::star_entry("other", Some(vec!["default".to_owned()]))]
    fn fallback_targets_prefer_the_groups_own_entry(
        #[case] group: &str,
        #[case] expected: Option<Vec<String>>,
    ) {
        let dynamic: RouterDynamicConfig = serde_json::from_value(serde_json::json!({
            "fallbacks": [{"gpt-4": ["gpt-3.5"]}, {"*": ["default"]}]
        }))
        .unwrap();
        assert_eq!(dynamic.fallback_targets(&[group]), expected);
    }

    #[rstest]
    #[case::group_name(json!({"model_group_alias": {"gpt": "gpt-4"}}))]
    #[case::hidden_entry(json!({"model_group_alias": {"gpt": {"model": "gpt-4", "hidden": true}}}))]
    #[case::native_name(json!({"model_alias_map": {"gpt": "gpt-4"}}))]
    fn model_group_alias_reads_both_python_spellings(#[case] wire: serde_json::Value) {
        let dynamic: RouterDynamicConfig = serde_json::from_value(wire).unwrap();
        assert_eq!(
            dynamic.model_alias_map,
            BTreeMap::from([("gpt".to_owned(), "gpt-4".to_owned())])
        );
    }

    #[rstest]
    #[case::native("simple-shuffle", RoutingStrategy::SimpleShuffle)]
    #[case::unknown("my-strategy", RoutingStrategy::Other("my-strategy".to_owned()))]
    fn routing_strategy_parses_its_wire_name(
        #[case] wire: &str,
        #[case] expected: RoutingStrategy,
    ) {
        assert_eq!(wire.parse::<RoutingStrategy>().unwrap(), expected);
        assert_eq!(expected.to_string(), wire);
    }
}
