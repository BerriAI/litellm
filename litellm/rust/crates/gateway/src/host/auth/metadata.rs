//! The `metadata` of a key or a team that the native app reads, an allowlist both rows share and a
//! team's extends with its default member budget.

use std::collections::BTreeMap;

use litellm_gateway_auth_core::record::TeamPolicyMetadata;
use litellm_python_compat::{Value, json};
use litellm_redis_store::auth_cache::Bag;
use serde::{Deserialize, de::DeserializeOwned};

/// The `metadata` the native app honors: the row's own limit for each model it names one for,
/// which `/key/generate` and `/team/new` store from `model_rpm_limit` and `model_tpm_limit`, and
/// the priority class `dynamic_rate_limiter_v3` counts the row under, when it is written as text.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Metadata {
    #[serde(default)]
    pub(super) model_rpm_limit: BTreeMap<String, i64>,
    #[serde(default)]
    pub(super) model_tpm_limit: BTreeMap<String, i64>,
    pub(super) priority: Option<String>,
}

impl Metadata {
    /// `None` when `metadata` holds anything else, or a limit that is not a whole number.
    pub(super) fn read(metadata: &Bag) -> Option<Self> {
        allowlisted(metadata)
    }
}

/// A team's `metadata`: what a key's may hold, and the budget row `/team/new` names as the default
/// budget of a member whose membership sets none.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TeamMetadata {
    #[serde(default)]
    pub(super) model_rpm_limit: BTreeMap<String, i64>,
    #[serde(default)]
    pub(super) model_tpm_limit: BTreeMap<String, i64>,
    pub(super) priority: Option<String>,
    pub(super) team_member_budget_id: Option<String>,
    #[serde(flatten)]
    pub(super) policies: TeamPolicyMetadata,
}

impl TeamMetadata {
    /// `None` when `metadata` holds anything else, or a value of the wrong type.
    pub(super) fn read(metadata: &Bag) -> Option<Self> {
        allowlisted(metadata)
    }
}

fn allowlisted<T: Default + DeserializeOwned>(metadata: &Bag) -> Option<T> {
    if metadata.is_empty() {
        return Some(T::default());
    }
    let text = json::dumps(&Value::Dict(metadata.entries().to_vec())).ok()?;
    serde_json::from_str(&text).ok()
}

pub(super) fn policies<T: Default + DeserializeOwned + PartialEq>(
    metadata: Option<&Bag>,
) -> Result<Option<Box<T>>, &'static str> {
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    let parsed = allowlisted::<T>(metadata).ok_or("metadata")?;
    Ok((parsed != T::default()).then(|| Box::new(parsed)))
}
