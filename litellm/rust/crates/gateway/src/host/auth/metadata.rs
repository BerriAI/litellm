//! The `metadata` of a key or a team that the native app reads, an allowlist both rows share.

use std::collections::BTreeMap;

use litellm_python_compat::{Value, json};
use litellm_redis_store::auth_cache::Bag;
use serde::Deserialize;

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
        if metadata.is_empty() {
            return Some(Self::default());
        }
        let text = json::dumps(&Value::Dict(metadata.entries().to_vec())).ok()?;
        serde_json::from_str(&text).ok()
    }
}
