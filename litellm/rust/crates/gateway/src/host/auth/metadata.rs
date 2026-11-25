//! The `metadata` of a key or a team that the native app reads, an allowlist both rows share.

use std::collections::BTreeMap;

use litellm_python_compat::{Value, json};
use litellm_redis_store::auth_cache::Bag;
use serde::Deserialize;
use serde_json::Map;

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

/// `metadata` as a JSON object in the order it was written, which the proxy's
/// `strip_callback_config` leaves whole for a row whose [`Metadata`] was read.
pub(super) fn logged(metadata: &Bag) -> Map<String, serde_json::Value> {
    match json::to_json(&Value::Dict(metadata.entries().to_vec())) {
        Ok(serde_json::Value::Object(entries)) => entries,
        _ => Map::new(),
    }
}
