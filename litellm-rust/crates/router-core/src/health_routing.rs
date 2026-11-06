//! Health-check-driven routing: the background health checks' results, kept in the shared
//! [`RouterCache`](crate::cache::RouterCache) under one key as Python's `DeploymentHealthCache`
//! writes them, and the filter `_async_filter_health_check_unhealthy_deployments` /
//! `_filter_health_check_unhealthy_deployments` run before the cooldown filter.

use std::{collections::BTreeSet, sync::Arc, time::UNIX_EPOCH};

use serde_json::Value;

use crate::{health::Health, registry::Deployment, settings::Settings};

/// `DeploymentHealthCache.CACHE_KEY`.
pub const HEALTH_STATE_KEY: &str = "litellm:health_check:deployment_health_state";

/// `_extract_unhealthy_ids`: the ids marked unhealthy whose result is younger than `staleness`.
pub fn unhealthy_ids(raw: Option<&Value>, now: f64, staleness: f64) -> BTreeSet<String> {
    let Some(Value::Object(states)) = raw else {
        return BTreeSet::new();
    };
    states
        .iter()
        .filter(|(_, state)| {
            let Value::Object(state) = state else {
                return false;
            };
            // `not state.get("is_healthy", True)`: any falsy value is unhealthy.
            let unhealthy = match state.get("is_healthy") {
                None => false,
                Some(Value::Null) => true,
                Some(Value::Bool(value)) => !value,
                Some(Value::Number(value)) => value.as_f64() == Some(0.0),
                Some(Value::String(value)) => value.is_empty(),
                Some(Value::Array(value)) => value.is_empty(),
                Some(Value::Object(value)) => value.is_empty(),
            };
            let timestamp = state
                .get("timestamp")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            unhealthy && now - timestamp < staleness
        })
        .map(|(id, _)| id.clone())
        .collect()
}

/// What the filter does when it would leave nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhenAllUnhealthy {
    /// The safety net: every candidate is kept.
    KeepAll,
    /// A health check probe sees no deployment.
    Nothing,
}

impl Health {
    /// The deployments the background health checks have not marked unhealthy, as the
    /// router's health-check routing settings say.
    pub async fn health_check_filter(
        &self,
        settings: &Settings,
        deployments: Vec<Arc<Deployment>>,
        when_all_unhealthy: WhenAllUnhealthy,
    ) -> Vec<Arc<Deployment>> {
        if !settings.enable_health_check_routing {
            return deployments;
        }
        let scoped = settings.background_health_check_model_groups.as_deref();
        // With an allowed_fails_policy cooldowns are the only exclusion, unless the filter is
        // scoped to listed groups.
        if settings.allowed_fails_policy.is_some() && scoped.is_none() {
            return deployments;
        }
        // A read that fails counts as no health state.
        let raw = self
            .cache()
            .async_get_cache(HEALTH_STATE_KEY)
            .await
            .ok()
            .flatten();
        let now = self
            .now_time()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs_f64())
            .unwrap_or_default();
        let unhealthy = unhealthy_ids(
            raw.as_ref(),
            now,
            settings.health_check_staleness.as_secs_f64(),
        );
        if unhealthy.is_empty() {
            return deployments;
        }
        let filtered: Vec<Arc<Deployment>> = deployments
            .iter()
            .filter(|d| {
                !unhealthy.contains(&d.id)
                    || scoped.is_some_and(|groups| !groups.contains(&d.model_name))
            })
            .cloned()
            .collect();
        match (filtered.is_empty(), when_all_unhealthy) {
            (false, _) => filtered,
            (true, WhenAllUnhealthy::KeepAll) => deployments,
            (true, WhenAllUnhealthy::Nothing) => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::fresh_unhealthy(json!({"a": {"is_healthy": false, "timestamp": 95.0}}), vec!["a"])]
    #[case::stale(json!({"a": {"is_healthy": false, "timestamp": 50.0}}), vec![])]
    #[case::healthy(json!({"a": {"is_healthy": true, "timestamp": 95.0}}), vec![])]
    #[case::no_flag(json!({"a": {"timestamp": 95.0}}), vec![])]
    #[case::not_a_dict(json!({"a": 1}), vec![])]
    fn unhealthy_ids_follow_python(#[case] raw: Value, #[case] expected: Vec<&str>) {
        let ids = unhealthy_ids(Some(&raw), 100.0, 10.0);
        assert_eq!(ids.iter().map(String::as_str).collect::<Vec<_>>(), expected);
    }
}
