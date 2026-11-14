//! Health-check-driven routing: the background health checks' results, read from the host's
//! [`RouterCache`](crate::cache::RouterCache) as Python's `DeploymentHealthCache` writes them, and
//! the filter `_async_filter_health_check_unhealthy_deployments` /
//! `_filter_health_check_unhealthy_deployments` run before the cooldown filter.

use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
    time::UNIX_EPOCH,
};

use crate::{cache::HealthState, health::Health, registry::Deployment, settings::Settings};

/// `_extract_unhealthy_ids`: the ids marked unhealthy whose result is younger than `staleness`.
pub fn unhealthy_ids(
    states: &HashMap<String, HealthState>,
    now: f64,
    staleness: f64,
) -> BTreeSet<String> {
    states
        .iter()
        .filter(|(_, state)| !state.is_healthy && now - state.timestamp < staleness)
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
        let states = self.cache().health_states().await.unwrap_or_default();
        let now = self
            .now_time()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs_f64())
            .unwrap_or_default();
        let unhealthy = unhealthy_ids(&states, now, settings.health_check_staleness.as_secs_f64());
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

    use super::*;

    #[rstest]
    #[case::fresh_unhealthy(false, 95.0, vec!["a"])]
    #[case::stale(false, 50.0, vec![])]
    #[case::healthy(true, 95.0, vec![])]
    fn unhealthy_ids_follow_python(
        #[case] is_healthy: bool,
        #[case] timestamp: f64,
        #[case] expected: Vec<&str>,
    ) {
        let state = HealthState {
            is_healthy,
            timestamp,
        };
        let states = HashMap::from([("a".to_owned(), state)]);
        let ids = unhealthy_ids(&states, 100.0, 10.0);
        assert_eq!(ids.iter().map(String::as_str).collect::<Vec<_>>(), expected);
    }
}
