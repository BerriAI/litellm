//! Selection: resolve a name, filter its deployments in the order of
//! `Router._core_async_get_healthy_deployments` (team, cooldown, blocked, plugins, order, retry
//! skips), then pick one with the strategy. Simple-shuffle is native
//! (`router_strategy/simple_shuffle.py`); other strategies go through `Host::select`.

use std::{collections::BTreeMap, sync::Arc};

use parking_lot::Mutex;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    Core, Snapshot,
    config::{Op, RoutingStrategy},
    error::Error,
    host::{RouterRequest, RoutingContext, SelectStage},
    registry::{Deployment, ResolveKind, ResolveScope},
};

/// Randomness for the shuffle and retry jitter, injectable so tests are deterministic.
pub trait Randomness: Send + Sync {
    /// A float in `[0, 1)`, as Python's `random.random()`.
    fn next_f64(&self) -> f64;
}

pub struct ThreadRandomness;

impl Randomness for ThreadRandomness {
    fn next_f64(&self) -> f64 {
        rand::thread_rng().r#gen()
    }
}

pub struct SeededRandomness(Mutex<StdRng>);

impl SeededRandomness {
    pub fn new(seed: u64) -> Self {
        Self(Mutex::new(StdRng::seed_from_u64(seed)))
    }
}

impl Randomness for SeededRandomness {
    fn next_f64(&self) -> f64 {
        self.0.lock().r#gen()
    }
}

/// A group's deployments before and after filtering.
#[derive(Clone, Debug)]
pub struct Candidates {
    pub model_group: String,
    pub kind: ResolveKind,
    /// Every deployment the caller may use (after the team filter).
    pub all: Vec<Arc<Deployment>>,
    pub healthy: Vec<Arc<Deployment>>,
}

/// Per-attempt selection inputs the call loop carries.
#[derive(Clone, Debug, Default)]
pub(crate) struct SelectInputs {
    pub target_order: Option<i64>,
    pub skipped_ids: Vec<String>,
    /// Stop before the plugins, order and skip filters: the group less cooldowns and blocked
    /// deployments, which `_async_get_healthy_deployments` counts when deciding whether to retry.
    pub count_only: bool,
}

/// `simple_shuffle`: weighted by `_router_weights[group]`, else `weight`, else `rpm`, else
/// `tpm`, each normalized by its maximum, else uniform.
pub fn simple_shuffle(
    deployments: &[Arc<Deployment>],
    model_group: &str,
    router_weights: Option<&BTreeMap<String, BTreeMap<String, f64>>>,
    rng: &dyn Randomness,
) -> Option<Arc<Deployment>> {
    if deployments.is_empty() {
        return None;
    }
    let scoped = router_weights
        .and_then(|weights| weights.get(model_group))
        .filter(|weights| !weights.is_empty())
        .map(|weights| {
            deployments
                .iter()
                .map(|d| weights.get(&d.id).copied().unwrap_or(0.0))
                .collect::<Vec<_>>()
        });
    let metric_sets = ["weight", "rpm", "tpm"].map(|metric| {
        deployments
            .iter()
            .map(|d| d.metric(metric).unwrap_or(0.0))
            .collect::<Vec<_>>()
    });
    for weights in scoped.into_iter().chain(metric_sets) {
        let largest = weights.iter().copied().fold(0.0, f64::max);
        if largest <= 0.0 {
            continue;
        }
        let normalized: Vec<f64> = weights.iter().map(|weight| weight / largest).collect();
        let total: f64 = normalized.iter().sum();
        if total <= 0.0 {
            continue;
        }
        // `random.choices`: bisect_right over the cumulative weights.
        let target = rng.next_f64() * total;
        let mut cumulative = 0.0;
        let index = normalized
            .iter()
            .position(|weight| {
                cumulative += weight;
                target < cumulative
            })
            .unwrap_or(deployments.len() - 1);
        return Some(deployments[index].clone());
    }
    let index = ((rng.next_f64() * deployments.len() as f64) as usize).min(deployments.len() - 1);
    Some(deployments[index].clone())
}

impl Core {
    fn context<'a>(
        &'a self,
        snapshot: &'a Snapshot,
        op: &'a Op,
        request: &'a RouterRequest,
        model_group: &'a str,
    ) -> RoutingContext<'a> {
        RoutingContext {
            registry: &snapshot.registry,
            settings: &snapshot.settings,
            health: &self.health,
            request,
            op,
            model_group,
        }
    }

    /// The eligible deployments for `name`, in filter-chain order.
    pub(crate) async fn candidates(
        &self,
        snapshot: &Snapshot,
        op: &Op,
        request: &RouterRequest,
        name: &str,
        inputs: &SelectInputs,
    ) -> Result<Candidates, Error> {
        let registry = &snapshot.registry;
        let scope = ResolveScope {
            specific_deployment: request.specific_deployment,
        };
        let dynamic = &snapshot.settings.dynamic;
        let resolution = registry
            .resolve(name, scope)
            .or_else(|| {
                // No deployment for the name: route to the first default fallback.
                let default = dynamic
                    .fallbacks
                    .default_target()
                    .or_else(|| dynamic.default_fallbacks.first().map(String::as_str))?;
                registry.resolve(default, scope)
            })
            .ok_or_else(|| Error::UnknownModel {
                model: name.to_owned(),
            })?;
        let team_id = request.team_id.as_deref();
        let all: Vec<Arc<Deployment>> = resolution
            .deployments
            .into_iter()
            .filter(|d| d.usable_by_team(team_id))
            .collect();

        let ids: Vec<String> = all.iter().map(|d| d.id.clone()).collect();
        let cooling = self.health.active_cooldowns(&ids).await?;
        let mut healthy: Vec<Arc<Deployment>> = all
            .iter()
            .filter(|d| !cooling.iter().any(|(id, _)| *id == d.id))
            .cloned()
            .collect();
        let settings = &snapshot.settings;
        if healthy.is_empty()
            && settings.enable_health_check_routing
            && settings.allowed_fails_policy.is_some()
        {
            healthy = all.clone();
        }
        healthy.retain(|d| !d.model_info.blocked);
        if inputs.count_only {
            return Ok(Candidates {
                model_group: resolution.group,
                kind: resolution.kind,
                all,
                healthy,
            });
        }

        if !settings.plugins.is_empty() && !healthy.is_empty() {
            let ctx = self.context(snapshot, op, request, &resolution.group);
            healthy = self
                .host
                .select(&SelectStage::Plugins, &ctx, healthy)
                .await?;
        }

        let order = inputs
            .target_order
            .or_else(|| healthy.iter().filter_map(|d| d.order()).min());
        if let Some(order) = order {
            healthy.retain(|d| d.order() == Some(order));
        }

        if !inputs.skipped_ids.is_empty() {
            let kept: Vec<Arc<Deployment>> = healthy
                .iter()
                .filter(|d| !inputs.skipped_ids.contains(&d.id))
                .cloned()
                .collect();
            if !kept.is_empty() {
                healthy = kept;
            }
        }

        Ok(Candidates {
            model_group: resolution.group,
            kind: resolution.kind,
            all,
            healthy,
        })
    }

    /// Pick one of `candidates.healthy` with the group's strategy.
    pub(crate) async fn pick(
        &self,
        snapshot: &Snapshot,
        op: &Op,
        request: &RouterRequest,
        candidates: &Candidates,
    ) -> Result<Arc<Deployment>, Error> {
        let no_deployment = || Error::NoHealthyDeployments {
            model: candidates.model_group.clone(),
        };
        let strategy = match candidates.kind {
            ResolveKind::RoutingGroup => snapshot
                .registry
                .resolve_routing_group_strategy(&candidates.model_group),
            ResolveKind::SpecificDeployment
            | ResolveKind::Id
            | ResolveKind::Alias
            | ResolveKind::Group
            | ResolveKind::Wildcard => None,
        }
        .unwrap_or_else(|| snapshot.settings.routing_strategy.clone());
        if strategy == RoutingStrategy::SimpleShuffle {
            return simple_shuffle(
                &candidates.healthy,
                &candidates.model_group,
                request.router_weights.as_ref(),
                self.rng.as_ref(),
            )
            .ok_or_else(no_deployment);
        }
        let ctx = self.context(snapshot, op, request, &candidates.model_group);
        self.host
            .select(
                &SelectStage::Strategy(strategy),
                &ctx,
                candidates.healthy.clone(),
            )
            .await?
            .into_iter()
            .next()
            .ok_or_else(no_deployment)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::registry::DeploymentSpec;

    struct Fixed(f64);

    impl Randomness for Fixed {
        fn next_f64(&self) -> f64 {
            self.0
        }
    }

    fn deployment(id: &str, params: serde_json::Value) -> Arc<Deployment> {
        let spec: DeploymentSpec = serde_json::from_value(json!({
            "model_name": "g", "litellm_params": params, "model_info": {"id": id},
        }))
        .unwrap();
        Arc::new(Deployment::from_spec(spec).unwrap())
    }

    #[rstest]
    #[case::weight_low_draw(0.1, "a")]
    #[case::weight_high_draw(0.9, "b")]
    fn shuffle_uses_weight_before_rpm(#[case] draw: f64, #[case] expected: &str) {
        let deployments = [
            deployment("a", json!({"model": "openai/a", "weight": 1, "rpm": 1000})),
            deployment("b", json!({"model": "openai/b", "weight": 3})),
        ];
        let picked = simple_shuffle(&deployments, "g", None, &Fixed(draw)).unwrap();
        assert_eq!(picked.id, expected);
    }

    #[rstest]
    fn shuffle_prefers_request_weights() {
        let deployments = [
            deployment("a", json!({"model": "openai/a", "weight": 100})),
            deployment("b", json!({"model": "openai/b"})),
        ];
        let weights = BTreeMap::from([("g".to_owned(), BTreeMap::from([("b".to_owned(), 1.0)]))]);
        let picked = simple_shuffle(&deployments, "g", Some(&weights), &Fixed(0.99)).unwrap();
        assert_eq!(picked.id, "b");
    }
}
