//! Selection: resolve a name, filter its deployments in the order of
//! `Router._core_async_get_healthy_deployments` (team, cooldown, blocked, plugins, pre-call
//! checks, tags, order, retry skips), then pick one with the strategy. Simple-shuffle is native
//! (`router_strategy/simple_shuffle.py`); other strategies go through `Host::select`.

use std::{collections::BTreeMap, sync::Arc};

use parking_lot::Mutex;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    Core, Snapshot,
    config::{Op, RoutingStrategy},
    error::Error,
    health_routing::WhenAllUnhealthy,
    host::{RouterRequest, RoutingContext, SelectStage},
    registry::{Deployment, ResolveKind},
    resolve::{Caller, resolve_request},
    strategy::{StrategyScope, routing_context},
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

/// Which pass-through pick a request makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassThrough {
    /// `get_available_deployment_for_pass_through` (and `select(kind="pass_through")`): the
    /// pass-through deployments are chosen first, then cooldown and blocked filters, and only a
    /// priority group filters by order.
    Listed,
    /// `async_get_available_deployment_for_pass_through`: the full healthy chain, then the
    /// pass-through filter.
    Healthy,
}

/// A group's deployments before and after filtering.
#[derive(Clone, Debug)]
pub struct Candidates {
    /// The name that was resolved (Python's `model`).
    pub name: String,
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
    /// A weighted failover hop's `_excluded_deployment_ids`: never picked.
    pub excluded_ids: Arc<[String]>,
    /// Stop before the plugins, order and skip filters: the group less cooldowns and blocked
    /// deployments, which `_async_get_healthy_deployments` counts when deciding whether to retry.
    pub count_only: bool,
}

/// `simple_shuffle`: weighted by `_router_weights[group]`, else `weight`, else `rpm`, else
/// `tpm`, each normalized by its maximum, else uniform. A metric is read only when every set
/// before it weighed nothing, and a non-numeric one fails the pick then (`_metric_weight`).
pub fn simple_shuffle(
    deployments: &[Arc<Deployment>],
    model_group: &str,
    router_weights: Option<&BTreeMap<String, BTreeMap<String, f64>>>,
    rng: &dyn Randomness,
) -> Result<Option<Arc<Deployment>>, Error> {
    if deployments.is_empty() {
        return Ok(None);
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
    let metric_sets = ["weight", "rpm", "tpm"].into_iter().map(|metric| {
        deployments
            .iter()
            .map(|d| d.metric_weight(metric))
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::NonNumericWeight { metric })
    });
    for weights in scoped.into_iter().map(Ok).chain(metric_sets) {
        let weights = weights?;
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
        return Ok(Some(deployments[index].clone()));
    }
    let index = ((rng.next_f64() * deployments.len() as f64) as usize).min(deployments.len() - 1);
    Ok(Some(deployments[index].clone()))
}

/// A pre-routing strategy marker (`classify_strategy_router_model`): its model is `auto_router/...`.
pub fn is_strategy_marker(deployment: &Deployment) -> bool {
    deployment.params.model.starts_with("auto_router/")
}

/// `litellm_params.use_in_pass_through`, read for truthiness as Python does.
fn uses_pass_through(deployment: &Deployment) -> bool {
    match deployment.raw_params.get("use_in_pass_through") {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(value)) => *value,
        Some(serde_json::Value::Number(value)) => value.as_f64().is_some_and(|n| n != 0.0),
        Some(serde_json::Value::String(value)) => !value.is_empty(),
        Some(serde_json::Value::Array(value)) => !value.is_empty(),
        Some(serde_json::Value::Object(value)) => !value.is_empty(),
    }
}

/// `filter_team_based_models`: a team-owned deployment serves only its team, except that a proxy
/// admin without a team keeps deployments that are all team-owned and listed under the model the
/// request names, unless they belong to several teams.
fn filter_team_based(
    deployments: Vec<Arc<Deployment>>,
    request: &RouterRequest,
) -> Result<Vec<Arc<Deployment>>, Error> {
    let team_id = request.team_id.as_deref();
    if team_id.is_none()
        && request.proxy_admin
        && let Some(requested) = request.admin_model.as_deref()
        && !deployments.is_empty()
        && deployments.iter().all(|d| {
            d.model_info.team_id.is_some()
                && (d.model_name == requested
                    || d.model_info.team_public_model_name.as_deref() == Some(requested))
        })
    {
        let teams: std::collections::BTreeSet<&str> = deployments
            .iter()
            .filter_map(|d| d.model_info.team_id.as_deref())
            .collect();
        if teams.len() > 1 {
            return Err(Error::MultipleTeams {
                model: requested.to_owned(),
            });
        }
        return Ok(deployments);
    }
    Ok(deployments
        .into_iter()
        .filter(|d| d.usable_by_team(team_id))
        .collect())
}

impl Core {
    fn context<'a>(
        &'a self,
        snapshot: &'a Snapshot,
        op: &'a Op,
        request: &'a RouterRequest,
        model_group: &'a str,
        name: &'a str,
        scope: StrategyScope,
    ) -> RoutingContext<'a> {
        RoutingContext {
            registry: &snapshot.registry,
            settings: &snapshot.settings,
            health: &self.health,
            request,
            op,
            model_group,
            name,
            scope,
            pool: &[],
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
        // The async pick's pre-routing hook runs the routing plugins before anything resolves;
        // every pick but a listed pass-through one reads what they left on the request.
        let plugin_models = if snapshot.settings.routing_plugins
            && !inputs.count_only
            && request.pass_through != Some(PassThrough::Listed)
        {
            let ctx = self.context(snapshot, op, request, name, name, StrategyScope::Default);
            self.host
                .routing_plugin_models(&ctx, request.pre_routes)
                .await?
        } else {
            None
        };
        // No deployment for the name: route to the first default fallback.
        let default_target = crate::resolve::default_target(&snapshot.settings.dynamic);
        let caller = Caller {
            team_id: request.team_id.as_deref(),
            proxy_admin: request.proxy_admin,
            allowed_models: &request.allowed_models,
        };
        let resolution = resolve_request(
            registry,
            name,
            request.specific_deployment,
            caller,
            default_target,
            snapshot.settings.dynamic.default_ref(),
            jiff::Timestamp::now(),
        )?
        .ok_or_else(|| Error::UnknownModel {
            model: name.to_owned(),
        })?;
        let first_model = resolution
            .deployments
            .first()
            .map(|d| d.params.model.clone());
        let team_id = request.team_id.as_deref();
        let usable = filter_team_based(resolution.deployments.clone(), request)?;
        // `_filter_reserved_deployments`: PTU shares keep holders to their share and everyone
        // else to the deployments split into none.
        let (mut all, withheld) = crate::ptu::filter_shared(usable, team_id);
        if withheld && all.is_empty() {
            return Err(Error::PtuReserved {
                model: match resolution.kind {
                    ResolveKind::Id => first_model.unwrap_or_else(|| name.to_owned()),
                    ResolveKind::Wildcard | ResolveKind::SpecificDeployment => name.to_owned(),
                    _ => resolution.group.clone(),
                },
            });
        }
        let listed =
            request.pass_through == Some(PassThrough::Listed) && !resolution.kind.is_single();
        if listed {
            all.retain(|d| uses_pass_through(d));
            if all.is_empty() {
                return Err(Error::NoPassThroughDeployments {
                    model: name.to_owned(),
                });
            }
        }

        // A deployment named by its id is served even while it cools down
        // (`get_deployment_by_model_id` returns before the cooldown filter).
        let settings = &snapshot.settings;
        let cooling = if resolution.kind.is_single() {
            Vec::new()
        } else {
            let ids: Vec<String> = all.iter().map(|d| d.id.clone()).collect();
            self.health.active_cooldowns(&ids).await?
        };
        // Health-check filtering runs before the cooldown filter, except for a deployment named
        // by its id and the healthy count a retry decision reads.
        let checked = if resolution.kind.is_single() || inputs.count_only {
            all.clone()
        } else {
            let when_all_unhealthy = if request.health_check_probe {
                WhenAllUnhealthy::Nothing
            } else {
                WhenAllUnhealthy::KeepAll
            };
            self.health
                .health_check_filter(settings, all.clone(), when_all_unhealthy)
                .await
        };
        let mut healthy: Vec<Arc<Deployment>> = checked
            .iter()
            .filter(|d| !cooling.iter().any(|(id, _)| *id == d.id))
            .cloned()
            .collect();
        // The safety net when health-check routing drives cooldowns (an allowed_fails_policy):
        // never on a listed pass-through pick or the healthy count.
        if healthy.is_empty()
            && !listed
            && !inputs.count_only
            && settings.enable_health_check_routing
            && settings.allowed_fails_policy.is_some()
        {
            healthy = checked;
        }
        if resolution.kind.is_single()
            && let Some(deployment) = all.first()
        {
            // `_core_async_get_healthy_deployments`: a deployment named by id is checked alone.
            if deployment.model_info.blocked {
                return Err(Error::Paused {
                    model: deployment.params.model.clone(),
                });
            }
            if request.pass_through.is_some() && !uses_pass_through(deployment) {
                return Err(Error::NotPassThrough {
                    id: deployment.id.clone(),
                });
            }
        }
        healthy.retain(|d| !d.model_info.blocked);
        // `continuation_deployments`: a mid-stream continuation never lands on a deployment the host
        // named as unable to take its prefill.
        if let Some(excluded) = request
            .body
            .get("continuation_excluded_ids")
            .and_then(serde_json::Value::as_array)
        {
            healthy.retain(|d| !excluded.iter().any(|id| id.as_str() == Some(d.id.as_str())));
        }
        // `async_callback_filter_deployments` runs after the cooldown and blocked filters on every
        // async candidate list but a deployment named by id and a listed pass-through pick.
        if settings.callback_filters
            && !inputs.count_only
            && !resolution.kind.is_single()
            && request.pass_through != Some(PassThrough::Listed)
        {
            let ctx = self.context(
                snapshot,
                op,
                request,
                &resolution.group,
                name,
                StrategyScope::Default,
            );
            healthy = self.host.callback_filters(&ctx, healthy).await?;
        }
        if inputs.count_only {
            return Ok(Candidates {
                name: name.to_owned(),
                model_group: resolution.group,
                kind: resolution.kind,
                all,
                healthy,
            });
        }

        if listed {
            if settings.enable_pre_call_checks && !resolution.kind.is_single() {
                healthy = self
                    .pre_call_checks(
                        snapshot,
                        op,
                        request,
                        &resolution.group,
                        name,
                        &all,
                        healthy,
                    )
                    .await?;
            }
            // Only a priority group narrows a listed pass-through pick by order.
            let priority = resolution.kind == ResolveKind::RoutingGroup
                && registry.routing_groups().iter().any(|g| {
                    g.group_name == resolution.group && crate::routing_groups::is_priority(g)
                });
            if priority {
                let order = request
                    .target_order
                    .or_else(|| healthy.iter().filter_map(|d| d.order()).min());
                if let Some(order) = order {
                    healthy.retain(|d| d.order() == Some(order));
                }
            }
            return Ok(Candidates {
                name: name.to_owned(),
                model_group: resolution.group,
                kind: resolution.kind,
                all,
                healthy,
            });
        }

        if !settings.plugins.is_empty() && !healthy.is_empty() {
            let ctx = self.context(
                snapshot,
                op,
                request,
                &resolution.group,
                name,
                StrategyScope::Default,
            );
            healthy = self
                .host
                .select(&SelectStage::Plugins, &ctx, healthy)
                .await?;
        }

        if settings.enable_pre_call_checks && !resolution.kind.is_single() {
            healthy = self
                .pre_call_checks(
                    snapshot,
                    op,
                    request,
                    &resolution.group,
                    name,
                    &all,
                    healthy,
                )
                .await?;
        }

        if request.filter_tags && !healthy.is_empty() {
            let mut ctx = self.context(
                snapshot,
                op,
                request,
                &resolution.group,
                name,
                StrategyScope::Default,
            );
            ctx.pool = &all;
            healthy = self.host.select(&SelectStage::Tags, &ctx, healthy).await?;
        }

        if !resolution.kind.is_single() {
            healthy = crate::routing_plugins::narrow(healthy, plugin_models.as_deref())?;
        }

        // `_without_declared_text_only_deployments`: an image request skips the deployments that
        // declare no vision, unless that leaves nothing.
        if request.has_image && !resolution.kind.is_single() && !healthy.is_empty() {
            let ctx = self.context(
                snapshot,
                op,
                request,
                &resolution.group,
                name,
                StrategyScope::Default,
            );
            let text_only = self
                .host
                .select(&SelectStage::TextOnly, &ctx, healthy.clone())
                .await?;
            let vision: Vec<Arc<Deployment>> = healthy
                .iter()
                .filter(|d| !text_only.iter().any(|t| t.id == d.id))
                .cloned()
                .collect();
            if !vision.is_empty() {
                healthy = vision;
            }
        }

        if let Some(pinned) = request.pinned_deployment_id.as_deref()
            && healthy.iter().any(|d| d.id == pinned)
        {
            healthy.retain(|d| d.id == pinned);
        }

        let order = inputs
            .target_order
            .or(request.target_order)
            .or_else(|| healthy.iter().filter_map(|d| d.order()).min());
        if let Some(order) = order {
            healthy.retain(|d| d.order() == Some(order));
        }

        healthy.retain(|d| {
            !request.excluded_ids.contains(&d.id) && !inputs.excluded_ids.contains(&d.id)
        });

        if !inputs.skipped_ids.is_empty() || !request.retry_skipped_ids.is_empty() {
            let kept: Vec<Arc<Deployment>> = healthy
                .iter()
                .filter(|d| {
                    !inputs.skipped_ids.contains(&d.id)
                        && !request.retry_skipped_ids.contains(&d.id)
                })
                .cloned()
                .collect();
            if !kept.is_empty() {
                healthy = kept;
            }
        }

        if request.pass_through == Some(PassThrough::Healthy)
            && !resolution.kind.is_single()
            && !healthy.is_empty()
        {
            healthy.retain(|d| uses_pass_through(d));
            if healthy.is_empty() {
                return Err(Error::NoPassThroughDeployments {
                    model: name.to_owned(),
                });
            }
        }

        Ok(Candidates {
            name: name.to_owned(),
            model_group: resolution.group,
            kind: resolution.kind,
            all,
            healthy,
        })
    }

    /// `_pre_call_checks` over `healthy`, with the host's facts.
    #[allow(clippy::too_many_arguments)]
    async fn pre_call_checks(
        &self,
        snapshot: &Snapshot,
        op: &Op,
        request: &RouterRequest,
        model_group: &str,
        name: &str,
        pool: &[Arc<Deployment>],
        healthy: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error> {
        let mut ctx = self.context(
            snapshot,
            op,
            request,
            model_group,
            name,
            StrategyScope::Default,
        );
        ctx.pool = pool;
        let facts = self.host.pre_call_facts(&ctx, &healthy).await?;
        crate::precall::filter(model_group, healthy, pool, &facts)
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
        // A deployment named by its id is returned as is, before any strategy runs.
        if candidates.kind.is_single() {
            return candidates
                .healthy
                .first()
                .cloned()
                .ok_or_else(no_deployment);
        }
        let (strategy, scope) = routing_context(
            &snapshot.registry,
            &snapshot.settings.routing_strategy,
            request.routing_strategy.as_deref(),
            &candidates.kind,
            &candidates.model_group,
        );
        if strategy == RoutingStrategy::Unset {
            return Err(no_deployment());
        }
        if strategy == RoutingStrategy::SimpleShuffle {
            let weights = crate::weights::validate(request.router_weights.as_ref())
                .map_err(|_| Error::InvalidRouterWeights)?;
            return simple_shuffle(
                &candidates.healthy,
                &candidates.model_group,
                weights.as_ref(),
                self.rng.as_ref(),
            )?
            .ok_or_else(no_deployment);
        }
        let ctx = self.context(
            snapshot,
            op,
            request,
            &candidates.model_group,
            &candidates.name,
            scope,
        );
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
        let picked = simple_shuffle(&deployments, "g", None, &Fixed(draw))
            .unwrap()
            .unwrap();
        assert_eq!(picked.id, expected);
    }

    #[rstest]
    fn shuffle_prefers_request_weights() {
        let deployments = [
            deployment("a", json!({"model": "openai/a", "weight": 100})),
            deployment("b", json!({"model": "openai/b"})),
        ];
        let weights = BTreeMap::from([("g".to_owned(), BTreeMap::from([("b".to_owned(), 1.0)]))]);
        let picked = simple_shuffle(&deployments, "g", Some(&weights), &Fixed(0.99))
            .unwrap()
            .unwrap();
        assert_eq!(picked.id, "b");
    }
}
