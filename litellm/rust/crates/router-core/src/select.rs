//! Selection: resolve a name, filter its deployments in the order of
//! `Router._core_async_get_healthy_deployments` (team, web search, cooldown, blocked, plugins,
//! pre-call checks, tags, order, retry skips), then pick one with the strategy. Simple-shuffle is
//! native (`router_strategy/simple_shuffle.py`); other strategies go through `RouterHost::select`.

use std::{collections::BTreeMap, sync::Arc};

use parking_lot::Mutex;
use rand::{Rng, SeedableRng, rngs::StdRng};

use litellm_router_types::RoutingStrategy;

use crate::{
    Call, RouterStateSnapshot,
    config::Op,
    error::Error,
    health_routing::WhenAllUnhealthy,
    host::{RouterHost, RouterRequest, RoutingContext, SelectStage},
    registry::{Deployment, Registry, Resolution, ResolveKind},
    resolve::{Caller, resolve_request},
    strategy::{StrategyScope, request_override, routing_context},
};

/// Randomness for the shuffle and retry jitter, injectable so tests are deterministic.
pub trait Randomness: Send + Sync {
    /// A float in `[0, 1)`, as Python's `random.random()`.
    fn next_f64(&self) -> f64;

    /// An index below `len`, as Python's `random.choice` draws one.
    fn index_below(&self, len: usize) -> usize {
        ((self.next_f64() * len as f64) as usize).min(len - 1)
    }
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
    /// Every deployment the caller may use (after the team filter), less the batch-only ones it skips.
    pub all: Vec<Arc<Deployment>>,
    pub healthy: Vec<Arc<Deployment>>,
    /// The call skips the group's batch-only deployments, decided before a capability filter
    /// drops any of the group's other deployments.
    pub skips_batch_only: bool,
    /// The provider-prefixed spelling the name resolved under (`get_pattern`), when the host
    /// was asked for one.
    pub wildcard_request: Option<String>,
    /// The group the host stamped the pick as served for when a pre-routing hook rewrote the name
    /// (a Claude Code subagent's bound router): a fallback lookup key.
    pub served_as: Option<String>,
}

impl Candidates {
    /// The `model` a no-deployments error names: what `_common_checks_available_deployment`
    /// returned, so the alias-resolved name a wildcard serves and not its pattern.
    /// The error an empty candidate list raises: it names the deployments the request could
    /// route to before the cooldown filter.
    pub(crate) fn none_healthy(&self, registry: &Registry) -> Error {
        Error::NoHealthyDeployments {
            model: self.failed_model(registry),
            request_ids: Some(self.all.iter().map(|d| d.id.clone()).collect()),
        }
    }

    /// How many of `all` the request can route to: a call outside the batch and file APIs does
    /// not count the batch-only deployments it skips.
    pub(crate) fn routable(&self) -> usize {
        if self.skips_batch_only {
            return self.all.iter().filter(|d| !d.serves_only_batches()).count();
        }
        self.all.len()
    }

    pub(crate) fn failed_model(&self, registry: &Registry) -> String {
        let model = match self.kind {
            ResolveKind::Wildcard => registry.alias_of(&self.name).unwrap_or(&self.name),
            _ => self.model_group.as_str(),
        };
        model.to_owned()
    }
}

/// Per-attempt selection inputs the call loop carries.
#[derive(Clone, Debug, Default)]
pub(crate) struct SelectInputs {
    pub target_order: Option<i64>,
    pub skipped_ids: Vec<String>,
    /// Every deployment an attempt of this request failed on, whatever the status: the callback
    /// filters read it as `_retry_skipped_deployment_ids`, so an affinity pin can leave one.
    pub failed_ids: Vec<String>,
    /// A weighted failover hop's `_excluded_deployment_ids`: never picked.
    pub excluded_ids: Arc<[String]>,
    /// Stop before the plugins, order and skip filters: the group less cooldowns and blocked
    /// deployments, which `_async_get_healthy_deployments` counts when deciding whether to retry.
    pub count_only: bool,
    /// The level's name, which an endpoint refusal reads (`endpoint_selection(model=...)`): the
    /// name before pre-routing replaced it.
    pub requested: Option<String>,
    /// A fallback hop: `run_async_fallback` pops the request's `_target_order`, so only the
    /// hop's own order applies.
    pub fallback_hop: bool,
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
    Ok(Some(
        deployments[rng.index_below(deployments.len())].clone(),
    ))
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

/// `without_deployments_that_cannot_batch`: a batch create on a group the caller may use a healthy
/// batch deployment of skips the Azure deployments that cannot run batches. A batch upload skips
/// them while the caller may use any batch deployment of the group, healthy or not, and the retry
/// count reads the same list.
fn without_deployments_that_cannot_batch(
    request: &RouterRequest,
    all: &[Arc<Deployment>],
    healthy: Vec<Arc<Deployment>>,
) -> Vec<Arc<Deployment>> {
    let has_batch = |pool: &[Arc<Deployment>]| pool.iter().any(|d| d.is_batch_capable());
    if !request.batch_create || !(has_batch(&healthy) || (request.batch_upload && has_batch(all))) {
        return healthy;
    }
    healthy
        .into_iter()
        .filter(|d| !d.cannot_run_batches())
        .collect()
}

pub(crate) fn skips_batch_only(request: &RouterRequest, pool: &[Arc<Deployment>]) -> bool {
    !request.batch_create
        && !request.batch_api
        && !pinned_to_batch_only(request, pool)
        && pool.iter().any(|d| !d.serves_only_batches())
}

fn pinned_to_batch_only(request: &RouterRequest, pool: &[Arc<Deployment>]) -> bool {
    request
        .pinned_deployment_id
        .as_deref()
        .is_some_and(|pinned| {
            pool.iter()
                .any(|d| d.id == pinned && d.serves_only_batches())
        })
}

fn without_batch_only_deployments(
    skips_batch_only: bool,
    healthy: Vec<Arc<Deployment>>,
) -> Vec<Arc<Deployment>> {
    if !skips_batch_only {
        return healthy;
    }
    healthy
        .into_iter()
        .filter(|d| !d.serves_only_batches())
        .collect()
}

/// A kind of pass a stage of the filter chain sits out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Except {
    /// A deployment the request names by its id.
    ById,
    /// A listed pass-through pick ([`PassThrough::Listed`]).
    Listed,
    /// The count a retry decision reads ([`SelectInputs::count_only`]).
    Count,
}

/// One filter of the chain. [`Stage::skipped_on`] says which passes it sits out, and
/// `Call::keep` what it keeps on the others.
#[derive(Clone, Copy, Debug)]
enum Stage {
    /// [`without_batch_only_deployments`], ahead of every capability filter so the later stages
    /// see the group as if the batch-only deployments were absent. A listed pass-through request
    /// keeps them in the pool.
    BatchOnly,
    /// `filter_web_search_deployments`: runs before `request_deployments` is read, so every later
    /// stage sees the group without the deployments it dropped.
    WebSearch,
    /// `_without_deployments_that_cannot_serve_route`: the deployments whose provider cannot
    /// serve the call, unless that leaves nothing (`_route_capability_pin`).
    RouteCapable,
    /// A listed pass-through request keeps only the pass-through deployments, in the count too.
    ListedPassThrough,
    /// `_core_async_get_healthy_deployments`: a deployment named by id is checked alone, and
    /// refused when it is paused or serves no pass-through.
    NamedById,
    Blocked,
    /// `continuation_deployments`: a mid-stream continuation never lands on a deployment the host
    /// named as unable to take its prefill.
    Continuation,
    /// `async_callback_filter_deployments`, which also names the deployment it prefers.
    CallbackFilters,
    Plugins,
    /// `_pre_call_checks`, with the host's facts.
    PreCallChecks,
    /// Only a priority group narrows a listed pass-through pick by order.
    ListedPriority,
    Tags,
    /// The models the routing plugins left on the request.
    RoutingPlugins,
    /// `_without_declared_text_only_deployments`: an image request skips the deployments that
    /// declare no vision, unless that leaves nothing.
    Vision,
    /// `_without_decisions_deployments_refusing_media`: a Decisions call carrying audio or videos
    /// skips the deployments that refuse them, unless that leaves nothing.
    DecisionsMedia,
    /// `deployments_admitting_embedding_batch`: an embedding pick keeps the deployments whose
    /// input limit admits the batch; when none does, the host keeps what the pick waits for or
    /// the 400 names.
    EmbeddingBatch,
    /// [`without_deployments_that_cannot_batch`], then [`without_batch_only_deployments`] for the
    /// count of a listed pass-through request, whose pool kept them.
    Batch,
    /// The deployment the request pins, else the one the callback filters prefer, when healthy.
    Pinned,
    /// The hop's order, the request's, else the lowest order among the healthy.
    Order,
    Excluded,
    /// The deployments a retry skips, unless that leaves nothing.
    RetrySkips,
    /// `async_get_available_deployment_for_pass_through`: the pass-through filter after the full
    /// healthy chain.
    HealthyPassThrough,
}

/// The stages over every deployment the caller may use, in order.
const POOL: &[Stage] = &[
    Stage::BatchOnly,
    Stage::WebSearch,
    Stage::RouteCapable,
    Stage::ListedPassThrough,
];

/// The stages over the pool's deployments that are not cooling down, in order. The count a retry
/// decision reads ends at [`Stage::Batch`], and a listed pass-through pick at
/// [`Stage::ListedPriority`], by what the later stages sit out.
const HEALTHY: &[Stage] = &[
    Stage::NamedById,
    Stage::Blocked,
    Stage::Continuation,
    Stage::CallbackFilters,
    Stage::Plugins,
    Stage::PreCallChecks,
    Stage::ListedPriority,
    Stage::Tags,
    Stage::RoutingPlugins,
    Stage::Vision,
    Stage::DecisionsMedia,
    Stage::EmbeddingBatch,
    Stage::Batch,
    Stage::Pinned,
    Stage::Order,
    Stage::Excluded,
    Stage::RetrySkips,
    Stage::HealthyPassThrough,
];

impl Stage {
    /// The passes the stage sits out.
    const fn skipped_on(self) -> &'static [Except] {
        use Except::{ById, Count, Listed};
        match self {
            Self::ListedPassThrough
            | Self::NamedById
            | Self::Blocked
            | Self::Continuation
            | Self::ListedPriority => &[],
            Self::BatchOnly => &[ById],
            Self::WebSearch | Self::RouteCapable | Self::PreCallChecks => &[ById, Count],
            Self::Batch => &[ById, Listed],
            Self::Plugins
            | Self::Tags
            | Self::Pinned
            | Self::Order
            | Self::Excluded
            | Self::RetrySkips => &[Listed, Count],
            Self::CallbackFilters
            | Self::RoutingPlugins
            | Self::Vision
            | Self::DecisionsMedia
            | Self::EmbeddingBatch
            | Self::HealthyPassThrough => &[ById, Listed, Count],
        }
    }
}

/// One run of the filter chain: the request, the name it resolved to and what the stages share.
struct Pass<'a> {
    snapshot: &'a RouterStateSnapshot,
    op: &'a Op,
    request: &'a RouterRequest,
    name: &'a str,
    inputs: &'a SelectInputs,
    /// The model group the name resolved to.
    group: &'a str,
    kind: &'a ResolveKind,
    /// What the routing plugins left on the request.
    plugin_models: Option<Vec<String>>,
    /// [`skips_batch_only`], decided on the caller's deployments before [`POOL`] ran.
    skips_batch_only: bool,
    /// Every deployment the caller may use, once [`POOL`] ran.
    pool: &'a [Arc<Deployment>],
    /// The deployment the callback filters prefer.
    preferred: Option<String>,
}

impl Pass<'_> {
    fn is(&self, except: Except) -> bool {
        match except {
            Except::ById => self.kind.is_single(),
            Except::Listed => self.lists_pass_through() && !self.inputs.count_only,
            Except::Count => self.inputs.count_only,
        }
    }

    /// A listed pass-through request for a group, whether it picks or counts.
    fn lists_pass_through(&self) -> bool {
        self.request.pass_through == Some(PassThrough::Listed) && !self.kind.is_single()
    }

    /// The request's `_target_order`, which `run_async_fallback` pops for a fallback hop.
    fn request_order(&self) -> Option<i64> {
        self.request
            .target_order
            .filter(|_| !self.inputs.fallback_hop)
    }
}

/// `resolution`'s deployments the caller may use: the team filter, then
/// `_filter_reserved_deployments`, where PTU shares keep holders to their share and everyone
/// else to the deployments split into none.
fn usable_by_caller(
    resolution: &Resolution,
    request: &RouterRequest,
    name: &str,
) -> Result<Vec<Arc<Deployment>>, Error> {
    let usable = filter_team_based(resolution.deployments.clone(), request)?;
    let (all, withheld) = crate::ptu::filter_shared(usable, request.team_id.as_deref());
    if withheld && all.is_empty() {
        let first_model = resolution
            .deployments
            .first()
            .map(|d| d.params.model.clone());
        return Err(Error::PtuReserved {
            model: match resolution.kind {
                ResolveKind::Id => first_model.unwrap_or_else(|| name.to_owned()),
                ResolveKind::Wildcard | ResolveKind::SpecificDeployment => name.to_owned(),
                _ => resolution.group.clone(),
            },
        });
    }
    Ok(all)
}

/// `list` without `dropped`, unless that leaves nothing.
fn without_unless_none_left(
    list: Vec<Arc<Deployment>>,
    dropped: &[Arc<Deployment>],
) -> Vec<Arc<Deployment>> {
    let kept: Vec<Arc<Deployment>> = list
        .iter()
        .filter(|d| !dropped.iter().any(|gone| gone.id == d.id))
        .cloned()
        .collect();
    if kept.is_empty() { list } else { kept }
}

impl<H: RouterHost> Call<'_, H> {
    fn context<'a>(
        &'a self,
        snapshot: &'a RouterStateSnapshot,
        op: &'a Op,
        request: &'a RouterRequest,
        model_group: &'a str,
        name: &'a str,
        scope: StrategyScope,
    ) -> RoutingContext<'a> {
        RoutingContext {
            registry: &snapshot.registry,
            settings: &snapshot.settings,
            request,
            op,
            model_group,
            name,
            scope,
            pool: &[],
            excluded_ids: &[],
            skipped_ids: &[],
            target_order: None,
        }
    }

    /// The eligible deployments for `name`: the name resolved for the caller, then [`POOL`] over
    /// every deployment the caller may use and [`HEALTHY`] over those not cooling down.
    pub(crate) async fn candidates(
        &self,
        snapshot: &RouterStateSnapshot,
        op: &Op,
        request: &RouterRequest,
        name: &str,
        inputs: &SelectInputs,
    ) -> Result<Candidates, Error> {
        let plugin_models = self
            .plugin_models(snapshot, op, request, name, inputs)
            .await?;
        let (resolution, wildcard_request) =
            self.resolution(snapshot, op, request, name, inputs).await?;
        let usable = usable_by_caller(&resolution, request, name)?;
        let skip_batch_only = skips_batch_only(request, &usable);
        let mut pass = Pass {
            snapshot,
            op,
            request,
            name,
            inputs,
            group: &resolution.group,
            kind: &resolution.kind,
            plugin_models,
            skips_batch_only: skip_batch_only,
            pool: &[],
            preferred: None,
        };
        let all = self.filtered(POOL, &mut pass, usable).await?;
        let mut pass = Pass { pool: &all, ..pass };
        let healthy = self.not_cooling(&pass).await?;
        let healthy = self.filtered(HEALTHY, &mut pass, healthy).await?;
        Ok(Candidates {
            name: name.to_owned(),
            model_group: resolution.group,
            kind: resolution.kind,
            all,
            healthy,
            skips_batch_only: skip_batch_only,
            wildcard_request,
            served_as: None,
        })
    }

    /// What the routing plugins left on the request. The async pick's pre-routing hook runs them
    /// before anything resolves, and every pick but a listed pass-through one reads the result.
    async fn plugin_models(
        &self,
        snapshot: &RouterStateSnapshot,
        op: &Op,
        request: &RouterRequest,
        name: &str,
        inputs: &SelectInputs,
    ) -> Result<Option<Vec<String>>, Error> {
        if !snapshot.settings.routing_plugins
            || inputs.count_only
            || request.pass_through == Some(PassThrough::Listed)
        {
            return Ok(None);
        }
        let ctx = self.context(snapshot, op, request, name, name, StrategyScope::Default);
        self.host
            .routing_plugin_models(&ctx, request.pre_routes)
            .await
    }

    /// `name` resolved for the caller, with the provider-prefixed spelling it resolved under when
    /// the host was asked for one.
    async fn resolution(
        &self,
        snapshot: &RouterStateSnapshot,
        op: &Op,
        request: &RouterRequest,
        name: &str,
        inputs: &SelectInputs,
    ) -> Result<(Resolution, Option<String>), Error> {
        let registry = &snapshot.registry;
        // No deployment for the name: route to the first default fallback.
        let default_target = crate::resolve::default_target(&snapshot.settings.dynamic);
        // `get_pattern`: a name no pattern matches is tried again under its inferred provider.
        let wildcard_name = crate::resolve::wildcard_name(
            registry,
            name,
            request.specific_deployment,
            request.team_id.as_deref(),
        );
        let wildcard_request = match wildcard_name {
            Some(model) => {
                let scope = StrategyScope::Default;
                let ctx = self.context(snapshot, op, request, model, model, scope);
                self.host.provider_request(&ctx).await?
            }
            None => None,
        };
        // `endpoint_selection` wraps the attempt, not the count a retry decision reads.
        let endpoint = request.endpoint.as_deref().filter(|_| !inputs.count_only);
        let caller = Caller {
            team_id: request.team_id.as_deref(),
            proxy_admin: request.proxy_admin,
            allowed_models: &request.allowed_models,
            endpoint,
            requested: inputs.requested.as_deref().unwrap_or(name),
            wildcard_request: wildcard_request.as_deref(),
            refuse_cross_group_litellm_model: !inputs.count_only
                && snapshot.settings.enable_cross_model_group_collision_check,
        };
        let resolution = resolve_request(
            registry,
            name,
            request.specific_deployment,
            caller,
            default_target,
            snapshot.settings.dynamic.default_ref(),
            self.health().timestamp(),
        )?
        .ok_or_else(|| Error::UnknownModel {
            // The name `_common_checks_available_deployment` looked for, past the alias.
            model: registry.alias_of(name).unwrap_or(name).to_owned(),
        })?;
        Ok((resolution, wildcard_request))
    }

    /// `deployments` after every stage of `stages` the pass does not sit out, in order.
    async fn filtered(
        &self,
        stages: &[Stage],
        pass: &mut Pass<'_>,
        deployments: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error> {
        let mut kept = deployments;
        for stage in stages {
            if stage.skipped_on().iter().any(|except| pass.is(*except)) {
                continue;
            }
            kept = self.keep(*stage, pass, kept).await?;
        }
        Ok(kept)
    }

    /// The pool's deployments that are not cooling down, past the health-check filter.
    async fn not_cooling(&self, pass: &Pass<'_>) -> Result<Vec<Arc<Deployment>>, Error> {
        let settings = &pass.snapshot.settings;
        // A deployment named by its id is served even while it cools down
        // (`get_deployment_by_model_id` returns before the cooldown filter).
        let cooling = if pass.is(Except::ById) || pass.pool.is_empty() {
            Vec::new()
        } else {
            let ids: Vec<String> = pass.pool.iter().map(|d| d.id.clone()).collect();
            self.health().cooling(&ids, !pass.skips_batch_only).await?
        };
        // Health-check filtering runs before the cooldown filter, except for a deployment named
        // by its id and the healthy count a retry decision reads.
        let checked = if pass.is(Except::ById) || pass.is(Except::Count) {
            pass.pool.to_vec()
        } else {
            let when_all_unhealthy = if pass.request.health_check_probe {
                WhenAllUnhealthy::Nothing
            } else {
                WhenAllUnhealthy::KeepAll
            };
            self.health()
                .health_check_filter(settings, pass.pool.to_vec(), when_all_unhealthy)
                .await
        };
        let healthy: Vec<Arc<Deployment>> = checked
            .iter()
            .filter(|d| !cooling.iter().any(|(id, _)| *id == d.id))
            .cloned()
            .collect();
        // The safety net when health-check routing drives cooldowns (an allowed_fails_policy):
        // never on a listed pass-through pick or the healthy count.
        if healthy.is_empty()
            && !pass.is(Except::Listed)
            && !pass.is(Except::Count)
            && settings.enable_health_check_routing
            && settings.allowed_fails_policy.is_some()
        {
            return Ok(checked);
        }
        Ok(healthy)
    }

    /// The host's view of the pass: the group and the name, with no pool, ids or order.
    fn pass_context<'a>(&'a self, pass: &'a Pass<'_>) -> RoutingContext<'a> {
        self.context(
            pass.snapshot,
            pass.op,
            pass.request,
            pass.group,
            pass.name,
            StrategyScope::Default,
        )
    }

    /// What `stage` keeps of `list`.
    async fn keep(
        &self,
        stage: Stage,
        pass: &mut Pass<'_>,
        mut list: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error> {
        let (request, inputs, name) = (pass.request, pass.inputs, pass.name);
        let (registry, settings) = (&pass.snapshot.registry, &pass.snapshot.settings);
        match stage {
            Stage::BatchOnly => {
                if !pass.lists_pass_through() {
                    list = without_batch_only_deployments(pass.skips_batch_only, list);
                }
            }
            Stage::WebSearch => {
                if request.web_search {
                    list.retain(|d| d.supports_web_search());
                }
            }
            Stage::RouteCapable => {
                if let Some(handler) = request.route_handler.as_deref()
                    && !list.is_empty()
                {
                    let incapable = SelectStage::RouteIncapable {
                        handler: handler.to_owned(),
                    };
                    let ctx = self.pass_context(pass);
                    let incapable = self.host.select(&incapable, &ctx, list.clone()).await?;
                    list = without_unless_none_left(list, &incapable);
                }
            }
            Stage::ListedPassThrough => {
                if pass.lists_pass_through() {
                    list.retain(|d| uses_pass_through(d));
                    if list.is_empty() {
                        // The name `_common_checks_available_deployment` resolved: an alias's
                        // target, a routing group's name, the alias-resolved name a wildcard
                        // serves.
                        let model = match pass.kind {
                            ResolveKind::Wildcard => registry.alias_of(name).unwrap_or(name),
                            ResolveKind::SpecificDeployment => name,
                            _ => pass.group,
                        };
                        return Err(Error::NoPassThroughDeployments {
                            model: model.to_owned(),
                        });
                    }
                }
            }
            Stage::NamedById => {
                if pass.is(Except::ById)
                    && let Some(deployment) = pass.pool.first()
                {
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
            }
            Stage::Blocked => list.retain(|d| !d.model_info.blocked),
            Stage::Continuation => {
                if let Some(excluded) = request
                    .body
                    .get("continuation_excluded_ids")
                    .and_then(serde_json::Value::as_array)
                {
                    list.retain(|d| !excluded.iter().any(|id| id.as_str() == Some(d.id.as_str())));
                }
            }
            Stage::CallbackFilters => {
                if settings.callback_filters {
                    let mut ctx = self.pass_context(pass);
                    ctx.excluded_ids = &inputs.excluded_ids;
                    ctx.skipped_ids = &inputs.failed_ids;
                    ctx.target_order = inputs.target_order.or(pass.request_order());
                    let filtered = self.host.callback_filters(&ctx, list).await?;
                    pass.preferred = filtered.preferred;
                    list = filtered.kept;
                }
            }
            Stage::Plugins => {
                if !settings.plugins.is_empty() && !list.is_empty() {
                    let ctx = self.pass_context(pass);
                    list = self.host.select(&SelectStage::Plugins, &ctx, list).await?;
                }
            }
            Stage::PreCallChecks => {
                if settings.enable_pre_call_checks {
                    let mut ctx = self.pass_context(pass);
                    ctx.pool = pass.pool;
                    let facts = self.host.pre_call_facts(&ctx, &list).await?;
                    list = crate::precall::filter(pass.group, list, pass.pool, &facts)?;
                }
            }
            Stage::ListedPriority => {
                let priority = pass.is(Except::Listed)
                    && *pass.kind == ResolveKind::RoutingGroup
                    && registry.routing_groups().iter().any(|g| {
                        g.group_name == pass.group && crate::routing_groups::is_priority(g)
                    });
                if priority {
                    let order = request
                        .target_order
                        .or_else(|| list.iter().filter_map(|d| d.order()).min());
                    if let Some(order) = order {
                        list.retain(|d| d.order() == Some(order));
                    }
                }
            }
            Stage::Tags => {
                if request.filter_tags && !list.is_empty() {
                    let mut ctx = self.pass_context(pass);
                    ctx.pool = pass.pool;
                    list = self.host.select(&SelectStage::Tags, &ctx, list).await?;
                }
            }
            Stage::RoutingPlugins => {
                list = crate::routing_plugins::narrow(list, pass.plugin_models.as_deref())?;
            }
            Stage::Vision => {
                if request.has_image && !list.is_empty() {
                    let ctx = self.pass_context(pass);
                    let text_only = self
                        .host
                        .select(&SelectStage::TextOnly, &ctx, list.clone())
                        .await?;
                    list = without_unless_none_left(list, &text_only);
                }
            }
            Stage::DecisionsMedia => {
                if request.decisions_media && !list.is_empty() {
                    let ctx = self.pass_context(pass);
                    let refusing = self
                        .host
                        .select(&SelectStage::MediaRefusing, &ctx, list.clone())
                        .await?;
                    list = without_unless_none_left(list, &refusing);
                }
            }
            Stage::EmbeddingBatch => {
                if request.embedding_batch && !list.is_empty() {
                    let mut ctx = self.pass_context(pass);
                    ctx.pool = pass.pool;
                    list = self
                        .host
                        .select(&SelectStage::EmbeddingBatch, &ctx, list)
                        .await?;
                }
            }
            Stage::Batch => {
                let batching = without_deployments_that_cannot_batch(request, pass.pool, list);
                list = without_batch_only_deployments(pass.skips_batch_only, batching);
            }
            Stage::Pinned => {
                if let Some(pinned) = [
                    request.pinned_deployment_id.as_deref(),
                    pass.preferred.as_deref(),
                ]
                .into_iter()
                .flatten()
                .find(|id| list.iter().any(|d| d.id == *id))
                {
                    list.retain(|d| d.id == pinned);
                }
            }
            Stage::Order => {
                let order = inputs
                    .target_order
                    .or(pass.request_order())
                    .or_else(|| list.iter().filter_map(|d| d.order()).min());
                if let Some(order) = order {
                    list.retain(|d| d.order() == Some(order));
                }
            }
            Stage::Excluded => list.retain(|d| {
                !request.excluded_ids.contains(&d.id) && !inputs.excluded_ids.contains(&d.id)
            }),
            Stage::RetrySkips => {
                if !inputs.skipped_ids.is_empty() || !request.retry_skipped_ids.is_empty() {
                    let kept: Vec<Arc<Deployment>> = list
                        .iter()
                        .filter(|d| {
                            !inputs.skipped_ids.contains(&d.id)
                                && !request.retry_skipped_ids.contains(&d.id)
                        })
                        .cloned()
                        .collect();
                    if !kept.is_empty() {
                        list = kept;
                    }
                }
            }
            Stage::HealthyPassThrough => {
                if request.pass_through == Some(PassThrough::Healthy) && !list.is_empty() {
                    list.retain(|d| uses_pass_through(d));
                    if list.is_empty() {
                        return Err(Error::NoPassThroughDeployments {
                            model: name.to_owned(),
                        });
                    }
                }
            }
        }
        Ok(list)
    }

    /// Pick one of `candidates.healthy` with the group's strategy.
    pub(crate) async fn pick(
        &self,
        snapshot: &RouterStateSnapshot,
        op: &Op,
        request: &RouterRequest,
        candidates: &Candidates,
    ) -> Result<Arc<Deployment>, Error> {
        let no_deployment = || Error::NoHealthyDeployments {
            model: candidates.failed_model(&snapshot.registry),
            request_ids: None,
        };
        // A deployment named by its id is returned as is, before any strategy runs, past a
        // per-request strategy override's pre-call check.
        if candidates.kind.is_single() {
            let named = candidates
                .healthy
                .first()
                .cloned()
                .ok_or_else(no_deployment)?;
            if let Some(strategy) = request_override(request.routing_strategy.as_deref())
                && !matches!(
                    strategy,
                    RoutingStrategy::SimpleShuffle | RoutingStrategy::Unset
                )
            {
                let ctx = self.context(
                    snapshot,
                    op,
                    request,
                    &candidates.model_group,
                    &candidates.name,
                    StrategyScope::Override,
                );
                let kept = self
                    .host
                    .select(
                        &SelectStage::OverrideCheck(strategy),
                        &ctx,
                        vec![named.clone()],
                    )
                    .await?;
                if kept.is_empty() {
                    return Err(no_deployment());
                }
            }
            return Ok(named);
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
        let mut ctx = self.context(
            snapshot,
            op,
            request,
            &candidates.model_group,
            &candidates.name,
            scope,
        );
        ctx.pool = &candidates.all;
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

    use std::{collections::HashMap, time::Duration};

    use super::*;
    use crate::{
        Router,
        cache::{
            HealthState, InMemoryRouterCache, Outcomes, RouterCache, UsageCounts, UsageWindow,
        },
        config::{RouterConfig, RouterDynamicConfig},
        error::{AttemptError, FailureKind},
        health::CooldownValue,
        host::{Attempt, AttemptFailure, CallbackFiltered, RouterEvent, Served},
        precall::PreCallFacts,
        registry::DeploymentSpec,
    };

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

    type Calls = Arc<Mutex<Vec<String>>>;

    fn listing(deployments: &[Arc<Deployment>]) -> String {
        let ids: Vec<&str> = deployments.iter().map(|d| d.id.as_str()).collect();
        ids.join(",")
    }

    /// An [`InMemoryRouterCache`] that keeps the reads the filter chain makes, among the host's
    /// calls.
    #[derive(Default)]
    struct AskedCache {
        inner: InMemoryRouterCache,
        calls: Calls,
    }

    impl RouterCache for AskedCache {
        async fn cooldowns(
            &self,
            ids: &[String],
        ) -> Result<Vec<Option<CooldownValue>>, litellm_cache::Error> {
            self.calls
                .lock()
                .push(format!("cooldowns {}", ids.join(",")));
            self.inner.cooldowns(ids).await
        }

        async fn put_cooldown(
            &self,
            id: &str,
            cooldown: &CooldownValue,
            ttl: Duration,
        ) -> Result<(), litellm_cache::Error> {
            self.inner.put_cooldown(id, cooldown, ttl).await
        }

        async fn increment_allowed_fails(
            &self,
            id: &str,
            suffix: Option<&str>,
            ttl: Duration,
        ) -> Result<f64, litellm_cache::Error> {
            self.inner.increment_allowed_fails(id, suffix, ttl).await
        }

        async fn add_usage(
            &self,
            window: &UsageWindow,
            tokens: f64,
            requests: f64,
        ) -> Result<(), litellm_cache::Error> {
            self.inner.add_usage(window, tokens, requests).await
        }

        async fn add_request(&self, window: &UsageWindow) -> Result<(), litellm_cache::Error> {
            self.inner.add_request(window).await
        }

        async fn usage(&self, window: &UsageWindow) -> Result<UsageCounts, litellm_cache::Error> {
            self.inner.usage(window).await
        }

        async fn health_states(
            &self,
        ) -> Result<HashMap<String, HealthState>, litellm_cache::Error> {
            self.calls.lock().push("health_states".to_owned());
            self.inner.health_states().await
        }

        async fn outcomes(&self, id: &str) -> Result<Outcomes, litellm_cache::Error> {
            self.inner.outcomes(id).await
        }

        async fn count_success(&self, id: &str) -> Result<(), litellm_cache::Error> {
            self.inner.count_success(id).await
        }

        async fn count_failure(&self, id: &str) -> Result<(), litellm_cache::Error> {
            self.inner.count_failure(id).await
        }
    }

    /// Keeps every call the filter chain makes, in order. A stage answers with the candidates it
    /// was given, unless the test names the ones it answers with.
    #[derive(Default)]
    struct Asked {
        calls: Calls,
        cache: AskedCache,
        text_only: Option<&'static [&'static str]>,
        preferred: Option<&'static str>,
    }

    impl Asked {
        fn new() -> Self {
            let calls = Calls::default();
            Self {
                cache: AskedCache {
                    calls: calls.clone(),
                    ..AskedCache::default()
                },
                calls,
                ..Self::default()
            }
        }

        fn asked(&self, call: &str, candidates: &[Arc<Deployment>], pool: &[Arc<Deployment>]) {
            let of = match pool {
                [] => String::new(),
                pool => format!(" of {}", listing(pool)),
            };
            self.calls
                .lock()
                .push(format!("{call} {}{of}", listing(candidates)));
        }
    }

    impl RouterHost for Asked {
        type Response = ();
        type Failure = ();
        type Cache = AskedCache;

        async fn attempt(&self, _attempt: &Attempt) -> Result<Served<()>, AttemptFailure<()>> {
            Err(AttemptFailure::Refused(()))
        }

        async fn select(
            &self,
            stage: &SelectStage,
            ctx: &RoutingContext<'_>,
            candidates: Vec<Arc<Deployment>>,
        ) -> Result<Vec<Arc<Deployment>>, Error> {
            let (name, answer) = match stage {
                SelectStage::Plugins => ("plugins", None),
                SelectStage::Tags => ("tags", None),
                SelectStage::TextOnly => ("text_only", self.text_only),
                SelectStage::RouteIncapable { .. } => ("route_incapable", None),
                SelectStage::MediaRefusing => ("media_refusing", None),
                SelectStage::EmbeddingBatch => ("embedding_batch", None),
                SelectStage::Strategy(_) | SelectStage::OverrideCheck(_) => ("strategy", None),
            };
            self.asked(name, &candidates, ctx.pool);
            Ok(match answer {
                Some(ids) => candidates
                    .into_iter()
                    .filter(|d| ids.contains(&d.id.as_str()))
                    .collect(),
                None => candidates,
            })
        }

        async fn routing_plugin_models(
            &self,
            _ctx: &RoutingContext<'_>,
            _run: bool,
        ) -> Result<Option<Vec<String>>, Error> {
            self.calls.lock().push("plugin_models".to_owned());
            Ok(None)
        }

        async fn pre_call_facts(
            &self,
            ctx: &RoutingContext<'_>,
            candidates: &[Arc<Deployment>],
        ) -> Result<PreCallFacts, Error> {
            self.asked("pre_call_facts", candidates, ctx.pool);
            Ok(PreCallFacts::default())
        }

        async fn callback_filters(
            &self,
            ctx: &RoutingContext<'_>,
            candidates: Vec<Arc<Deployment>>,
        ) -> Result<CallbackFiltered, Error> {
            self.asked("callback_filters", &candidates, ctx.pool);
            Ok(CallbackFiltered {
                kept: candidates,
                preferred: self.preferred.map(str::to_owned),
            })
        }

        async fn allow_fallback(&self, _target: &str) -> Result<bool, AttemptError> {
            Ok(true)
        }

        async fn sleep(&self, _duration: Duration) {}

        fn observe(&self, _event: &RouterEvent) {}

        fn cache(&self) -> &AskedCache {
            &self.cache
        }
    }

    /// `g`: `d1` at order 1, and two Azure pass-through deployments, `d2` at order 2 and the
    /// batch-only deployment `d3` at order 1, which only a batch call, a listed pass-through pick
    /// and a pick by id keep. `solo`: the pass-through `s1`. The priority group `rg` serves `g`
    /// before `solo`.
    fn groups(settings: serde_json::Value) -> Router {
        let mut config = json!({
            "model_list": [
                {"model_name": "g", "litellm_params": {"model": "openai/a", "order": 1}, "model_info": {"id": "d1"}},
                {"model_name": "g", "litellm_params": {"model": "azure/b", "order": 2, "use_in_pass_through": true}, "model_info": {"id": "d2"}},
                {"model_name": "g", "litellm_params": {"model": "azure/c", "order": 1, "use_in_pass_through": true}, "model_info": {"id": "d3", "mode": "batch"}},
                {"model_name": "solo", "litellm_params": {"model": "anthropic/s", "use_in_pass_through": true}, "model_info": {"id": "s1"}},
            ],
            "routing_groups": [
                {"group_name": "rg", "models": ["g", "solo"], "routing_strategy": "priority", "model_priorities": {"g": 1, "solo": 2}},
            ],
        });
        config
            .as_object_mut()
            .unwrap()
            .extend(settings.as_object().unwrap().clone());
        let config: RouterConfig = serde_json::from_value(config).unwrap();
        Router::new(config, RouterDynamicConfig::default()).unwrap()
    }

    /// Every stage that asks the host is switched on.
    fn every_host_stage() -> serde_json::Value {
        json!({
            "routing_plugins": true,
            "callback_filters": true,
            "enable_pre_call_checks": true,
            "enable_health_check_routing": true,
            "plugins": [{"kind": "auto", "name": "p"}],
        })
    }

    /// A request that gives every stage of the chain something to filter on.
    fn asking_every_stage(name: &str, pass_through: Option<PassThrough>) -> RouterRequest {
        RouterRequest {
            model: name.to_owned(),
            pass_through,
            route_handler: Some("decisions".to_owned()),
            filter_tags: true,
            has_image: true,
            decisions_media: true,
            embedding_batch: true,
            ..RouterRequest::default()
        }
    }

    fn asking(name: &str) -> RouterRequest {
        RouterRequest {
            model: name.to_owned(),
            ..RouterRequest::default()
        }
    }

    fn counting(count_only: bool) -> SelectInputs {
        SelectInputs {
            count_only,
            ..SelectInputs::default()
        }
    }

    /// The ids the chain leaves healthy for `request`.
    async fn healthy(
        router: &Router,
        host: &Asked,
        request: &RouterRequest,
        inputs: &SelectInputs,
    ) -> Vec<String> {
        let snapshot = router.core.snapshot.load_full();
        let candidates = Call::new(&router.core, host)
            .candidates(
                &snapshot,
                &Op::from("acompletion"),
                request,
                &request.model,
                inputs,
            )
            .await
            .unwrap();
        candidates.healthy.iter().map(|d| d.id.clone()).collect()
    }

    const LISTED: Option<PassThrough> = Some(PassThrough::Listed);

    /// What each kind of pass asks the host and the cache, in order, with the candidates each
    /// stage was handed and the pool it could see.
    #[rstest]
    #[case::a_pick_of_a_group("g", None, false, &[
        "plugin_models",
        "route_incapable d1,d2",
        "cooldowns d1,d2",
        "health_states",
        "callback_filters d1,d2",
        "plugins d1,d2",
        "pre_call_facts d1,d2 of d1,d2",
        "tags d1,d2 of d1,d2",
        "text_only d1,d2",
        "media_refusing d1,d2",
        "embedding_batch d1,d2 of d1,d2",
    ], &["d1"])]
    #[case::the_count_of_a_group("g", None, true, &["cooldowns d1,d2"], &["d1", "d2"])]
    #[case::a_pick_by_id("d1", None, false, &["plugin_models", "plugins d1", "tags d1 of d1"], &["d1"])]
    #[case::the_count_by_id("d1", None, true, &[], &["d1"])]
    #[case::a_listed_pass_through_pick("g", LISTED, false, &[
        "route_incapable d1,d2,d3",
        "cooldowns d2,d3",
        "health_states",
        "pre_call_facts d2,d3 of d2,d3",
    ], &["d2", "d3"])]
    #[case::the_count_of_a_listed_pass_through("g", LISTED, true, &["cooldowns d2,d3"], &["d2"])]
    #[tokio::test]
    async fn each_kind_of_pass_runs_its_stages_in_the_chains_order(
        #[case] name: &str,
        #[case] pass_through: Option<PassThrough>,
        #[case] count_only: bool,
        #[case] asked: &[&str],
        #[case] left: &[&str],
    ) {
        let host = Asked::new();
        let request = asking_every_stage(name, pass_through);
        let healthy = healthy(
            &groups(every_host_stage()),
            &host,
            &request,
            &counting(count_only),
        )
        .await;
        assert_eq!(*host.calls.lock(), asked);
        assert_eq!(healthy, left);
    }

    /// With health-check routing driving cooldowns, a group whose deployments all cool down is
    /// still served by a pick, but not counted and not listed.
    #[rstest]
    #[case::a_pick(None, false, &["d1"])]
    #[case::the_count(None, true, &[])]
    #[case::a_listed_pass_through_pick(LISTED, false, &[])]
    #[tokio::test]
    async fn only_a_pick_keeps_a_group_that_cools_down_whole(
        #[case] pass_through: Option<PassThrough>,
        #[case] count_only: bool,
        #[case] left: &[&str],
    ) {
        let router = groups(json!({
            "enable_health_check_routing": true,
            "allowed_fails_policy": {"RateLimitErrorAllowedFails": 1},
        }));
        let host = Asked::new();
        let rate_limited = AttemptError::new(FailureKind::RateLimit, Some(429));
        let health = router.health(host.cache());
        let minute = Duration::from_secs(60);
        health
            .add_cooldown("d1", &rate_limited, minute)
            .await
            .unwrap();
        health
            .add_cooldown("d2", &rate_limited, minute)
            .await
            .unwrap();
        health
            .add_cooldown("d3", &rate_limited, minute)
            .await
            .unwrap();
        let request = RouterRequest {
            pass_through,
            ..asking("g")
        };
        let healthy = healthy(&router, &host, &request, &counting(count_only)).await;
        assert_eq!(healthy, left);
    }

    /// A priority group narrows a listed pass-through pick to its first priority. Any other pick
    /// is narrowed by the hop's order before the request's.
    #[rstest]
    #[case::a_listed_pick_of_a_priority_group("rg", LISTED, None, None, &["d2", "d3"])]
    #[case::a_listed_pick_of_a_plain_group("g", LISTED, None, None, &["d2", "d3"])]
    #[case::a_listed_pick_at_the_order_the_request_names("rg", LISTED, Some(2), None, &["s1"])]
    #[case::a_pick_at_the_hops_order("rg", None, Some(1), Some(2), &["s1"])]
    #[case::a_pick_at_the_order_the_request_names("g", None, Some(2), None, &["d2"])]
    #[tokio::test]
    async fn a_pass_is_narrowed_to_one_order(
        #[case] name: &str,
        #[case] pass_through: Option<PassThrough>,
        #[case] request_order: Option<i64>,
        #[case] hop_order: Option<i64>,
        #[case] left: &[&str],
    ) {
        let request = RouterRequest {
            pass_through,
            target_order: request_order,
            ..asking(name)
        };
        let inputs = SelectInputs {
            target_order: hop_order,
            ..SelectInputs::default()
        };
        let healthy = healthy(&groups(json!({})), &Asked::new(), &request, &inputs).await;
        assert_eq!(healthy, left);
    }

    #[rstest]
    #[case::a_healthy_deployment(Some("d2"), &["d2"])]
    #[case::a_deployment_outside_the_group(Some("s1"), &["d1"])]
    #[case::none(None, &["d1"])]
    #[tokio::test]
    async fn a_pick_goes_to_the_deployment_the_callback_filters_prefer(
        #[case] preferred: Option<&'static str>,
        #[case] left: &[&str],
    ) {
        let host = Asked {
            preferred,
            ..Asked::new()
        };
        let router = groups(json!({"callback_filters": true}));
        let healthy = healthy(&router, &host, &asking("g"), &counting(false)).await;
        assert_eq!(healthy, left);
    }

    /// An image request skips the deployments the host names as text only, unless that leaves
    /// nothing.
    #[rstest]
    #[case::some_declare_no_vision(&["d1"], &["d2"])]
    #[case::all_declare_no_vision(&["d1", "d2", "d3"], &["d1"])]
    #[tokio::test]
    async fn a_stage_drops_what_the_host_names_unless_nothing_is_left(
        #[case] text_only: &'static [&'static str],
        #[case] left: &[&str],
    ) {
        let host = Asked {
            text_only: Some(text_only),
            ..Asked::new()
        };
        let request = RouterRequest {
            has_image: true,
            ..asking("g")
        };
        let healthy = healthy(&groups(json!({})), &host, &request, &counting(false)).await;
        assert_eq!(healthy, left);
    }

    /// A batch create drops the Azure deployments that cannot run batches from a pick and from
    /// the count, a listed pass-through count among them, but not from a listed pass-through
    /// pick or a deployment named by id.
    #[rstest]
    #[case::the_count_of_a_group("g", None, true, &["d1", "d3"])]
    #[case::the_count_of_a_listed_pass_through("g", LISTED, true, &["d3"])]
    #[case::a_listed_pass_through_pick("g", LISTED, false, &["d2", "d3"])]
    #[case::the_count_by_id("d2", None, true, &["d2"])]
    #[case::a_pick_by_id("d2", None, false, &["d2"])]
    #[tokio::test]
    async fn a_batch_create_drops_what_cannot_batch_where_the_pass_filters_by_it(
        #[case] name: &str,
        #[case] pass_through: Option<PassThrough>,
        #[case] count_only: bool,
        #[case] left: &[&str],
    ) {
        let request = RouterRequest {
            pass_through,
            batch_create: true,
            ..asking(name)
        };
        let inputs = counting(count_only);
        let healthy = healthy(&groups(json!({})), &Asked::new(), &request, &inputs).await;
        assert_eq!(healthy, left);
    }

    /// A deployment named by id still passes the filters every pick runs last.
    #[rstest]
    #[case::excluded(RouterRequest { excluded_ids: vec!["d1".to_owned()], ..asking("d1") }, &[])]
    #[case::at_another_order(RouterRequest { target_order: Some(2), ..asking("d1") }, &[])]
    #[case::pinned_elsewhere(RouterRequest { pinned_deployment_id: Some("d2".to_owned()), ..asking("d1") }, &["d1"])]
    #[case::skipped_by_a_retry_with_nothing_else_left(
        RouterRequest { retry_skipped_ids: vec!["d1".to_owned()], ..asking("d1") },
        &["d1"]
    )]
    #[tokio::test]
    async fn a_deployment_named_by_id_passes_the_last_filters_of_a_pick(
        #[case] request: RouterRequest,
        #[case] left: &[&str],
    ) {
        let healthy = healthy(
            &groups(json!({})),
            &Asked::new(),
            &request,
            &counting(false),
        )
        .await;
        assert_eq!(healthy, left);
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

    fn declared(id: &str, params: serde_json::Value, mode: &str) -> Arc<Deployment> {
        let spec: DeploymentSpec = serde_json::from_value(json!({
            "model_name": "g", "litellm_params": params, "model_info": {"id": id, "mode": mode},
        }))
        .unwrap();
        Arc::new(Deployment::from_spec(spec).unwrap())
    }

    fn chat() -> Arc<Deployment> {
        declared("chat", json!({"model": "azure/chat"}), "chat")
    }

    fn azure_batch(id: &str) -> Arc<Deployment> {
        declared(id, json!({"model": "azure/batch"}), "batch")
    }

    fn ids(deployments: &[Arc<Deployment>]) -> Vec<&str> {
        deployments.iter().map(|d| d.id.as_str()).collect()
    }

    #[rstest]
    #[case::an_azure_model(json!({"model": "azure/batch"}), vec!["chat"])]
    #[case::an_azure_ai_model(json!({"model": "azure_ai/batch"}), vec!["chat"])]
    #[case::azure_named_by_custom_llm_provider(
        json!({"model": "batch", "custom_llm_provider": "azure"}),
        vec!["chat"]
    )]
    #[case::another_provider_serves_more_than_batches(json!({"model": "openai/batch"}), vec!["chat", "batch"])]
    fn a_call_outside_the_batch_api_skips_azure_batch_only_deployments(
        #[case] batch_params: serde_json::Value,
        #[case] expected: Vec<&str>,
    ) {
        let pool = vec![chat(), declared("batch", batch_params, "batch")];
        let kept = without_batch_only_deployments(
            skips_batch_only(&RouterRequest::default(), &pool),
            pool.clone(),
        );
        assert_eq!(ids(&kept), expected);
    }

    #[rstest]
    #[case::a_batch_create(
        RouterRequest { batch_create: true, ..RouterRequest::default() },
        vec![chat(), azure_batch("batch")],
        vec![chat(), azure_batch("batch")],
        vec!["chat", "batch"]
    )]
    #[case::a_batch_or_file_call(
        RouterRequest { batch_api: true, ..RouterRequest::default() },
        vec![chat(), azure_batch("batch")],
        vec![chat(), azure_batch("batch")],
        vec!["chat", "batch"]
    )]
    #[case::a_group_with_only_batch_deployments(
        RouterRequest::default(),
        vec![azure_batch("b1"), azure_batch("b2")],
        vec![azure_batch("b1"), azure_batch("b2")],
        vec!["b1", "b2"]
    )]
    #[case::a_cooling_chat_deployment_leaves_nothing(
        RouterRequest::default(),
        vec![chat(), azure_batch("batch")],
        vec![azure_batch("batch")],
        vec![]
    )]
    fn batch_only_deployments_stay_for_batch_calls_and_batch_only_groups(
        #[case] request: RouterRequest,
        #[case] all: Vec<Arc<Deployment>>,
        #[case] healthy: Vec<Arc<Deployment>>,
        #[case] expected: Vec<&str>,
    ) {
        let kept = without_batch_only_deployments(skips_batch_only(&request, &all), healthy);
        assert_eq!(ids(&kept), expected);
    }
}
