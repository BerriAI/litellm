//! Routing for LiteLLM: decide which deployment serves a request, and drive retries, fallbacks
//! and cooldowns around the attempts.
//!
//! This crate defines the boundary `litellm.Router` is moving behind. Today the Python router in
//! `litellm/router.py` implements all of it; this crate is where the Rust core will implement the
//! same interface. Until then it holds this overview and nothing else, so the interface is written
//! down in code before either side changes.
//!
//! # The public surface
//! Code outside the router uses `litellm.Router` through two kinds of members and nothing else:
//! - The endpoint methods (`acompletion`, `aembedding`, `aresponses`, `aanthropic_messages`, the
//!   files, batches, assistants and vector store wrappers, 37 in all). Each is a thin wrapper over
//!   `call(op, request)`.
//! - Eight entry points: `registry()`, `settings()`, `health()`, `select()`, `apply()`,
//!   `observe()`, `register_plugin()` and `retire()`.
//!
//! Every other member that outside code uses today has a named replacement among those. The old
//! names stay available for about a month through a `DeprecatedRouterMembers` mixin, which
//! implements them over the entry points and warns on use. A CI check
//! (`tests/code_coverage_tests/router_surface_check.py`) keeps an allowlist of the members outside
//! code touches; the list may only shrink.
//!
//! # Structure
//! - `Registry`: the configured deployments, indexed by id, model group, alias, wildcard pattern
//!   and team. Immutable and versioned. Seven queries: `deployment(id)`, `deployments(filter)`,
//!   `names(filter)`, `resolve(name, scope)`, `group(name)`, `access_groups()` and
//!   `credentials(id)`. "Registry" is the router's view of what is configured; per-model pricing
//!   and capabilities are the model catalog (`litellm-model-catalog`), which the router reads
//!   through its dynamic config.
//! - `RouterConfig` / `RouterDynamicConfig`: everything the router needs to know, passed in. The
//!   router never reads process globals. Static config (model list, strategy, retry, timeout and
//!   cooldown policy, Redis settings, plugins) is fixed for the router's life. Dynamic config
//!   (pricing, model aliases, fallbacks, retry counts, callbacks) is a versioned snapshot read
//!   once per call.
//! - `Settings`: one typed, frozen view of the configuration a caller may read.
//! - `Health`: cooldowns, blocked deployments and usage, kept in shared cache keys so Python and
//!   Rust workers see the same state. `report(id, outcome)` records outcomes the router did not
//!   see itself (a health check, an interceptor that marks a deployment failed).
//! - `select(model, request)`: resolve a name, filter eligible deployments, pick one with the
//!   configured strategy, without calling it.
//! - `apply(changes)`: one `ChangeSet` (upserts, deletes, settings), applied atomically, producing
//!   a new registry version.
//! - `call(op, request)`: select, run the attempt, and retry or fall back until it succeeds or the
//!   policy is exhausted.
//!
//! # Hosts: the only boundary
//! The core reaches outside itself only through [`Host`], and every attempt runs through
//! `Host::invoke`: the core picks the deployment, the host runs the call, the core records the
//! outcome. There is no second, "native" attempt path in the core. Which code runs an attempt is
//! the host's business:
//! - `PythonHost` (inside a Python process) builds the request for the picked deployment and calls
//!   `litellm.<op>`. It never calls back into a Python router.
//!
//! An op or provider a host cannot run fails `Host::supports` when the router is built or
//! changed, never partway through a request.
//!
//! The core also builds its deployments through the host ([`build`]): for each configured entry
//! it finds the secret references and asks `secrets`, then `validate`, `provider` and `added`.
//!
//! Besides `invoke` and `supports`, a host provides `select` for strategies and plugins not yet
//! ported, `allow_fallback` for access and budget checks, `refuses_fallbacks` for a chain's dict
//! targets, `sleep`, `observe` for routing events, and `cache`, a DualCache-style store that
//! holds all routing state under the same keys, encodings and TTLs the Python router uses. The
//! core writes cooldowns, allowed-fails counters and usage itself, through that cache.
//!
//! # Plugins
//! Routing strategies, pre-call checks and sub-routers (auto, complexity, quality, adaptive)
//! register through `register_plugin` and see a `RoutingContext` (registry, settings, health and
//! the current request), never the router itself.
//!
//! # Modules
//! - [`registry`]: deployments and their indices, `deployment_id`, `ChangeSet`.
//! - [`config`] / [`settings`]: static and dynamic config, and the frozen `Settings` view.
//! - [`cache`] / [`health`]: the shared routing cache, cooldowns and usage.
//! - [`select`]: resolve, filter chain, simple-shuffle.
//! - [`call`]: retries and fallbacks.
//! - [`eligibility`]: the endpoints a deployment refuses to serve.
//! - [`precall`]: the `enable_pre_call_checks` filter.
//! - [`model_info`]: model info merged from the cost map, asked of the host at query time.
//! - [`host`]: the `Host` boundary.

pub mod access;
pub mod build;
pub mod cache;
pub mod call;
pub mod config;
pub mod eligibility;
pub mod error;
pub mod fallback;
pub mod group_info;
pub mod health;
pub mod health_routing;
pub mod host;
pub mod host_index;
pub mod lookup;
pub mod mock;
pub mod model_info;
pub mod outcome;
pub mod precall;
pub mod prerouting;
pub mod ptu;
pub mod registry;
pub mod resolve;
pub mod retry;
pub mod routing_groups;
pub mod routing_plugins;
pub mod select;
pub mod settings;
pub mod strategy;
pub mod weights;

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use arc_swap::ArcSwap;
use parking_lot::Mutex;

pub use crate::{
    cache::{InMemoryRouterCache, RouterCache},
    call::RouterResponse,
    config::{Op, RouterConfig, RouterDynamicConfig, RoutingStrategy},
    error::{AttemptError, ConfigError, Error, FailureKind, Payload, Unsupported},
    health::{Clock, Health, Outcome, SystemClock},
    host::{
        Attempt, Blocked, BuildAsk, BuildPath, Built, Host, Output, Provided, RetryCause,
        RouterEvent, RouterRequest, RoutingContext, SecretAt, SecretRef, SelectStage,
    },
    outcome::{OutcomeCallbacks, OutcomeWrite},
    prerouting::{PreRoute, PreRouted},
    registry::{ChangeSet, Deployment, DeploymentSpec, Registry, RegistryVersion},
    select::{Randomness, SeededRandomness, ThreadRandomness},
    settings::Settings,
};
use crate::{
    health::{GroupShape, routing_group_has_alternatives, team_model_has_alternatives},
    registry::DeploymentFilter,
    select::SelectInputs,
};

/// The registry and settings one call reads, swapped atomically by `apply`.
pub(crate) struct RouterStateSnapshot {
    pub registry: Arc<Registry>,
    pub settings: Arc<Settings>,
}

impl RouterStateSnapshot {
    /// `_get_routing_context` for a pick of `name`: unless the request overrides the strategy,
    /// a stale member map naming a group no longer configured raises `KeyError(group)`.
    pub(crate) fn missing_routing_group(
        &self,
        request: &RouterRequest,
        name: &str,
    ) -> Result<(), Error> {
        if strategy::request_override(request.routing_strategy.as_deref()).is_some() {
            return Ok(());
        }
        match self.registry.missing_member_group(name) {
            Some(group) => Err(Error::MissingRoutingGroup {
                group: group.to_owned(),
            }),
            None => Ok(()),
        }
    }
}

pub(crate) struct Core {
    host: Arc<dyn Host>,
    rng: Arc<dyn Randomness>,
    health: Health,
    config: RouterConfig,
    snapshot: ArcSwap<RouterStateSnapshot>,
    writer: Mutex<()>,
    retired: AtomicBool,
}

impl Core {
    /// Reject what this host cannot serve, before any request.
    fn validate(&self, registry: &Registry, settings: &Settings) -> Result<(), ConfigError> {
        let can_select = self.host.can_select();
        let strategies = std::iter::once(&settings.routing_strategy).chain(
            registry
                .routing_groups()
                .iter()
                .map(|group| &group.routing_strategy),
        );
        for strategy in strategies {
            if !strategy.is_native() && !can_select {
                return Err(Unsupported::Strategy(strategy.clone()).into());
            }
        }
        if let Some(plugin) = settings.plugins.first()
            && !can_select
        {
            return Err(Unsupported::Plugin(plugin.name.clone()).into());
        }
        for deployment in registry.deployments(&DeploymentFilter::default()) {
            self.validate_deployment(&deployment)?;
        }
        Ok(())
    }

    /// Reject a deployment whose provider this host cannot run a served op on.
    fn validate_deployment(&self, deployment: &Deployment) -> Result<(), ConfigError> {
        for op in &self.config.served_ops {
            if !self.host.supports(op, deployment.provider()) {
                return Err(Unsupported::Op {
                    op: op.clone(),
                    provider: deployment.provider().to_owned(),
                }
                .into());
            }
        }
        Ok(())
    }
}

/// How a reported failure is recorded.
#[derive(Clone, Debug, Default)]
pub struct FailureReport {
    /// Cool down for the deployment's own `cooldown_time` (then the provider's retry-after),
    /// as the failure callback does, rather than the router default.
    pub deployment_cooldown: bool,
    /// The `cooldown_time` the failed call's own params carried (`model_info`, then
    /// `litellm_params`), which wins over the provider's retry-after for a callback report.
    pub cooldown_time: Option<f64>,
    /// Count the failure only: its status is one no cooldown rule reads (an empty status).
    pub count_only: bool,
    /// The model group the request was for, whose fallbacks keep a lone deployment cooling.
    pub requested_group: Option<String>,
    /// Cool down without counting the failure (`set_cooldown_deployments` alone).
    pub uncounted: bool,
    /// Also count the failed request against the deployment's per-minute requests
    /// (`async_deployment_callback_on_failure`).
    pub counts_request: bool,
    /// The model the request's usage is keyed by (metadata's `deployment`, which a wildcard route
    /// fills), when it is not the deployment's own.
    pub usage_model: Option<String>,
    /// The failure callbacks of a logged failure; a report no callback makes keeps the default.
    pub callbacks: OutcomeCallbacks,
}

/// How a reported success is recorded.
#[derive(Clone, Debug, Default)]
pub struct SuccessReport {
    /// See [`host::Output::tracks_usage`].
    pub tracks_usage: Option<bool>,
    /// See [`FailureReport::usage_model`].
    pub usage_model: Option<String>,
    /// See [`FailureReport::callbacks`].
    pub callbacks: OutcomeCallbacks,
}

/// Builds a [`Router`], with test seams.
pub struct RouterBuilder {
    config: RouterConfig,
    dynamic: RouterDynamicConfig,
    host: Arc<dyn Host>,
    rng: Arc<dyn Randomness>,
    clock: Arc<dyn Clock>,
}

impl RouterBuilder {
    #[must_use]
    pub fn randomness(self, rng: Arc<dyn Randomness>) -> Self {
        Self { rng, ..self }
    }

    #[must_use]
    pub fn clock(self, clock: Arc<dyn Clock>) -> Self {
        Self { clock, ..self }
    }

    pub fn build(self) -> Result<Router, ConfigError> {
        let registry = Registry::build(
            self.config.model_list.clone(),
            self.dynamic.model_alias_map.clone(),
            self.config.routing_groups.clone(),
        )?
        .with_member_groups(self.config.routing_group_members.clone());
        let settings = Settings::new(&self.config, Arc::new(self.dynamic));
        let core = Core {
            health: Health::new(self.host.cache(), self.clock),
            host: self.host,
            rng: self.rng,
            config: self.config,
            snapshot: ArcSwap::from_pointee(RouterStateSnapshot {
                registry: Arc::new(Registry::default()),
                settings: Arc::new(settings.clone()),
            }),
            writer: Mutex::new(()),
            retired: AtomicBool::new(false),
        };
        core.validate(&registry, &settings)?;
        core.snapshot.store(Arc::new(RouterStateSnapshot {
            registry: Arc::new(registry),
            settings: Arc::new(settings),
        }));
        Ok(Router {
            core: Arc::new(core),
        })
    }
}

/// The router: the entry points `litellm.Router` delegates to.
#[derive(Clone)]
pub struct Router {
    core: Arc<Core>,
}

impl Router {
    pub fn new(
        config: RouterConfig,
        dynamic: RouterDynamicConfig,
        host: Arc<dyn Host>,
    ) -> Result<Self, ConfigError> {
        Self::builder(config, dynamic, host).build()
    }

    pub fn builder(
        config: RouterConfig,
        dynamic: RouterDynamicConfig,
        host: Arc<dyn Host>,
    ) -> RouterBuilder {
        RouterBuilder {
            config,
            dynamic,
            host,
            rng: Arc::new(ThreadRandomness),
            clock: Arc::new(SystemClock),
        }
    }

    fn live(&self) -> Result<(), Error> {
        if self.core.retired.load(Ordering::Acquire) {
            return Err(Error::Retired);
        }
        Ok(())
    }

    /// Select, run the attempt, and retry or fall back until it succeeds or the policy is
    /// exhausted.
    pub async fn call(&self, op: Op, request: RouterRequest) -> Result<RouterResponse, Error> {
        self.live()?;
        self.core.call(op, Arc::new(request)).await
    }

    /// Resolve, filter and pick a deployment without calling it.
    pub async fn select(&self, op: &Op, request: &RouterRequest) -> Result<Arc<Deployment>, Error> {
        self.live()?;
        let snapshot = self.core.snapshot.load_full();
        routing_plugins::refuse_sync_pick(
            snapshot.settings.routing_plugins,
            &snapshot.settings.routing_strategy,
            request,
        )?;
        let routed = self
            .core
            .pre_routed(PreRoute {
                op,
                request,
                name: &request.model,
                fallback_depth: 0,
                retry: 0,
            })
            .await?;
        let name = match routed {
            PreRouted::Model(name) | PreRouted::Bound { model: name, .. } => name,
            PreRouted::Deployment(deployment) => return Ok(deployment),
        };
        snapshot.missing_routing_group(request, &name)?;
        let candidates = self
            .core
            .candidates(&snapshot, op, request, &name, &SelectInputs::default())
            .await?;
        // An empty candidate list names what the request could route to before the cooldown filter.
        if candidates.healthy.is_empty() && !candidates.kind.is_single() {
            return Err(candidates.none_healthy(&snapshot.registry));
        }
        self.core.pick(&snapshot, op, request, &candidates).await
    }

    /// The deployments a pick for `request` would choose among: resolved, team- and
    /// cooldown-filtered, through the plugins and the lowest `order`.
    pub async fn healthy(
        &self,
        op: &Op,
        request: &RouterRequest,
    ) -> Result<select::Candidates, Error> {
        self.live()?;
        let snapshot = self.core.snapshot.load_full();
        let candidates = self
            .core
            .candidates(
                &snapshot,
                op,
                request,
                &request.model,
                &SelectInputs::default(),
            )
            .await?;
        // `_core_async_get_healthy_deployments` raises when nothing is left.
        if candidates.healthy.is_empty() {
            return Err(candidates.none_healthy(&snapshot.registry));
        }
        Ok(candidates)
    }

    pub fn registry(&self) -> Arc<Registry> {
        self.core.snapshot.load().registry.clone()
    }

    pub fn settings(&self) -> Arc<Settings> {
        self.core.snapshot.load().settings.clone()
    }

    pub fn health(&self) -> &Health {
        &self.core.health
    }

    /// Record an outcome the router did not see itself, such as a health check. Returns whether
    /// the deployment was cooled down.
    pub async fn report(&self, id: &str, outcome: Outcome) -> Result<bool, Error> {
        match outcome {
            Outcome::Success { total_tokens } => self
                .report_success(id, total_tokens as f64, 1.0, SuccessReport::default())
                .await
                .map(|_| false),
            Outcome::Failure(failure) => self
                .report_failure(id, &failure, FailureReport::default())
                .await
                .map(|cooldown| cooldown.is_some()),
        }
    }

    /// A success a logging callback saw (`deployment_callback_on_success`): counted for a
    /// configured deployment, with its tokens and requests when it tracks usage. Returns the
    /// tokens counter's key when it added usage (`_increment_deployment_usage`), else `None`.
    pub async fn report_success(
        &self,
        id: &str,
        total_tokens: f64,
        requests: f64,
        report: SuccessReport,
    ) -> Result<Option<String>, Error> {
        report
            .callbacks
            .record(
                OutcomeWrite::Success,
                self.reported_success(id, total_tokens, requests, &report),
            )
            .await
    }

    async fn reported_success(
        &self,
        id: &str,
        total_tokens: f64,
        requests: f64,
        report: &SuccessReport,
    ) -> Result<Option<String>, Error> {
        let snapshot = self.core.snapshot.load_full();
        let Some(deployment) = snapshot.registry.deployment(id) else {
            return Ok(None);
        };
        self.core.health.count_success(id).await?;
        if !report
            .tracks_usage
            .unwrap_or_else(|| deployment.tracks_usage())
        {
            return Ok(None);
        }
        self.core
            .health
            .add_usage(
                &deployment,
                report.usage_model.as_deref(),
                total_tokens,
                requests,
            )
            .await
    }

    /// `sync_deployment_callback_on_success`: count a success against `id`, configured or not,
    /// and return the counter's key.
    pub async fn count_success(
        &self,
        id: &str,
        callbacks: OutcomeCallbacks,
    ) -> Result<Option<String>, Error> {
        callbacks
            .record(OutcomeWrite::Success, async {
                self.core.health.count_success(id).await.map(Some)
            })
            .await
    }

    /// A failure the router did not see itself: counted against `id` (known or not), then
    /// cooled down by the rules (`record_failure`, `deployment_callback_on_failure`). Returns the
    /// cooldown applied, if any.
    pub async fn report_failure(
        &self,
        id: &str,
        failure: &AttemptError,
        report: FailureReport,
    ) -> Result<Option<std::time::Duration>, Error> {
        report
            .callbacks
            .record(
                OutcomeWrite::Failure,
                self.reported_failure(id, failure, &report),
            )
            .await
    }

    async fn reported_failure(
        &self,
        id: &str,
        failure: &AttemptError,
        report: &FailureReport,
    ) -> Result<Option<std::time::Duration>, Error> {
        if !report.uncounted && !failure.outside_deployment {
            self.core.health.count_failure(id).await?;
        }
        let snapshot = self.core.snapshot.load_full();
        let Some(deployment) = snapshot.registry.deployment(id) else {
            return Ok(None);
        };
        if report.counts_request {
            self.core
                .health
                .count_failed_request(&deployment, report.usage_model.as_deref())
                .await?;
        }
        if failure.outside_deployment {
            return Ok(None);
        }
        // An empty status is no cooldown (`_is_cooldown_required`), unless the deployment's
        // own policy names the error.
        if report.count_only
            && crate::health::deployment_allowed_fails(
                deployment.model_info.allowed_fails_policy.as_ref(),
                failure,
            )
            .is_none()
        {
            return Ok(None);
        }
        let cooldown = if report.deployment_cooldown {
            report
                .cooldown_time
                .filter(|seconds| *seconds >= 0.0)
                .and_then(|seconds| std::time::Duration::try_from_secs_f64(seconds).ok())
                .or(failure.retry_after)
                .unwrap_or(snapshot.settings.cooldown_time)
        } else {
            snapshot.settings.cooldown_time
        };
        let shape = GroupShape {
            single_deployment: snapshot
                .registry
                .group(&deployment.model_name)
                .is_some_and(|group| group.len() == 1)
                && !report
                    .requested_group
                    .as_deref()
                    .is_some_and(|group| routing_group_has_alternatives(&snapshot.registry, group))
                && !team_model_has_alternatives(&snapshot.registry, &deployment),
            has_fallback_group: snapshot
                .settings
                .dynamic
                .fallback_targets(&[report
                    .requested_group
                    .as_deref()
                    .unwrap_or(&deployment.model_name)])
                .is_some(),
        };
        self.core
            .health
            .cool_down_after(&snapshot.settings, &deployment, shape, failure, cooldown)
            .await
    }

    /// Apply one change set atomically and return the new registry version.
    pub fn apply(&self, changes: ChangeSet) -> Result<RegistryVersion, ConfigError> {
        let _writer = self.core.writer.lock();
        let current = self.core.snapshot.load_full();
        let aliases = changes
            .settings
            .as_ref()
            .map(|dynamic| dynamic.model_alias_map.clone());
        let registry = current.registry.apply(&changes, aliases)?;
        let settings = match changes.settings {
            Some(dynamic) => Settings::new(&self.core.config, Arc::new(dynamic)),
            None => Settings::clone(&current.settings),
        };
        self.core.validate(&registry, &settings)?;
        let version = registry.version();
        self.core.snapshot.store(Arc::new(RouterStateSnapshot {
            registry: Arc::new(registry),
            settings: Arc::new(settings),
        }));
        Ok(version)
    }

    /// `get_router_model_info` for one deployment, its cost-map reads asked of the host now
    /// ([`model_info::router_model_info`]).
    pub async fn router_model_info(
        &self,
        query: &model_info::RouterModelInfoQuery,
    ) -> Result<Option<model_info::MergedModelInfo>, Error> {
        model_info::router_model_info(self.core.host.as_ref(), query).await
    }

    /// `_set_model_group_info` for the deployments `query` names ([`group_info::model_group_info`]).
    pub async fn model_group_info(
        &self,
        query: &group_info::GroupInfoQuery,
    ) -> Result<Option<group_info::GroupInfo>, Error> {
        group_info::model_group_info(self.core.host.as_ref(), query).await
    }

    /// `get_configured_token_limits` for deployment `id` ([`group_info::configured_token_limits`]).
    pub async fn configured_token_limits(
        &self,
        id: serde_json::Value,
        configured: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(Option<i64>, Option<i64>), Error> {
        group_info::configured_token_limits(self.core.host.as_ref(), id, configured).await
    }

    /// `get_model_listing_info` over a listed name's deployments ([`group_info::model_listing_info`]).
    pub async fn model_listing_info(
        &self,
        deployments: &[serde_json::Value],
    ) -> Result<group_info::ListingInfo, Error> {
        group_info::model_listing_info(self.core.host.as_ref(), deployments).await
    }

    /// `get_deployment_model_info` for deployment `model_id` serving `model_name`
    /// ([`model_info::deployment_model_info`]).
    pub async fn deployment_model_info(
        &self,
        model_id: &str,
        model_name: &serde_json::Value,
    ) -> Result<Option<serde_json::Map<String, serde_json::Value>>, Error> {
        model_info::deployment_model_info(self.core.host.as_ref(), model_id, model_name).await
    }

    /// Build `entries` through the host ([`build::build`]), installing each deployment it
    /// accepts as soon as it does, as `install` says: a replace empties the registry first (and
    /// takes `set_model_list`'s `model_names` once every entry is built), an append or upsert
    /// adds at the end. When a step fails, what was installed before it stays, as the Python
    /// router keeps the deployments it added before the one that raised.
    ///
    /// The build installs into a registry of its own, indexed in place, and serves it once it
    /// ends (or fails), so n deployments build in O(n). A write that lands while it runs (a step
    /// answer that applies a changeset) is kept: what the build installed is added over it.
    pub async fn build(
        &self,
        path: BuildPath,
        entries: Vec<serde_json::Map<String, serde_json::Value>>,
        resolve_secrets: bool,
        install: build::Install,
    ) -> Result<Vec<build::BuiltDeployment>, Error> {
        let model_names: Vec<String> = entries
            .iter()
            .map(|entry| match entry.get("model_name") {
                Some(serde_json::Value::String(name)) => name.clone(),
                None | Some(serde_json::Value::Null) => "None".to_owned(),
                Some(other) => other.to_string(),
            })
            .collect();
        let replace = install == build::Install::Replace;
        let base = self.core.snapshot.load_full().registry.clone();
        let mut working = if replace {
            base.emptied()
        } else {
            Registry::clone(&base)
        };
        let mut installed: Vec<DeploymentSpec> = Vec::new();
        let add = path == BuildPath::Add;
        let core = &self.core;
        let built = build::build(
            core.host.as_ref(),
            path,
            entries,
            resolve_secrets,
            |deployment| {
                working.push_built(deployment.spec.clone(), add)?;
                installed.push(deployment.spec.clone());
                core.validate_deployment(working.last().expect("just installed"))
                    .map_err(Error::from)
            },
        )
        .await;
        if built.is_ok() && replace && path == BuildPath::Create {
            working.set_model_names(model_names);
        }
        self.serve_built(&base, working, installed, add)?;
        built
    }

    /// Serve the registry a build made from `base`. When another write replaced `base` while
    /// the build ran, the build's deployments are added over that write instead.
    fn serve_built(
        &self,
        base: &Arc<Registry>,
        built: Registry,
        installed: Vec<DeploymentSpec>,
        add: bool,
    ) -> Result<(), ConfigError> {
        let _writer = self.core.writer.lock();
        let current = self.core.snapshot.load_full();
        let registry = if Arc::ptr_eq(&current.registry, base) {
            built
        } else {
            let mut next = Registry::clone(&current.registry);
            for spec in installed {
                next.push_built(spec, add)?;
            }
            next
        };
        self.core.snapshot.store(Arc::new(RouterStateSnapshot {
            registry: Arc::new(registry),
            settings: current.settings.clone(),
        }));
        Ok(())
    }

    /// Keep the providers `from`'s host resolved for the deployments this router serves too (a
    /// router rebuilt over the same deployments for new static settings).
    pub fn adopt_providers(&self, from: &Self) {
        let _writer = self.core.writer.lock();
        let current = self.core.snapshot.load_full();
        let registry = current
            .registry
            .with_providers_of(&from.core.snapshot.load().registry);
        self.core.snapshot.store(Arc::new(RouterStateSnapshot {
            registry: Arc::new(registry),
            settings: current.settings.clone(),
        }));
    }

    /// Stop serving: every later call and selection fails with [`Error::Retired`].
    pub fn retire(&self) {
        self.core.retired.store(true, Ordering::Release);
    }
}
