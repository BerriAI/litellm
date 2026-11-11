//! The host behind `RouterCoreHandle`. Rust never awaits Python: each host operation the call
//! loop needs becomes a [`HostCall`] parked in the call's [`Exchange`], and the call future stays
//! pending until Python runs it and hands the [`HostCallResponse`] back through [`Driver::resume`].
//!
//! The driver polls the call future with a no-op waker while the call's exchange is bound to the
//! current thread, so a host future finds its exchange without the router passing it through.
//!
//! The routing cache is a host operation too: [`StepCache`] parks every [`RouterCache`] op as a
//! `cache` step, and Python runs it against the router's own `DualCache` (memory plus Redis), so
//! Python and Rust read and write the same cooldowns, fail counters and usage.

use std::{
    cell::RefCell,
    future::{Future, poll_fn},
    pin::Pin,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    task::{Context, Poll, Waker},
    time::Duration,
};

use async_trait::async_trait;
use litellm_router_core::{
    Attempt, AttemptError, Blocked, BuildAsk, BuildPath, Built, CallbackFiltered, Deployment,
    Error, Host, Output, Payload, PreRoute, PreRouted, Provided, Randomness, RouterCache,
    RouterEvent, RouterRequest, RouterResponse, RoutingContext, SecretRef, SelectStage,
    ThreadRandomness, build::BuiltDeployment, host::ModelInfoLookup, model_info::MergedModelInfo,
    precall::PreCallFacts,
};
use pyo3::{PyResult, Python, types::PyAnyMethods};
use serde_json::{Map, Value};

/// The `kind` of a step dict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display, strum::IntoStaticStr)]
pub(crate) enum StepKind {
    #[strum(serialize = "attempt")]
    Attempt,
    #[strum(serialize = "sleep")]
    Sleep,
    #[strum(serialize = "allow_fallback")]
    AllowFallback,
    #[strum(serialize = "refuse_fallbacks")]
    RefuseFallbacks,
    #[strum(serialize = "select")]
    Select,
    #[strum(serialize = "pre_route")]
    PreRoute,
    #[strum(serialize = "cache")]
    Cache,
    #[strum(serialize = "secrets")]
    Secrets,
    #[strum(serialize = "validate")]
    Validate,
    #[strum(serialize = "provider")]
    Provider,
    #[strum(serialize = "added")]
    Added,
    #[strum(serialize = "model_info")]
    ModelInfo,
    #[strum(serialize = "done")]
    Done,
    #[strum(serialize = "failed")]
    Failed,
}

/// Which selection stage Python is asked to run.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Stage {
    /// Pre-call checks and sub-routers: return the deployments that remain.
    Plugins,
    /// A strategy router-core does not implement: return the candidates with the pick first.
    /// When candidates share an id, `records` holds each candidate's deployment, in order, and the
    /// host answers with the pick's position, so a pick splits over every entry as Python's does.
    Strategy {
        strategy: String,
        records: Vec<Value>,
    },
    /// Tag routing: return the deployments that remain.
    Tags,
    /// `enable_pre_call_checks`: return the facts the filter decides by.
    PreCall,
    /// `_deployment_declares_no_vision`: return the candidates that declare no vision support.
    TextOnly,
    /// `_deployment_serves_route`: return the candidates that cannot serve `handler`'s call.
    RouteIncapable { handler: String },
    /// `_decisions_deployment_refuses_media`: return the candidates that refuse the call's media.
    MediaRefusing,
    /// `deployments_admitting_embedding_batch`: return the candidates the embedding pick keeps.
    EmbeddingBatch,
    /// The override strategy's pre-call check on a deployment named by id.
    OverrideCheck(String),
    /// The routing plugins: run them when `run`, then return the candidate models they left.
    RoutingPlugins { run: bool },
    /// The host's `CustomLogger` deployment filters: return the ids that remain, or a failure.
    /// They read the ids this request already failed on, as Python's request kwargs carry them.
    CallbackFilters {
        excluded_ids: Vec<String>,
        skipped_ids: Vec<String>,
        target_order: Option<i64>,
    },
    /// `f"{provider}/{model}"` for a name no wildcard pattern matches as spelled: return it as
    /// the one entry of `models`.
    ProviderRequest,
}

/// The `op` of a `cache` step: one per [`RouterCache`] method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display, strum::IntoStaticStr)]
pub(crate) enum CacheOp {
    /// `async_get_cache(key)`: answered with the stored JSON value, or `None`.
    #[strum(serialize = "get")]
    Get,
    /// `async_set_cache(key, value, ttl)`: answered with `None`.
    #[strum(serialize = "set")]
    Set,
    /// `async_increment(key, value, ttl)`: answered with the new number. The TTL applies when
    /// the key is created.
    #[strum(serialize = "increment")]
    Increment,
    /// `async_increment_usage(keys, value, ttl)`: `value` is one amount per key, added as
    /// `_increment_deployment_usage` adds them. Answered with `None`.
    #[strum(serialize = "increment_usage")]
    IncrementUsage,
    /// `async_batch_get_cache(keys)`: answered with one JSON value or `None` per key, in order.
    #[strum(serialize = "batch_get")]
    BatchGet,
    /// `async_delete_cache(key)`: answered with `None`.
    #[strum(serialize = "delete")]
    Delete,
}

/// Python's outcome of a `cache` step.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CacheOutcome {
    /// `get`: the stored value (`None` when missing).
    Value(Option<Value>),
    /// `batch_get`: one entry per key.
    Values(Vec<Option<Value>>),
    /// `increment`: the counter after the increment.
    Number(f64),
    /// `set`, `delete` and `increment_usage`.
    Stored,
}

impl CacheOutcome {
    /// The op this outcome answers (`Stored` answers both `set` and `delete`).
    fn answers(&self, op: CacheOp) -> bool {
        matches!(
            (self, op),
            (Self::Value(_), CacheOp::Get)
                | (Self::Values(_), CacheOp::BatchGet)
                | (Self::Number(_), CacheOp::Increment)
                | (
                    Self::Stored,
                    CacheOp::Set | CacheOp::Delete | CacheOp::IncrementUsage
                )
        )
    }
}

/// A fallback chain's dict targets, which can carry an api key.
#[derive(Clone, PartialEq, veil::Redact)]
pub(crate) struct DictTargets(#[redact(with = "[REDACTED]")] pub(crate) Vec<Map<String, Value>>);

/// What a call is blocked on, for Python to run.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum HostCall {
    Attempt {
        deployment_id: String,
        /// The picked deployment as configured: `model_name`, `litellm_params`, `model_info`
        /// (with its `id`). The host builds the request from it; it never looks the deployment up
        /// in a Python router.
        deployment: Value,
        /// The name the attempt resolved: the requested name or the fallback target.
        model: String,
        model_group: String,
        /// The group a pre-routing hook selected for `model`, when it differs.
        routed: Option<String>,
        /// The `order` level of a same-group order fallback.
        target_order: Option<i64>,
        /// Every deployment `model` resolved to for this caller, the picked one included.
        candidate_ids: Vec<String>,
        retry: u32,
        fallback_depth: u32,
        fallback_level: u32,
        timeout: Option<f64>,
        /// Request params the fallback walk's dict targets merged in.
        params: Option<Box<Value>>,
        /// The walk level whose request metadata the attempt carries.
        scope: u32,
        max_retries: u32,
        model_group_size: usize,
    },
    Sleep {
        seconds: f64,
    },
    AllowFallback {
        target: String,
    },
    RefuseFallbacks {
        /// The walk level whose chain the targets are.
        level: u32,
        targets: DictTargets,
    },
    Select {
        stage: Stage,
        model_group: String,
        /// The name the request resolved from, which strategies key their state by.
        name: String,
        /// Whose selector a strategy reads: `default`, `group` (with `group`) or `override`.
        scope: &'static str,
        group: Option<String>,
        /// The deployments the name resolved to, before health filters (the tag stage reads them).
        pool: Vec<String>,
        candidates: Vec<String>,
    },
    /// Pre-routing before a pick resolves `model`: answered with the model group to resolve
    /// instead (a string), a custom strategy's deployment document (a mapping), or `None`.
    PreRoute {
        model: String,
        fallback_depth: u32,
        retry: u32,
    },
    /// One [`RouterCache`] op against the host's cache. `key` is set for every op but
    /// `batch_get`, which sets `keys`; `value` is set for `set` (the JSON to store) and
    /// `increment` (the amount); `ttl` is in seconds.
    Cache {
        op: CacheOp,
        key: Option<String>,
        keys: Option<Vec<String>>,
        value: Option<Value>,
        ttl: Option<f64>,
        /// The op reads or writes this worker's in-memory tier only (`local_only=True`).
        local: bool,
    },
    /// The deployment build's `os.environ/` references: answered with one value per ref.
    Secrets {
        index: usize,
        refs: Vec<SecretRef>,
    },
    /// The deployment build's `validate`, `provider` and `added` steps for one deployment.
    Build {
        kind: StepKind,
        path: BuildPath,
        index: usize,
        entry: Value,
        declared_id: Option<String>,
        duplicate_id: bool,
        /// The provider the `provider` step resolved (`added` only).
        provider: Option<String>,
    },
    /// One lookup of a model-info query ([`ModelInfoLookup`], as its `lookup` field and
    /// arguments): answered as the lookup documents.
    ModelInfo {
        lookup: Value,
    },
}

impl HostCall {
    pub(crate) fn kind(&self) -> StepKind {
        match self {
            Self::Attempt { .. } => StepKind::Attempt,
            Self::Sleep { .. } => StepKind::Sleep,
            Self::AllowFallback { .. } => StepKind::AllowFallback,
            Self::RefuseFallbacks { .. } => StepKind::RefuseFallbacks,
            Self::Select { .. } => StepKind::Select,
            Self::PreRoute { .. } => StepKind::PreRoute,
            Self::Cache { .. } => StepKind::Cache,
            Self::Secrets { .. } => StepKind::Secrets,
            Self::Build { kind, .. } => *kind,
            Self::ModelInfo { .. } => StepKind::ModelInfo,
        }
    }
}

/// Python's response to a [`HostCall`].
#[derive(Debug)]
pub(crate) enum HostCallResponse {
    /// The attempt's total tokens, or its failure.
    Attempt(Result<u64, AttemptError>),
    /// A successful attempt whose response the provider blocked, or a stream whose outcome the
    /// host reports when it ends (`deferred`), with its tokens.
    Served {
        total_tokens: u64,
        blocked: Option<Blocked>,
        deferred: bool,
        tracks_usage: Option<bool>,
    },
    Slept,
    AllowFallback(bool),
    /// Whether the chain a `refuse_fallbacks` step named is refused.
    RefuseFallbacks(bool),
    /// The candidate ids that remain, pick first.
    Select(Vec<String>),
    /// A strategy's pick among candidates sharing an id: positions in the candidates, pick first.
    Positions(Vec<usize>),
    /// The model group to resolve instead, a custom strategy's deployment, or `None`.
    PreRoute(Option<PreRouted>),
    /// The facts a `pre_call` select step asked for.
    PreCall(PreCallFacts),
    /// The candidate models a `routing_plugins` select step reported.
    Models(Option<Vec<String>>),
    /// The ids a `callback_filters` select step left with the one a filter prefers, or the error
    /// a filter raised.
    Filtered(Result<(Vec<String>, Option<String>), AttemptError>),
    /// What a pick's select or pre-route step, or a fallback check, raised on the host instead of
    /// answering: the walk takes it as the try's failure.
    Raised(AttemptError),
    Cache(CacheOutcome),
    /// One value per ref of a `secrets` step.
    Secrets(Vec<Value>),
    Validated(Built),
    Provided(Provided),
    Added(Built),
    /// A `model_info` lookup's response, as the host spelled it.
    ModelInfo(Value),
}

impl HostCallResponse {
    fn kind(&self) -> StepKind {
        match self {
            Self::Attempt(_) | Self::Served { .. } => StepKind::Attempt,
            // Answers whichever step raised it (`check`).
            Self::Raised(_) => StepKind::Select,
            Self::Slept => StepKind::Sleep,
            Self::AllowFallback(_) => StepKind::AllowFallback,
            Self::RefuseFallbacks(_) => StepKind::RefuseFallbacks,
            Self::Select(_)
            | Self::Positions(_)
            | Self::PreCall(_)
            | Self::Models(_)
            | Self::Filtered(_) => StepKind::Select,
            Self::PreRoute(_) => StepKind::PreRoute,
            Self::Cache(_) => StepKind::Cache,
            Self::Secrets(_) => StepKind::Secrets,
            Self::Validated(_) => StepKind::Validate,
            Self::Provided(_) => StepKind::Provider,
            Self::Added(_) => StepKind::Added,
            Self::ModelInfo(_) => StepKind::ModelInfo,
        }
    }

    fn check(&self, call: &HostCall) -> Result<(), DriveError> {
        if matches!(self, Self::Raised(_))
            && matches!(
                call.kind(),
                StepKind::Select | StepKind::PreRoute | StepKind::AllowFallback
            )
        {
            return Ok(());
        }
        if self.kind() != call.kind() {
            return Err(DriveError::WrongAnswer {
                expected: call.kind(),
                got: self.kind(),
            });
        }
        if let (
            Self::Select(ids) | Self::Filtered(Ok((ids, _))),
            HostCall::Select { candidates, .. },
        ) = (self, call)
            && let Some(id) = {
                let known: std::collections::HashSet<&str> =
                    candidates.iter().map(String::as_str).collect();
                ids.iter().find(|id| !known.contains(id.as_str()))
            }
        {
            return Err(DriveError::UnknownCandidate { id: id.clone() });
        }
        if let (Self::Positions(positions), HostCall::Select { candidates, .. }) = (self, call)
            && let Some(position) = positions.iter().find(|p| **p >= candidates.len())
        {
            return Err(DriveError::UnknownCandidate {
                id: format!("#{position}"),
            });
        }
        if let (Self::Secrets(values), HostCall::Secrets { refs, .. }) = (self, call)
            && values.len() != refs.len()
        {
            return Err(DriveError::SecretsLength {
                expected: refs.len(),
                got: values.len(),
            });
        }
        if let (Self::Cache(outcome), HostCall::Cache { op, keys, .. }) = (self, call) {
            if !outcome.answers(*op) {
                return Err(DriveError::WrongCacheOutcome { op: *op });
            }
            if let (CacheOutcome::Values(values), Some(keys)) = (outcome, keys)
                && values.len() != keys.len()
            {
                return Err(DriveError::BatchLength {
                    expected: keys.len(),
                    got: values.len(),
                });
            }
        }
        Ok(())
    }
}

/// How a driven future ended well.
#[derive(Debug)]
pub(crate) enum Finish {
    /// `start`: the call's response.
    Call(RouterResponse),
    /// `select`: the deployment picked and the model group it serves.
    Selected {
        deployment_id: String,
        model_group: String,
        /// The deployment as served: a priority group's member carries its priority as `order`.
        deployment: Value,
    },
    /// `healthy`: the deployments a pick would choose among, as served.
    Healthy {
        /// A deployment named by its id, which the listing returns alone rather than as a list.
        single: bool,
        deployment_ids: Vec<String>,
        deployments: Vec<Value>,
    },
    /// `report_failure`: the cooldown applied, if the deployment was cooled down.
    Reported { cooldown: Option<Duration> },
    /// `report_success` / `count_success`: the success counter's key, if one was counted.
    Counted { key: Option<String> },
    /// `build`: the deployments it installed, in order.
    Built(Vec<BuiltDeployment>),
    /// `router_model_info`: the merged model info (`None` when the lookup answered `None`).
    RouterModelInfo(Option<MergedModelInfo>),
    /// `deployment_model_info`: the merged model info, if any.
    DeploymentModelInfo(Option<Map<String, Value>>),
    /// A model-info query that ends with one JSON answer (`model_group_info`,
    /// `configured_token_limits`, `model_listing_info`).
    Value(Value),
}

/// One step of a call: what it is blocked on, or how it ended.
#[derive(Debug)]
pub(crate) enum Step {
    HostCall(HostCall),
    Done(Finish),
    Failed(Error),
}

/// A misuse of the step protocol by the caller.
#[derive(Debug, thiserror::Error)]
pub(crate) enum DriveError {
    #[error("the router call already finished")]
    Finished,
    #[error("the router call has not made a host call yet, so the outcome must be None")]
    UnexpectedResponse,
    #[error("the router call is waiting for an outcome of its `{expected}` step")]
    MissingResponse { expected: StepKind },
    #[error("the router call is waiting for an outcome of its `{expected}` step, not `{got}`")]
    WrongAnswer { expected: StepKind, got: StepKind },
    #[error("deployment `{id}` was not one of the select step's candidates")]
    UnknownCandidate { id: String },
    #[error("the outcome does not answer a `{op}` cache step")]
    WrongCacheOutcome { op: CacheOp },
    #[error("a `batch_get` cache step for {expected} keys was answered with {got} values")]
    BatchLength { expected: usize, got: usize },
    #[error("a `secrets` step for {expected} references was answered with {got} values")]
    SecretsLength { expected: usize, got: usize },
    #[error("the router call is pending on something other than a step")]
    Stalled,
}

impl DriveError {
    /// Whether the caller passed a bad outcome, rather than driving a call that cannot go on.
    pub(crate) fn is_request(&self) -> bool {
        match self {
            Self::UnexpectedResponse
            | Self::MissingResponse { .. }
            | Self::WrongAnswer { .. }
            | Self::UnknownCandidate { .. }
            | Self::WrongCacheOutcome { .. }
            | Self::BatchLength { .. }
            | Self::SecretsLength { .. } => true,
            Self::Finished | Self::Stalled => false,
        }
    }
}

/// The host call a router call is blocked on and the response that releases it.
#[derive(Default)]
struct Exchange {
    call: Option<HostCall>,
    response: Option<HostCallResponse>,
    /// Events the call reported since the last step, handed to Python with the next one.
    events: Vec<RouterEvent>,
}

type Shared = Arc<Mutex<Exchange>>;

fn lock(exchange: &Mutex<Exchange>) -> MutexGuard<'_, Exchange> {
    exchange.lock().unwrap_or_else(PoisonError::into_inner)
}

thread_local! {
    static CURRENT: RefCell<Option<Shared>> = const { RefCell::new(None) };
}

/// Binds an exchange to this thread for one poll and restores the previous one after.
struct Bound(Option<Shared>);

impl Bound {
    fn new(exchange: &Shared) -> Self {
        Self(CURRENT.replace(Some(exchange.clone())))
    }
}

impl Drop for Bound {
    fn drop(&mut self) {
        CURRENT.set(self.0.take());
    }
}

/// Park the host call `call` in the bound exchange and wait for its response.
/// `response`, or the failure a step answered with instead ([`HostCallResponse::Raised`]).
fn answered(response: HostCallResponse) -> Result<HostCallResponse, AttemptError> {
    match response {
        HostCallResponse::Raised(failure) => Err(failure),
        response => Ok(response),
    }
}

async fn call_host(call: HostCall) -> HostCallResponse {
    let mut call = Some(call);
    poll_fn(move |_| {
        CURRENT.with_borrow(|current| {
            let Some(exchange) = current else {
                return Poll::Pending;
            };
            let mut exchange = lock(exchange);
            if let Some(call) = call.take() {
                exchange.call = Some(call);
                return Poll::Pending;
            }
            exchange.response.take().map_or(Poll::Pending, Poll::Ready)
        })
    })
    .await
}

const CHECKED: &str = "the driver checks each response against its host call";

fn ttl_seconds(ttl: Option<Duration>) -> Option<f64> {
    ttl.map(|ttl| ttl.as_secs_f64())
}

/// An increment amount as Python's counters store it: an int when it is whole.
fn amount_value(amount: f64) -> Value {
    if amount.fract() == 0.0 && amount.abs() < i64::MAX as f64 {
        Value::from(amount as i64)
    } else {
        Value::from(amount)
    }
}

/// The host's cache: every op is a `cache` step Python runs against the router's `DualCache`.
pub(crate) struct StepCache;

async fn cache_step(
    op: CacheOp,
    key: Option<&str>,
    keys: Option<&[String]>,
    value: Option<Value>,
    ttl: Option<Duration>,
) -> CacheOutcome {
    call_cache(op, key, keys, value, ttl, false).await
}

async fn call_cache(
    op: CacheOp,
    key: Option<&str>,
    keys: Option<&[String]>,
    value: Option<Value>,
    ttl: Option<Duration>,
    local: bool,
) -> CacheOutcome {
    let call = HostCall::Cache {
        op,
        key: key.map(str::to_owned),
        keys: keys.map(<[String]>::to_vec),
        value,
        ttl: ttl_seconds(ttl),
        local,
    };
    let HostCallResponse::Cache(outcome) = call_host(call).await else {
        unreachable!("{CHECKED}");
    };
    outcome
}

#[async_trait]
impl RouterCache for StepCache {
    async fn async_get_cache(&self, key: &str) -> Result<Option<Value>, litellm_cache::Error> {
        let CacheOutcome::Value(value) =
            cache_step(CacheOp::Get, Some(key), None, None, None).await
        else {
            unreachable!("{CHECKED}");
        };
        Ok(value)
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<(), litellm_cache::Error> {
        cache_step(CacheOp::Set, Some(key), None, Some(value), ttl).await;
        Ok(())
    }

    async fn async_increment(
        &self,
        key: &str,
        amount: f64,
        ttl: Option<Duration>,
    ) -> Result<f64, litellm_cache::Error> {
        let value = Some(amount_value(amount));
        let CacheOutcome::Number(next) =
            cache_step(CacheOp::Increment, Some(key), None, value, ttl).await
        else {
            unreachable!("{CHECKED}");
        };
        Ok(next)
    }

    async fn async_batch_get_cache(
        &self,
        keys: &[String],
    ) -> Result<Vec<Option<Value>>, litellm_cache::Error> {
        let CacheOutcome::Values(values) =
            cache_step(CacheOp::BatchGet, None, Some(keys), None, None).await
        else {
            unreachable!("{CHECKED}");
        };
        Ok(values)
    }

    async fn async_delete_cache(&self, key: &str) -> Result<(), litellm_cache::Error> {
        cache_step(CacheOp::Delete, Some(key), None, None, None).await;
        Ok(())
    }

    async fn async_increment_usage(
        &self,
        increments: &[(String, f64)],
        ttl: Option<Duration>,
    ) -> Result<(), litellm_cache::Error> {
        let keys: Vec<String> = increments.iter().map(|(key, _)| key.clone()).collect();
        let amounts = increments
            .iter()
            .map(|(_, amount)| amount_value(*amount))
            .collect();
        let value = Some(Value::Array(amounts));
        cache_step(CacheOp::IncrementUsage, None, Some(&keys), value, ttl).await;
        Ok(())
    }

    async fn async_get_local(&self, key: &str) -> Result<Option<Value>, litellm_cache::Error> {
        let CacheOutcome::Value(value) =
            call_cache(CacheOp::Get, Some(key), None, None, None, true).await
        else {
            unreachable!("{CHECKED}");
        };
        Ok(value)
    }

    async fn async_increment_local(
        &self,
        key: &str,
        amount: f64,
        ttl: Option<Duration>,
    ) -> Result<f64, litellm_cache::Error> {
        let value = Some(amount_value(amount));
        let CacheOutcome::Number(next) =
            call_cache(CacheOp::Increment, Some(key), None, value, ttl, true).await
        else {
            unreachable!("{CHECKED}");
        };
        Ok(next)
    }
}

/// A deployment as Python spells a `model_list` entry, with its id in `model_info`.
/// Whether two candidates share an id, which a pick answered by id can't tell apart.
fn shares_ids(candidates: &[Arc<Deployment>]) -> bool {
    let mut seen = std::collections::HashSet::new();
    !candidates.iter().all(|d| seen.insert(d.id.as_str()))
}

/// The candidates a host answer names, in its order. An id names every candidate that carries
/// it, so deployments sharing an id stay together through a filter, as Python's list keeps them.
fn answered_by_id(candidates: &[Arc<Deployment>], ids: &[String]) -> Vec<Arc<Deployment>> {
    let mut positions: std::collections::HashMap<&str, Vec<usize>> =
        std::collections::HashMap::new();
    for (position, deployment) in candidates.iter().enumerate() {
        positions
            .entry(deployment.id.as_str())
            .or_default()
            .push(position);
    }
    let mut kept = Vec::new();
    for id in ids {
        if let Some(found) = positions.remove(id.as_str()) {
            kept.extend(
                found
                    .into_iter()
                    .map(|position| candidates[position].clone()),
            );
        }
    }
    kept
}

pub(crate) fn deployment_record(deployment: &Deployment) -> Value {
    deployment.record()
}

const RANDOMNESS_MODULE: &str = "litellm.rust_bridge.host.randomness";

/// The process's `random` module: a Python router draws its shuffle and retry jitter there, so
/// a seeded process picks the same deployments on either core. The bare module functions share
/// one generator, so these calls interleave with every other draw the process makes.
pub(crate) struct PythonRandomness;

impl Randomness for PythonRandomness {
    fn next_f64(&self) -> f64 {
        Python::attach(|py| -> PyResult<f64> {
            py.import(RANDOMNESS_MODULE)?
                .call_method0("next_float")?
                .extract()
        })
        .unwrap_or_else(|_| ThreadRandomness.next_f64())
    }

    fn index_below(&self, len: usize) -> usize {
        // `random.choice` is `seq[self._randbelow(len(seq))]`, and so is `randrange(len)`.
        Python::attach(|py| -> PyResult<usize> {
            py.import(RANDOMNESS_MODULE)?
                .call_method1("index_below", (len,))?
                .extract()
        })
        .unwrap_or_else(|_| ThreadRandomness.index_below(len))
    }
}

/// The [`Host`] every `RouterCoreHandle` router runs on.
pub(crate) struct StepHost;

#[async_trait]
impl Host for StepHost {
    async fn invoke(&self, attempt: &Attempt) -> Result<Output, AttemptError> {
        let call = HostCall::Attempt {
            deployment_id: attempt.deployment.id.clone(),
            deployment: deployment_record(&attempt.deployment),
            model: attempt.name.clone(),
            model_group: attempt.model_group.clone(),
            routed: attempt.routed.clone(),
            target_order: attempt.target_order,
            candidate_ids: attempt.candidate_ids.clone(),
            retry: attempt.retry,
            fallback_depth: attempt.fallback_depth,
            fallback_level: attempt.fallback_level,
            timeout: attempt.timeout.map(|timeout| timeout.as_secs_f64()),
            params: attempt
                .params
                .as_ref()
                .map(|params| Box::new(Value::Object(Map::clone(params)))),
            scope: attempt.scope,
            max_retries: attempt.max_retries,
            model_group_size: attempt.model_group_size,
        };
        let (result, blocked, deferred, tracks_usage) = match call_host(call).await {
            HostCallResponse::Attempt(result) => (result, None, false, None),
            HostCallResponse::Served {
                total_tokens,
                blocked,
                deferred,
                tracks_usage,
            } => (Ok(total_tokens), blocked, deferred, tracks_usage),
            _ => unreachable!("{CHECKED}"),
        };
        result.map(|total_tokens| Output {
            value: Payload(Arc::new(())),
            total_tokens,
            blocked,
            deferred,
            tracks_usage,
        })
    }

    async fn select(
        &self,
        stage: &SelectStage,
        ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error> {
        let (scope, group) = ctx.scope.wire();
        let call = HostCall::Select {
            stage: match stage {
                SelectStage::Plugins => Stage::Plugins,
                SelectStage::Strategy(strategy) => Stage::Strategy {
                    strategy: strategy.to_string(),
                    records: if shares_ids(&candidates) {
                        candidates.iter().map(|d| deployment_record(d)).collect()
                    } else {
                        Vec::new()
                    },
                },
                SelectStage::Tags => Stage::Tags,
                SelectStage::TextOnly => Stage::TextOnly,
                SelectStage::RouteIncapable { handler } => Stage::RouteIncapable {
                    handler: handler.clone(),
                },
                SelectStage::MediaRefusing => Stage::MediaRefusing,
                SelectStage::EmbeddingBatch => Stage::EmbeddingBatch,
                SelectStage::OverrideCheck(strategy) => Stage::OverrideCheck(strategy.to_string()),
            },
            pool: ctx.pool.iter().map(|d| d.id.clone()).collect(),
            model_group: ctx.model_group.to_owned(),
            name: ctx.name.to_owned(),
            scope,
            group: group.map(str::to_owned),
            candidates: candidates.iter().map(|d| d.id.clone()).collect(),
        };
        match answered(call_host(call).await)? {
            HostCallResponse::Select(ids) => Ok(answered_by_id(&candidates, &ids)),
            HostCallResponse::Positions(positions) => Ok(positions
                .iter()
                .map(|position| candidates[*position].clone())
                .collect()),
            _ => unreachable!("{CHECKED}"),
        }
    }

    async fn pre_route(&self, pre: &PreRoute<'_>) -> Result<Option<PreRouted>, Error> {
        let call = HostCall::PreRoute {
            model: pre.name.to_owned(),
            fallback_depth: pre.fallback_depth,
            retry: pre.retry,
        };
        let HostCallResponse::PreRoute(routed) = answered(call_host(call).await)? else {
            unreachable!("{CHECKED}");
        };
        Ok(routed)
    }

    async fn routing_plugin_models(
        &self,
        ctx: &RoutingContext<'_>,
        run: bool,
    ) -> Result<Option<Vec<String>>, Error> {
        let (scope, group) = ctx.scope.wire();
        let call = HostCall::Select {
            stage: Stage::RoutingPlugins { run },
            pool: Vec::new(),
            model_group: ctx.model_group.to_owned(),
            name: ctx.name.to_owned(),
            scope,
            group: group.map(str::to_owned),
            candidates: Vec::new(),
        };
        let HostCallResponse::Models(models) = answered(call_host(call).await)? else {
            unreachable!("{CHECKED}");
        };
        Ok(models)
    }

    async fn provider_request(&self, ctx: &RoutingContext<'_>) -> Result<Option<String>, Error> {
        let (scope, group) = ctx.scope.wire();
        let call = HostCall::Select {
            stage: Stage::ProviderRequest,
            pool: Vec::new(),
            model_group: ctx.model_group.to_owned(),
            name: ctx.name.to_owned(),
            scope,
            group: group.map(str::to_owned),
            candidates: Vec::new(),
        };
        let HostCallResponse::Models(models) = answered(call_host(call).await)? else {
            unreachable!("{CHECKED}");
        };
        Ok(models.into_iter().flatten().next())
    }

    async fn pre_call_facts(
        &self,
        ctx: &RoutingContext<'_>,
        candidates: &[Arc<Deployment>],
    ) -> Result<PreCallFacts, Error> {
        let (scope, group) = ctx.scope.wire();
        let call = HostCall::Select {
            stage: Stage::PreCall,
            pool: ctx.pool.iter().map(|d| d.id.clone()).collect(),
            model_group: ctx.model_group.to_owned(),
            name: ctx.name.to_owned(),
            scope,
            group: group.map(str::to_owned),
            candidates: candidates.iter().map(|d| d.id.clone()).collect(),
        };
        let HostCallResponse::PreCall(facts) = answered(call_host(call).await)? else {
            unreachable!("{CHECKED}");
        };
        Ok(facts)
    }

    async fn callback_filters(
        &self,
        ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<CallbackFiltered, Error> {
        let (scope, group) = ctx.scope.wire();
        let joined = |own: &[String], walked: &[String]| [own, walked].concat();
        let call = HostCall::Select {
            stage: Stage::CallbackFilters {
                excluded_ids: joined(&ctx.request.excluded_ids, ctx.excluded_ids),
                skipped_ids: joined(&ctx.request.retry_skipped_ids, ctx.skipped_ids),
                target_order: ctx.target_order,
            },
            pool: Vec::new(),
            model_group: ctx.model_group.to_owned(),
            name: ctx.name.to_owned(),
            scope,
            group: group.map(str::to_owned),
            candidates: candidates.iter().map(|d| d.id.clone()).collect(),
        };
        let HostCallResponse::Filtered(result) = call_host(call).await else {
            unreachable!("{CHECKED}");
        };
        let (ids, preferred) = result.map_err(Error::Attempt)?;
        Ok(CallbackFiltered {
            kept: answered_by_id(&candidates, &ids),
            preferred,
        })
    }

    async fn allow_fallback(
        &self,
        _request: &RouterRequest,
        target: &str,
    ) -> Result<bool, AttemptError> {
        let call = HostCall::AllowFallback {
            target: target.to_owned(),
        };
        let HostCallResponse::AllowFallback(allowed) = answered(call_host(call).await)? else {
            unreachable!("{CHECKED}");
        };
        Ok(allowed)
    }

    async fn refuses_fallbacks(&self, level: u32, targets: &[Map<String, Value>]) -> bool {
        let call = HostCall::RefuseFallbacks {
            level,
            targets: DictTargets(targets.to_vec()),
        };
        let HostCallResponse::RefuseFallbacks(refused) = call_host(call).await else {
            unreachable!("{CHECKED}");
        };
        refused
    }

    /// A zero sleep is asked too: Python's retry loop awaits `asyncio.sleep(0)` between sibling
    /// retries, which yields to the event loop.
    async fn sleep(&self, duration: Duration) {
        let call = HostCall::Sleep {
            seconds: duration.as_secs_f64(),
        };
        let HostCallResponse::Slept = call_host(call).await else {
            unreachable!("{CHECKED}");
        };
    }

    /// Kept for Python only when it acts on it: an exhausted retry loop is stamped on the
    /// error Python raises, a fallback hop tells it which target a failed walk ended on, and a
    /// cooldown or a pick is reported to the host's Prometheus and service loggers (a pick also
    /// joins the router cache's per-deployment count, a failed walk the router's error log).
    fn observe(&self, event: &RouterEvent) {
        if !matches!(
            event,
            RouterEvent::RetriesExhausted { .. }
                | RouterEvent::Retry { .. }
                | RouterEvent::Fallback { .. }
                | RouterEvent::FallingBack { .. }
                | RouterEvent::OrderChainFailed { .. }
                | RouterEvent::FallbacksFailed { .. }
                | RouterEvent::LevelFailed { .. }
                | RouterEvent::PickFailed { .. }
                | RouterEvent::RetriesOpened { .. }
                | RouterEvent::RetryBudget { .. }
                | RouterEvent::MockFailed { .. }
                | RouterEvent::RetryLogged { .. }
                | RouterEvent::RetryPolicyRaised { .. }
                | RouterEvent::RetryCountRaised { .. }
                | RouterEvent::FallbackDone { .. }
                | RouterEvent::Blocked { .. }
                | RouterEvent::CooldownSet { .. }
                | RouterEvent::Selected { .. }
                | RouterEvent::PickStarted { .. }
        ) {
            return;
        }
        CURRENT.with_borrow(|current| {
            if let Some(exchange) = current {
                lock(exchange).events.push(event.clone());
            }
        });
    }

    fn cache(&self) -> Arc<dyn RouterCache> {
        Arc::new(StepCache)
    }

    async fn secrets(&self, index: usize, refs: &[SecretRef]) -> Result<Vec<Value>, Error> {
        let HostCallResponse::Secrets(values) = call_host(HostCall::Secrets {
            index,
            refs: refs.to_vec(),
        })
        .await
        else {
            unreachable!("{CHECKED}");
        };
        Ok(values)
    }

    async fn validate(&self, build: &BuildAsk<'_>) -> Result<Built, Error> {
        let HostCallResponse::Validated(built) =
            call_host(build_host_call(StepKind::Validate, build, None)).await
        else {
            unreachable!("{CHECKED}");
        };
        Ok(built)
    }

    async fn provider(&self, build: &BuildAsk<'_>) -> Result<Provided, Error> {
        let HostCallResponse::Provided(provided) =
            call_host(build_host_call(StepKind::Provider, build, None)).await
        else {
            unreachable!("{CHECKED}");
        };
        Ok(provided)
    }

    async fn added(&self, build: &BuildAsk<'_>, provider: Option<&str>) -> Result<Built, Error> {
        let HostCallResponse::Added(built) =
            call_host(build_host_call(StepKind::Added, build, provider)).await
        else {
            unreachable!("{CHECKED}");
        };
        Ok(built)
    }

    async fn model_info(&self, lookup: &ModelInfoLookup) -> Result<Value, Error> {
        let lookup = serde_json::to_value(lookup).unwrap_or(Value::Null);
        let HostCallResponse::ModelInfo(response) = call_host(HostCall::ModelInfo { lookup }).await
        else {
            unreachable!("{CHECKED}");
        };
        Ok(response)
    }
}

fn build_host_call(kind: StepKind, build: &BuildAsk<'_>, provider: Option<&str>) -> HostCall {
    HostCall::Build {
        kind,
        path: build.path,
        index: build.index,
        entry: Value::Object(build.entry.clone()),
        declared_id: build.declared_id.map(str::to_owned),
        duplicate_id: build.duplicate_id,
        provider: provider.map(str::to_owned),
    }
}

type CallFuture = Pin<Box<dyn Future<Output = Result<Finish, Error>> + Send>>;

/// Poll `future` once with `exchange` bound.
fn poll_bound<F: Future + ?Sized>(future: Pin<&mut F>, exchange: &Shared) -> Poll<F::Output> {
    let _bound = Bound::new(exchange);
    future.poll(&mut Context::from_waker(Waker::noop()))
}

/// Drives one router call a step at a time.
pub(crate) struct Driver {
    future: Option<CallFuture>,
    exchange: Shared,
    pending_call: Option<HostCall>,
}

impl Driver {
    pub(crate) fn new(call: impl Future<Output = Result<Finish, Error>> + Send + 'static) -> Self {
        Self {
            future: Some(Box::pin(call)),
            exchange: Shared::default(),
            pending_call: None,
        }
    }

    /// The host call the router call is blocked on, if it made one.
    pub(crate) fn pending_call(&self) -> Option<&HostCall> {
        self.pending_call.as_ref()
    }

    /// The events reported since the last call.
    pub(crate) fn take_events(&self) -> Vec<RouterEvent> {
        std::mem::take(&mut lock(&self.exchange).events)
    }

    /// Hand back the response to the pending host call (`None` on the first call) and run the call to
    /// its next step or its end.
    pub(crate) fn resume(
        &mut self,
        response: Option<HostCallResponse>,
    ) -> Result<Step, DriveError> {
        let Some(future) = self.future.as_mut() else {
            return Err(DriveError::Finished);
        };
        match (&self.pending_call, response) {
            (None, None) => {}
            (None, Some(_)) => return Err(DriveError::UnexpectedResponse),
            (Some(call), None) => {
                return Err(DriveError::MissingResponse {
                    expected: call.kind(),
                });
            }
            (Some(call), Some(response)) => {
                response.check(call)?;
                lock(&self.exchange).response = Some(response);
            }
        }
        self.pending_call = None;
        match poll_bound(future.as_mut(), &self.exchange) {
            Poll::Ready(result) => {
                self.future = None;
                Ok(match result {
                    Ok(finish) => Step::Done(finish),
                    Err(error) => Step::Failed(error),
                })
            }
            Poll::Pending => {
                let Some(call) = lock(&self.exchange).call.take() else {
                    self.future = None;
                    return Err(DriveError::Stalled);
                };
                self.pending_call = Some(call.clone());
                Ok(Step::HostCall(call))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        time::{SystemTime, UNIX_EPOCH},
    };

    use litellm_router_core::{
        FailureKind, Op, Router, RouterConfig, RouterDynamicConfig, SeededRandomness,
    };
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::*;

    fn router(config: Value, dynamic: Value) -> Router {
        let config: RouterConfig = serde_json::from_value(config).unwrap();
        let dynamic: RouterDynamicConfig = serde_json::from_value(dynamic).unwrap();
        Router::builder(config, dynamic, Arc::new(StepHost))
            .randomness(Arc::new(SeededRandomness::new(7)))
            .build()
            .unwrap()
    }

    fn deployment(group: &str, id: &str) -> Value {
        json!({"model_name": group, "litellm_params": {"model": format!("openai/{id}")}, "model_info": {"id": id}})
    }

    fn start(router: &Router, model: &str) -> Driver {
        let router = router.clone();
        let request = RouterRequest {
            model: model.to_owned(),
            ..RouterRequest::default()
        };
        Driver::new(async move {
            router
                .call(Op::from("acompletion"), request)
                .await
                .map(Finish::Call)
        })
    }

    fn rate_limited() -> HostCallResponse {
        HostCallResponse::Attempt(Err(AttemptError::new(FailureKind::RateLimit, Some(429))))
    }

    /// A dict-backed stand-in for Python's `DualCache` that answers `cache` steps and keeps
    /// every one it answered.
    #[derive(Default)]
    struct Store {
        entries: HashMap<String, Value>,
        calls: Vec<HostCall>,
    }

    impl Store {
        fn respond(&mut self, call: &HostCall) -> HostCallResponse {
            let HostCall::Cache {
                op,
                key,
                keys,
                value,
                ..
            } = call
            else {
                panic!("not a cache step: {call:?}");
            };
            self.calls.push(call.clone());
            let key = key.clone().unwrap_or_default();
            HostCallResponse::Cache(match op {
                CacheOp::Get => CacheOutcome::Value(self.entries.get(&key).cloned()),
                CacheOp::Set => {
                    self.entries.insert(key, value.clone().unwrap());
                    CacheOutcome::Stored
                }
                CacheOp::Increment => {
                    let current = self.entries.get(&key).and_then(Value::as_f64);
                    let next =
                        current.unwrap_or(0.0) + value.as_ref().and_then(Value::as_f64).unwrap();
                    self.entries.insert(key, amount_value(next));
                    CacheOutcome::Number(next)
                }
                CacheOp::IncrementUsage => {
                    let amounts = value.as_ref().and_then(Value::as_array).unwrap();
                    for (key, amount) in keys.iter().flatten().zip(amounts) {
                        let current = self.entries.get(key).and_then(Value::as_f64);
                        let next = current.unwrap_or(0.0) + amount.as_f64().unwrap();
                        self.entries.insert(key.clone(), amount_value(next));
                    }
                    CacheOutcome::Stored
                }
                CacheOp::BatchGet => CacheOutcome::Values(
                    keys.iter()
                        .flatten()
                        .map(|key| self.entries.get(key).cloned())
                        .collect(),
                ),
                CacheOp::Delete => {
                    self.entries.remove(&key);
                    CacheOutcome::Stored
                }
            })
        }

        /// Resume with `response`, answering every cache step on the way, up to the next other
        /// step.
        fn resume(&mut self, driver: &mut Driver, response: Option<HostCallResponse>) -> Step {
            let mut step = driver.resume(response).unwrap();
            while let Step::HostCall(call @ HostCall::Cache { .. }) = &step {
                let response = self.respond(call);
                step = driver.resume(Some(response)).unwrap();
            }
            step
        }
    }

    #[fixture]
    fn pair() -> Router {
        router(
            json!({"model_list": [deployment("primary", "a1"), deployment("primary", "a2")]}),
            json!({}),
        )
    }

    #[rstest]
    fn a_first_attempt_success_is_done(pair: Router) {
        let mut store = Store::default();
        let mut driver = start(&pair, "primary");
        let Step::HostCall(HostCall::Attempt {
            deployment_id,
            model_group,
            retry: 0,
            fallback_depth: 0,
            ..
        }) = store.resume(&mut driver, None)
        else {
            panic!("the call should start with an attempt");
        };
        assert_eq!(model_group, "primary");
        let Step::Done(Finish::Call(response)) =
            store.resume(&mut driver, Some(HostCallResponse::Attempt(Ok(12))))
        else {
            panic!("a successful attempt should finish the call");
        };
        assert_eq!(response.deployment_id, deployment_id);
        assert_eq!(response.output.total_tokens, 12);
        assert_eq!(
            (response.attempted_retries, response.attempted_fallbacks),
            (0, 0)
        );
        assert!(matches!(driver.resume(None), Err(DriveError::Finished)));
    }

    #[rstest]
    fn cooldown_reads_are_cache_steps(pair: Router) {
        let mut driver = start(&pair, "primary");
        let step = driver.resume(None).unwrap();
        let Step::HostCall(HostCall::Cache {
            op: CacheOp::BatchGet,
            key: None,
            keys: Some(keys),
            value: None,
            ttl: None,
            local: false,
        }) = step
        else {
            panic!("the call should read cooldowns first, got {step:?}");
        };
        assert_eq!(keys, ["deployment:a1:cooldown", "deployment:a2:cooldown"]);
    }

    #[rstest]
    #[case::lone_deployment_backs_off(vec![deployment("primary", "a1")], true)]
    #[case::sibling_retries_at_once(
        vec![deployment("primary", "a1"), deployment("primary", "a2")],
        false
    )]
    fn a_rate_limit_retries_after_the_backoff(
        #[case] model_list: Vec<Value>,
        #[case] backs_off: bool,
    ) {
        let router = router(json!({"model_list": model_list}), json!({"num_retries": 1}));
        let mut store = Store::default();
        let mut driver = start(&router, "primary");
        assert_eq!(
            store.resume(&mut driver, None).unwrap_kind(),
            StepKind::Attempt
        );
        let step = store.resume(&mut driver, Some(rate_limited()));
        let Step::HostCall(HostCall::Sleep { seconds }) = step else {
            panic!("a retry should sleep first, got {step:?}");
        };
        assert_eq!(seconds > 0.0, backs_off, "slept {seconds}s");
        let step = store.resume(&mut driver, Some(HostCallResponse::Slept));
        assert!(
            matches!(step, Step::HostCall(HostCall::Attempt { retry: 1, .. })),
            "expected the retry, got {step:?}"
        );
        let Step::Done(Finish::Call(response)) =
            store.resume(&mut driver, Some(HostCallResponse::Attempt(Ok(1))))
        else {
            panic!("the retry should finish the call");
        };
        assert_eq!(response.attempted_retries, 1);
    }

    #[rstest]
    fn a_cooldown_python_wrote_is_read_back_by_the_next_call() {
        let pair_without_retries = router(
            json!({"model_list": [deployment("primary", "a1"), deployment("primary", "a2")]}),
            json!({"num_retries": 0}),
        );
        let mut store = Store::default();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        store.entries.insert(
            "deployment:a1:cooldown".into(),
            json!({"exception_received": "slow down", "status_code": "429", "timestamp": now, "cooldown_time": 60}),
        );
        for _ in 0..8 {
            let mut call = start(&pair_without_retries, "primary");
            let Step::HostCall(HostCall::Attempt { deployment_id, .. }) =
                store.resume(&mut call, None)
            else {
                panic!("the call should attempt the healthy deployment");
            };
            assert_eq!(deployment_id, "a2", "the cooled-down deployment was picked");
        }
    }

    /// The host only runs the attempt; the core records its outcome (cooldowns, fail counters,
    /// usage) through the host's cache.
    #[rstest]
    #[case::failure(rate_limited())]
    #[case::success(HostCallResponse::Attempt(Ok(7)))]
    fn attempt_outcomes_are_recorded_by_the_core(#[case] outcome: HostCallResponse) {
        let tracked = |id: &str| {
            json!({
                "model_name": "primary",
                "litellm_params": {"model": format!("openai/{id}"), "tpm": 100, "cooldown_time": 7},
                "model_info": {"id": id, "allowed_fails": 3},
            })
        };
        let router = router(
            json!({"model_list": [tracked("a1"), tracked("a2")]}),
            json!({"num_retries": 0}),
        );
        let mut store = Store::default();
        let mut driver = start(&router, "primary");
        store.resume(&mut driver, None);
        let step = store.resume(&mut driver, Some(outcome));
        assert!(
            matches!(step, Step::Done(_) | Step::Failed(_)),
            "got {step:?}"
        );
        let writes: Vec<&HostCall> = store
            .calls
            .iter()
            .filter(|call| {
                matches!(call, HostCall::Cache { op, .. } if !matches!(op, CacheOp::Get | CacheOp::BatchGet))
            })
            .collect();
        assert!(!writes.is_empty(), "the core wrote no outcome");
    }

    #[rstest]
    fn a_failed_group_falls_back_once_python_allows_it() {
        let router = router(
            json!({"model_list": [deployment("primary", "a1"), deployment("backup", "b1")]}),
            json!({"fallbacks": [{"primary": ["backup"]}]}),
        );
        let mut store = Store::default();
        let mut driver = start(&router, "primary");
        assert_eq!(
            store.resume(&mut driver, None).unwrap_kind(),
            StepKind::Attempt
        );
        let bad_request =
            HostCallResponse::Attempt(Err(AttemptError::new(FailureKind::BadRequest, Some(400))));
        let step = store.resume(&mut driver, Some(bad_request));
        assert!(
            matches!(&step, Step::HostCall(HostCall::AllowFallback { target }) if target == "backup"),
            "expected the fallback check, got {step:?}"
        );
        let step = store.resume(&mut driver, Some(HostCallResponse::AllowFallback(true)));
        assert!(
            matches!(&step, Step::HostCall(HostCall::Attempt { deployment_id, fallback_depth: 1, .. }) if deployment_id == "b1"),
            "expected the fallback attempt, got {step:?}"
        );
        let Step::Done(Finish::Call(response)) =
            store.resume(&mut driver, Some(HostCallResponse::Attempt(Ok(3))))
        else {
            panic!("the fallback should finish the call");
        };
        assert_eq!(response.model_group, "backup");
        assert_eq!(response.attempted_fallbacks, 1);
    }

    #[rstest]
    fn a_refused_fallback_fails_with_the_attempt_error() {
        let router = router(
            json!({"model_list": [deployment("primary", "a1"), deployment("backup", "b1")]}),
            json!({"fallbacks": [{"primary": ["backup"]}]}),
        );
        let mut store = Store::default();
        let mut driver = start(&router, "primary");
        store.resume(&mut driver, None);
        let bad_request =
            HostCallResponse::Attempt(Err(AttemptError::new(FailureKind::BadRequest, Some(400))));
        store.resume(&mut driver, Some(bad_request));
        let Step::Failed(error) =
            store.resume(&mut driver, Some(HostCallResponse::AllowFallback(false)))
        else {
            panic!("a refused fallback should fail the call");
        };
        assert_eq!(
            error.attempt().map(|a| a.kind),
            Some(FailureKind::BadRequest)
        );
    }

    #[rstest]
    #[case::none_while_waiting(None, StepKind::Attempt)]
    #[case::wrong_kind(Some(HostCallResponse::Slept), StepKind::Attempt)]
    #[case::cache_outcome(Some(HostCallResponse::Cache(CacheOutcome::Stored)), StepKind::Attempt)]
    fn a_mismatched_outcome_is_rejected_and_the_step_kept(
        pair: Router,
        #[case] response: Option<HostCallResponse>,
        #[case] expected: StepKind,
    ) {
        let mut store = Store::default();
        let mut driver = start(&pair, "primary");
        store.resume(&mut driver, None);
        let error = driver.resume(response).unwrap_err();
        assert!(error.is_request(), "{error}");
        assert_eq!(driver.pending_call().map(HostCall::kind), Some(expected));
        assert!(matches!(
            store.resume(&mut driver, Some(HostCallResponse::Attempt(Ok(1)))),
            Step::Done(_)
        ));
    }

    #[rstest]
    #[case::wrong_op(HostCallResponse::Cache(CacheOutcome::Stored))]
    #[case::number_for_batch(HostCallResponse::Cache(CacheOutcome::Number(1.0)))]
    #[case::short_batch(HostCallResponse::Cache(CacheOutcome::Values(vec![None])))]
    #[case::not_a_cache_outcome(HostCallResponse::Slept)]
    fn a_mismatched_cache_outcome_is_rejected_and_the_step_kept(
        pair: Router,
        #[case] response: HostCallResponse,
    ) {
        let mut store = Store::default();
        let mut driver = start(&pair, "primary");
        let step = driver.resume(None).unwrap();
        let Step::HostCall(
            call @ HostCall::Cache {
                op: CacheOp::BatchGet,
                ..
            },
        ) = &step
        else {
            panic!("expected the cooldown read, got {step:?}");
        };
        let error = driver.resume(Some(response)).unwrap_err();
        assert!(error.is_request(), "{error}");
        assert_eq!(driver.pending_call(), Some(call));
        let response = store.respond(call);
        assert_eq!(
            store.resume(&mut driver, Some(response)).unwrap_kind(),
            StepKind::Attempt
        );
    }

    #[rstest]
    fn a_strategy_python_runs_asks_for_a_selection_and_keeps_the_pick() {
        let router = router(
            json!({
                "model_list": [deployment("primary", "a1"), deployment("primary", "a2")],
                "routing_strategy": "least-busy",
            }),
            json!({}),
        );
        let mut store = Store::default();
        let mut driver = start(&router, "primary");
        let step = store.resume(&mut driver, None);
        let Step::HostCall(HostCall::Select {
            stage: Stage::Strategy { strategy, records },
            candidates,
            ..
        }) = step
        else {
            panic!("a host strategy should ask for a selection, got {step:?}");
        };
        assert_eq!(strategy, "least-busy");
        assert_eq!(candidates, ["a1", "a2"]);
        assert!(records.is_empty(), "distinct ids send no records");
        let step = store.resume(
            &mut driver,
            Some(HostCallResponse::Select(vec!["a2".to_owned()])),
        );
        assert!(
            matches!(&step, Step::HostCall(HostCall::Attempt { deployment_id, .. }) if deployment_id == "a2"),
            "expected the picked deployment, got {step:?}"
        );
    }

    #[rstest]
    fn an_answer_by_id_keeps_its_order_and_every_candidate_sharing_an_id() {
        let candidates: Vec<Arc<Deployment>> = ["a", "b", "a", "c"]
            .into_iter()
            .map(|id| {
                let spec: litellm_router_core::DeploymentSpec =
                    serde_json::from_value(deployment("g", id)).unwrap();
                Arc::new(Deployment::from_spec(spec).unwrap())
            })
            .collect();
        let ids = ["c", "a", "x", "a"].map(String::from);

        let kept = answered_by_id(&candidates, &ids);

        let positions: Vec<usize> = kept
            .iter()
            .map(|d| candidates.iter().position(|c| Arc::ptr_eq(c, d)).unwrap())
            .collect();
        assert_eq!(positions, [3, 0, 2]);
    }
    #[rstest]
    fn a_pick_among_deployments_sharing_an_id_is_answered_by_position() {
        let mut second = deployment("primary", "a1");
        second["litellm_params"]["api_base"] = json!("https://second.example");
        let router = router(
            json!({
                "model_list": [deployment("primary", "a1"), second],
                "routing_strategy": "least-busy",
            }),
            json!({}),
        );
        let mut store = Store::default();
        let mut driver = start(&router, "primary");
        let step = store.resume(&mut driver, None);
        let Step::HostCall(HostCall::Select {
            stage: Stage::Strategy { records, .. },
            ..
        }) = step
        else {
            panic!("a host strategy should ask for a selection, got {step:?}");
        };
        assert_eq!(
            records.len(),
            2,
            "each candidate sharing an id sends its record"
        );
        let step = store.resume(&mut driver, Some(HostCallResponse::Positions(vec![1])));
        assert!(
            matches!(&step, Step::HostCall(HostCall::Attempt { deployment, .. })
                if deployment["litellm_params"]["api_base"] == "https://second.example"),
            "expected the second entry, got {step:?}"
        );
    }

    #[rstest]
    fn a_selection_is_driven_through_cache_steps(pair: Router) {
        let request = RouterRequest {
            model: "primary".to_owned(),
            ..RouterRequest::default()
        };
        let mut store = Store::default();
        let mut driver = Driver::new(async move {
            pair.select(&Op::from("acompletion"), &request)
                .await
                .map(|deployment| Finish::Selected {
                    deployment_id: deployment.id.clone(),
                    model_group: deployment.model_name.clone(),
                    deployment: deployment_record(&deployment),
                })
        });
        let Step::Done(Finish::Selected {
            deployment_id,
            model_group,
            ..
        }) = store.resume(&mut driver, None)
        else {
            panic!("a selection should finish with the pick");
        };
        assert!(["a1", "a2"].contains(&deployment_id.as_str()));
        assert_eq!(model_group, "primary");
        assert!(matches!(
            store.calls.as_slice(),
            [HostCall::Cache {
                op: CacheOp::BatchGet,
                ..
            }]
        ));
    }

    #[rstest]
    fn dict_targets_keep_their_api_key_out_of_debug() {
        let target = json!({"model": "b", "api_key": "credential-value"});
        let Value::Object(target) = target else {
            unreachable!("a JSON object");
        };
        let shown = format!("{:?}", DictTargets(vec![target]));
        assert!(!shown.contains("credential-value"));
        assert!(shown.contains("[REDACTED]"));
    }

    #[rstest]
    fn a_chain_with_a_dict_target_asks_python_before_any_hop() {
        let router = router(
            json!({"model_list": [deployment("primary", "a1"), deployment("backup", "b1")]}),
            json!({"fallbacks": [{"primary": [{"model": "backup", "api_key": "k"}]}]}),
        );
        let mut store = Store::default();
        let mut driver = start(&router, "primary");
        store.resume(&mut driver, None);
        let bad_request =
            HostCallResponse::Attempt(Err(AttemptError::new(FailureKind::BadRequest, Some(400))));
        let step = store.resume(&mut driver, Some(bad_request));
        assert!(
            matches!(&step, Step::HostCall(HostCall::RefuseFallbacks { level: 0, targets }) if targets.0.len() == 1),
            "expected the chain's check, got {step:?}"
        );
        let Step::Failed(error) =
            store.resume(&mut driver, Some(HostCallResponse::RefuseFallbacks(true)))
        else {
            panic!("a refused chain should fail the call");
        };
        assert_eq!(
            error.attempt().map(|a| a.kind),
            Some(FailureKind::BadRequest)
        );
    }

    trait StepKindOf {
        fn unwrap_kind(self) -> StepKind;
    }

    impl StepKindOf for Step {
        fn unwrap_kind(self) -> StepKind {
            match self {
                Step::HostCall(call) => call.kind(),
                Step::Done(_) => StepKind::Done,
                Step::Failed(_) => StepKind::Failed,
            }
        }
    }
}
