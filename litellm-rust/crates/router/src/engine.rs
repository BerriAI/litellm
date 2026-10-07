//! The execute loop: `async_function_with_fallbacks` -> `async_function_with_retries` ->
//! one attempt per pick, with cooldowns and usage recorded as Python's callbacks record them.

use std::{
    collections::HashSet,
    future::Future,
    pin::Pin,
    sync::{Mutex, PoisonError},
};

use crate::{
    cooldown::{self, status_is_retryable},
    failure::{Classified, ExceptionClass, Failure, Raised},
    fallback::{chain_for_groups, generic_targets},
    host::{
        Attempt, HopStamp, Invoked, MockFailure, Op, RetryStamp, RouterHost, Target, TypedFallback,
    },
    operation::Operation,
    random::PythonRandom,
    retry::{Backoff, RetryContext, should_retry, sleep_before_retry},
    selection::{self, SelectionRequest},
    settings::{FallbackEntry, Fallbacks},
    snapshot::{Registry, Snapshot},
    store::Store,
};

/// A request-level override of a router setting: absent, or set (possibly to `None`).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Override<T> {
    #[default]
    Inherit,
    Set(Option<T>),
}

impl<T> Override<T> {
    fn or<'a>(&'a self, router: Option<&'a T>) -> Option<&'a T> {
        match self {
            Self::Inherit => router,
            Self::Set(value) => value.as_ref(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RouterCall {
    pub operation: Operation,
    pub model: String,
    pub num_retries: Option<u32>,
    pub disable_fallbacks: bool,
    pub fallbacks: Override<Fallbacks>,
    pub context_window_fallbacks: Override<Fallbacks>,
    pub content_policy_fallbacks: Override<Fallbacks>,
    /// The caller passed one `litellm_logging_obj` for every attempt, so only the first failure
    /// runs the failure callbacks.
    pub shared_logging: bool,
    pub web_search: bool,
    pub mock: Option<MockFailure>,
}

impl RouterCall {
    fn fallback_mock(&self) -> Option<MockFailure> {
        self.mock.filter(|mock| *mock != MockFailure::RateLimit)
    }
}

impl RouterCall {
    pub fn new(operation: Operation, model: impl Into<String>) -> Self {
        Self {
            operation,
            model: model.into(),
            num_retries: None,
            disable_fallbacks: false,
            fallbacks: Override::Inherit,
            context_window_fallbacks: Override::Inherit,
            content_policy_fallbacks: Override::Inherit,
            shared_logging: false,
            web_search: false,
            mock: None,
        }
    }
}

/// What the router reports about a successful call, for the host to write where Python's
/// readers look (response headers and `_hidden_params`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub model_group: String,
    pub deployment_id: String,
    pub attempted_retries: u32,
    pub max_retries: Option<u32>,
    pub attempted_fallbacks: u32,
}

pub struct Routed<R, E> {
    pub response: R,
    pub outcome: Outcome,
    pub ops: Vec<Op<E>>,
}

pub enum RouteError<E, F> {
    /// Raise `error`, after applying `ops`.
    Failed(Box<Failed<E>>),
    Host(F),
}

pub struct Failed<E> {
    pub error: Raised<E>,
    pub ops: Vec<Op<E>>,
}

pub struct Engine {
    registry: Registry,
    store: Store,
    random: Mutex<PythonRandom>,
}

impl Engine {
    pub fn new(snapshot: Snapshot, store: Store, random: PythonRandom) -> Self {
        Self {
            registry: Registry::new(snapshot),
            store,
            random: Mutex::new(random),
        }
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub async fn route<H: RouterHost>(
        &self,
        host: &H,
        call: RouterCall,
    ) -> Result<Routed<H::Response, H::Error>, RouteError<H::Error, H::Fault>> {
        let snapshot = self.registry.load();
        let mut run = Run {
            engine: self,
            snapshot: &snapshot,
            host,
            call: &call,
            callbacks_logged: false,
            attempted_targets: HashSet::new(),
            next_bucket: 1,
            next_rejection: 0,
            ops: Vec::new(),
        };
        let model = call.model.clone();
        let result = run.hop(model.clone(), 0, 0, model).await;
        let ops = std::mem::take(&mut run.ops);
        match result {
            Ok(success) => Ok(Routed {
                response: success.response,
                outcome: success.outcome,
                ops,
            }),
            Err(Stop::Failed(failure)) => Err(RouteError::Failed(Box::new(Failed {
                error: failure.raised,
                ops,
            }))),
            Err(Stop::Host(fault)) => Err(RouteError::Host(fault)),
        }
    }
}

struct Success<R> {
    response: R,
    outcome: Outcome,
}

enum Stop<E, F> {
    Failed(Box<Failure<E>>),
    Host(F),
}

type Attempted<R, E, F> = Result<Success<R>, Stop<E, F>>;

type Step<'a, R, E, F> = Pin<Box<dyn Future<Output = Result<Success<R>, Stop<E, F>>> + Send + 'a>>;

struct Run<'a, H: RouterHost> {
    engine: &'a Engine,
    snapshot: &'a Snapshot,
    host: &'a H,
    call: &'a RouterCall,
    callbacks_logged: bool,
    attempted_targets: HashSet<String>,
    next_bucket: u32,
    next_rejection: u64,
    ops: Vec<Op<H::Error>>,
}

struct Hop {
    group: String,
    depth: u32,
    bucket: u32,
    original_group: String,
}

impl<'a, H: RouterHost> Run<'a, H> {
    fn fallbacks(&self) -> Option<&'a Fallbacks> {
        self.call
            .fallbacks
            .or(self.snapshot.settings.fallbacks.as_ref())
    }

    fn context_window_fallbacks(&self) -> Option<&'a Fallbacks> {
        self.call.context_window_fallbacks.or(self
            .snapshot
            .settings
            .context_window_fallbacks
            .as_ref())
    }

    fn content_policy_fallbacks(&self) -> Option<&'a Fallbacks> {
        self.call.content_policy_fallbacks.or(self
            .snapshot
            .settings
            .content_policy_fallbacks
            .as_ref())
    }

    /// `async_function_with_fallbacks` for one hop.
    fn hop(
        &mut self,
        group: String,
        depth: u32,
        bucket: u32,
        original_group: String,
    ) -> Step<'_, H::Response, H::Error, H::Fault> {
        Box::pin(async move {
            let hop = Hop {
                group,
                depth,
                bucket,
                original_group,
            };
            let first = match self.call.fallback_mock().filter(|_| depth == 0) {
                Some(mock) => self.mock(&hop, mock).await,
                None => self.retries(&hop).await,
            };
            match first {
                Ok(success) => Ok(success),
                Err(Stop::Host(fault)) => Err(Stop::Host(fault)),
                Err(Stop::Failed(failure)) => self.fallback(&hop, failure).await,
            }
        })
    }

    /// `_handle_mock_testing_fallbacks`: the host raises the mock error, which goes straight to
    /// the fallback layer.
    async fn mock(
        &mut self,
        hop: &Hop,
        mock: MockFailure,
    ) -> Attempted<H::Response, H::Error, H::Fault> {
        let attempt = Attempt {
            target: Target::Mock(mock),
            model_group: hop.group.clone(),
            bucket: hop.bucket,
            fallback_depth: hop.depth,
            retry: RetryStamp {
                model_group_size: 0,
                attempted_retries: 0,
                max_retries: 0,
            },
            ops: std::mem::take(&mut self.ops),
        };
        match self.host.invoke(attempt).await.map_err(Stop::Host)? {
            Invoked::Failure { error, classified } => Err(Stop::Failed(Box::new(Failure {
                raised: Raised::Host(error),
                classified,
                deployment_id: None,
            }))),
            Invoked::Success(response) => Ok(Success {
                response,
                outcome: Outcome {
                    model_group: hop.group.clone(),
                    deployment_id: String::new(),
                    attempted_retries: 0,
                    max_retries: None,
                    attempted_fallbacks: 0,
                },
            }),
        }
    }

    /// `async_function_with_retries`.
    async fn retries(&mut self, hop: &Hop) -> Attempted<H::Response, H::Error, H::Fault> {
        let snapshot = self.snapshot;
        let settings = &snapshot.settings;
        let mut num_retries = self.call.num_retries.unwrap_or(settings.num_retries);
        let group_size = snapshot.group(&hop.group).len();
        let mut skipped = Vec::new();
        let mut stamp = RetryStamp {
            model_group_size: group_size,
            attempted_retries: 0,
            max_retries: num_retries,
        };
        let first = if hop.depth == 0 && self.call.mock == Some(MockFailure::RateLimit) {
            self.mock(hop, MockFailure::RateLimit).await
        } else {
            self.attempt(hop, stamp, &skipped).await
        };
        let failure = match first {
            Ok(success) => return Ok(success),
            Err(stop) => self.failed(stop).map_err(Stop::Host)?,
        };
        if failure.classified.guardrail_intervention {
            return Err(Stop::Failed(failure));
        }
        if self.call.num_retries.is_none()
            && let Some(deployment_retries) = failure.classified.num_retries
        {
            num_retries = deployment_retries;
        }
        let (mut healthy, all) = self.healthy_counts(&hop.group).await;
        let policy = (self.call.num_retries != Some(0))
            .then(|| settings.retry_policy_for(&hop.group))
            .flatten()
            .and_then(|policy| {
                policy.retries_for(&failure.classified.classes, failure.classified.status_code)
            });
        if let Some(retries) = policy {
            num_retries = retries;
        }
        if policy.is_none() && !should_retry(&failure.classified, &self.retry_context(healthy, all))
        {
            return Err(Stop::Failed(failure));
        }
        if num_retries == 0 {
            return Err(Stop::Failed(failure));
        }
        self.log_retry(hop, &failure);
        add_retry_skip(&mut skipped, &failure);
        self.sleep(&failure, num_retries, num_retries, healthy, all)
            .await
            .map_err(Stop::Host)?;
        let mut latest = failure;
        for attempt in 0..num_retries {
            stamp.attempted_retries = attempt + 1;
            stamp.max_retries = num_retries;
            let failure = match self.attempt(hop, stamp, &skipped).await {
                Ok(mut success) => {
                    success.outcome.attempted_retries = attempt + 1;
                    success.outcome.max_retries = Some(num_retries);
                    return Ok(success);
                }
                Err(stop) => self.failed(stop).map_err(Stop::Host)?,
            };
            self.log_retry(hop, &failure);
            let remaining = num_retries - attempt - 1;
            healthy = self.healthy_counts(&hop.group).await.0;
            if policy.is_none()
                && !should_retry(&failure.classified, &self.retry_context(healthy, all))
            {
                return Err(Stop::Failed(failure));
            }
            add_retry_skip(&mut skipped, &failure);
            self.sleep(&failure, remaining, num_retries, healthy, all)
                .await
                .map_err(Stop::Host)?;
            latest = failure;
        }
        if latest.classified.exact_litellm_type {
            self.ops.push(Op::StampRetries {
                error: latest.raised.clone(),
                max_retries: num_retries,
                num_retries,
            });
        }
        Err(Stop::Failed(latest))
    }

    fn failed(&self, stop: Stop<H::Error, H::Fault>) -> Result<Box<Failure<H::Error>>, H::Fault> {
        match stop {
            Stop::Failed(failure) => Ok(failure),
            Stop::Host(fault) => Err(fault),
        }
    }

    fn retry_context(&self, healthy: usize, all: usize) -> RetryContext<'a> {
        RetryContext {
            healthy,
            all,
            context_window_fallbacks: self.context_window_fallbacks(),
            content_policy_fallbacks: self.content_policy_fallbacks(),
            fallbacks: self.fallbacks(),
        }
    }

    fn log_retry(&mut self, hop: &Hop, failure: &Failure<H::Error>) {
        self.ops.push(Op::LogRetry {
            bucket: hop.bucket,
            model: hop.group.clone(),
            error: failure.raised.clone(),
        });
    }

    async fn sleep(
        &mut self,
        failure: &Failure<H::Error>,
        remaining: u32,
        num_retries: u32,
        healthy: usize,
        all: usize,
    ) -> Result<(), H::Fault> {
        let seconds = {
            let mut random = self
                .engine
                .random
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            sleep_before_retry(
                &failure.classified,
                Backoff {
                    remaining,
                    num_retries,
                    healthy,
                    all,
                },
                self.snapshot.settings.retry_after,
                &self.snapshot.settings.tunables,
                &mut random,
            )
        };
        self.host.sleep(seconds).await
    }

    /// `_async_get_healthy_deployments`: the group's deployments, and those not cooling down.
    /// A model naming one deployment by id reports nothing healthy, as Python does.
    async fn healthy_counts(&self, model: &str) -> (usize, usize) {
        let snapshot = self.snapshot;
        let group = match snapshot.group(model) {
            group if !group.is_empty() => group,
            _ if snapshot.by_id(model).is_some() => return (0, DEPLOYMENT_DICT_KEYS),
            _ => snapshot.by_litellm_model(model),
        };
        let ids: Vec<&str> = group
            .iter()
            .map(|deployment| deployment.id.as_str())
            .collect();
        let cooling = self.engine.store.active_cooldowns(&ids).await;
        (ids.len() - cooling.len(), ids.len())
    }

    /// One `_acompletion`: pick, invoke, and the callbacks Python runs around the call.
    async fn attempt(
        &mut self,
        hop: &Hop,
        retry: RetryStamp,
        skipped: &[String],
    ) -> Attempted<H::Response, H::Error, H::Fault> {
        let snapshot = self.snapshot;
        let request = SelectionRequest {
            web_search: self.call.web_search,
            retry_skipped: skipped.to_vec(),
        };
        let deployment = match selection::pick(
            snapshot,
            &self.engine.store,
            &self.engine.random,
            &hop.group,
            &request,
        )
        .await
        {
            Ok(deployment) => deployment,
            Err(rejection) => {
                self.next_rejection += 1;
                return Err(Stop::Failed(Box::new(Failure::rejected(
                    self.next_rejection,
                    rejection,
                ))));
            }
        };
        let attempt = Attempt {
            target: Target::Deployment(deployment.id.clone()),
            model_group: hop.group.clone(),
            bucket: hop.bucket,
            fallback_depth: hop.depth,
            retry,
            ops: std::mem::take(&mut self.ops),
        };
        match self.host.invoke(attempt).await.map_err(Stop::Host)? {
            Invoked::Success(response) => {
                cooldown::record_success(&self.engine.store, &deployment.id).await;
                Ok(Success {
                    response,
                    outcome: Outcome {
                        model_group: hop.group.clone(),
                        deployment_id: deployment.id.clone(),
                        attempted_retries: 0,
                        max_retries: None,
                        attempted_fallbacks: 0,
                    },
                })
            }
            Invoked::Failure { error, classified } => {
                if self.run_failure_callbacks(&classified) {
                    cooldown::record_failure(snapshot, &self.engine.store, deployment, &classified)
                        .await;
                }
                Err(Stop::Failed(Box::new(Failure::host(
                    error,
                    classified,
                    deployment.id.clone(),
                ))))
            }
        }
    }

    /// A failure runs litellm's failure callbacks when litellm raised it, and only once per
    /// request when the caller shares one logging object across attempts.
    fn run_failure_callbacks(&mut self, classified: &Classified) -> bool {
        if !classified.callbacks_ran {
            return false;
        }
        if !self.call.shared_logging {
            return true;
        }
        !std::mem::replace(&mut self.callbacks_logged, true)
    }

    /// `async_function_with_fallbacks_common_utils`.
    async fn fallback(
        &mut self,
        hop: &Hop,
        failure: Box<Failure<H::Error>>,
    ) -> Attempted<H::Response, H::Error, H::Fault> {
        if self.call.disable_fallbacks || failure.classified.guardrail_intervention {
            return Err(Stop::Failed(failure));
        }
        let top_level = hop.depth == 0;
        let lookup_groups = dedupe([
            hop.group.as_str(),
            hop.group.as_str(),
            hop.original_group.as_str(),
        ]);
        let fallbacks = self.fallbacks();
        let chain: Result<Option<Vec<String>>, Explained> = (|| {
            if let Some(list) = fallbacks.filter(|list| is_client_side_list(list)) {
                return Ok(Some(list.iter().filter_map(bare).collect()));
            }
            for (class, kind, typed) in [
                (
                    ExceptionClass::ContextWindowExceeded,
                    TypedFallback::ContextWindow,
                    self.context_window_fallbacks(),
                ),
                (
                    ExceptionClass::ContentPolicyViolation,
                    TypedFallback::ContentPolicy,
                    self.content_policy_fallbacks(),
                ),
            ] {
                if !failure.classified.is(class) {
                    continue;
                }
                match typed {
                    Some(list) => {
                        return chain_for_groups(list, &lookup_groups, &self.snapshot.providers)
                            .map(|chain| Some(chain.targets()))
                            .ok_or(Explained::NoChain);
                    }
                    None => self.ops.push(Op::MissingTypedFallbacks {
                        error: failure.raised.clone(),
                        kind,
                        model_group: hop.group.clone(),
                    }),
                }
                break;
            }
            let Some(list) = fallbacks.filter(|_| !lookup_groups.is_empty()) else {
                return Ok(None);
            };
            match chain_for_groups(list, &lookup_groups, &self.snapshot.providers)
                .map(|chain| chain.targets())
                .or_else(|| generic_targets(list).map(<[String]>::to_vec))
            {
                Some(targets) => Ok(Some(targets)),
                None => Err(Explained::NoGroup),
            }
        })();
        let outcome = match chain {
            Ok(None) => FallbackResult::NotAttempted,
            Err(Explained::NoChain) => FallbackResult::Failed {
                attempted: None,
                last: failure.raised.clone(),
            },
            Err(Explained::NoGroup) => {
                if top_level {
                    self.ops.push(Op::NoFallbackGroup {
                        error: failure.raised.clone(),
                        lookup_groups: lookup_groups
                            .iter()
                            .map(|group| (*group).to_owned())
                            .collect(),
                    });
                }
                return Err(Stop::Failed(failure));
            }
            Ok(Some(targets)) => match self.run_fallbacks(hop, &targets, &failure).await {
                Ok(success) => return Ok(success),
                Err(Stop::Host(fault)) => return Err(Stop::Host(fault)),
                Err(Stop::Failed(last)) => FallbackResult::Failed {
                    attempted: Some(targets),
                    last: last.raised,
                },
            },
        };
        if top_level {
            let (attempted, last) = match outcome {
                FallbackResult::NotAttempted => (None, None),
                FallbackResult::Failed { attempted, last } => (attempted, Some(last)),
            };
            self.ops.push(Op::FallbackOutcome {
                error: failure.raised.clone(),
                model_group: hop.group.clone(),
                attempted,
                last,
            });
        }
        Err(Stop::Failed(failure))
    }

    /// `run_async_fallback`.
    async fn run_fallbacks(
        &mut self,
        hop: &Hop,
        targets: &[String],
        original: &Failure<H::Error>,
    ) -> Attempted<H::Response, H::Error, H::Fault> {
        let max_fallbacks = self.snapshot.settings.max_fallbacks;
        if hop.depth >= max_fallbacks {
            return Err(Stop::Failed(Box::new(original.clone())));
        }
        self.attempted_targets.insert(hop.group.clone());
        let mut depth = hop.depth;
        let mut bucket = hop.bucket;
        let mut model = hop.group.clone();
        let mut last = original.clone();
        for target in targets {
            if *target == hop.group || !self.attempted_targets.insert(target.clone()) {
                continue;
            }
            self.ops.push(Op::LogRetry {
                bucket,
                model: model.clone(),
                error: original.raised.clone(),
            });
            depth += 1;
            let id = self.next_bucket;
            self.next_bucket += 1;
            self.ops.push(Op::OpenBucket {
                id,
                copy_of: bucket,
                stamp: HopStamp {
                    original_model_group: hop.original_group.clone(),
                    model_group: target.clone(),
                    attempted_fallbacks: depth,
                    max_fallbacks,
                },
            });
            bucket = id;
            model = target.clone();
            match self
                .hop(target.clone(), depth, id, hop.original_group.clone())
                .await
            {
                Ok(mut success) => {
                    success.outcome.attempted_fallbacks = depth;
                    return Ok(success);
                }
                Err(Stop::Host(fault)) => return Err(Stop::Host(fault)),
                Err(Stop::Failed(failure)) => {
                    if self.call.shared_logging && self.callbacks_logged {
                        self.cooldown_failed_hop(&failure).await;
                    }
                    last = *failure;
                }
            }
        }
        Err(Stop::Failed(Box::new(last)))
    }

    /// `_trigger_cooldown_for_failed_deployment`: with a shared logging object, a fallback hop's
    /// failure never reaches the failure callbacks, so the fallback path cools it down itself.
    async fn cooldown_failed_hop(&self, failure: &Failure<H::Error>) {
        let generic_not_found =
            self.call.operation.is_generic() && failure.classified.status_code == Some(404);
        if failure.classified.exempt_from_cooldown || generic_not_found {
            return;
        }
        let Some(deployment) = failure
            .deployment_id
            .as_deref()
            .and_then(|id| self.snapshot.by_id(id))
        else {
            return;
        };
        cooldown::record_hop_failure(
            self.snapshot,
            &self.engine.store,
            deployment,
            &failure.classified,
        )
        .await;
    }
}

enum Explained {
    NoChain,
    NoGroup,
}

enum FallbackResult<E> {
    NotAttempted,
    Failed {
        attempted: Option<Vec<String>>,
        last: Raised<E>,
    },
}

/// `len(deployment.to_json(exclude_none=True))` for a deployment dict: `model_name`,
/// `litellm_params` and `model_info`.
const DEPLOYMENT_DICT_KEYS: usize = 3;

fn add_retry_skip<E>(skipped: &mut Vec<String>, failure: &Failure<E>) {
    let Some(id) = &failure.deployment_id else {
        return;
    };
    let Some(status) = failure.classified.status_code else {
        return;
    };
    if !status_is_retryable(status) && !skipped.contains(id) {
        skipped.push(id.clone());
        skipped.sort();
    }
}

/// `_check_non_standard_fallback_format` for the shapes the first increment parses: a
/// non-empty list of bare group names.
fn is_client_side_list(fallbacks: &[FallbackEntry]) -> bool {
    !fallbacks.is_empty()
        && fallbacks
            .iter()
            .all(|entry| matches!(entry, FallbackEntry::Bare(_)))
}

fn bare(entry: &FallbackEntry) -> Option<String> {
    match entry {
        FallbackEntry::Bare(group) => Some(group.clone()),
        FallbackEntry::Keyed { .. } => None,
    }
}

fn dedupe(groups: [&str; 3]) -> Vec<&str> {
    groups
        .iter()
        .enumerate()
        .filter(|(index, group)| !group.is_empty() && !groups[..*index].contains(group))
        .map(|(_, group)| *group)
        .collect()
}
