//! The host behind `RouterCoreHandle`. Rust never awaits Python: each host operation the call
//! loop needs becomes an [`Ask`] parked in the call's [`Exchange`], and the call future stays
//! pending until Python runs it and hands the [`Answer`] back through [`Driver::resume`].
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
    Attempt, AttemptError, Deployment, Error, Host, Output, Payload, RouterCache, RouterEvent,
    RouterRequest, RouterResponse, RoutingContext, SelectStage,
};
use serde_json::Value;

/// The `kind` of a step dict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display, strum::IntoStaticStr)]
pub(crate) enum StepKind {
    #[strum(serialize = "attempt")]
    Attempt,
    #[strum(serialize = "sleep")]
    Sleep,
    #[strum(serialize = "allow_fallback")]
    AllowFallback,
    #[strum(serialize = "select")]
    Select,
    #[strum(serialize = "cache")]
    Cache,
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
    Strategy(String),
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
    /// `set` and `delete`.
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
                | (Self::Stored, CacheOp::Set | CacheOp::Delete)
        )
    }
}

/// What a call is blocked on, for Python to run.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Ask {
    Attempt {
        deployment_id: String,
        /// The name the attempt resolved: the requested name or the fallback target.
        model: String,
        model_group: String,
        /// The `order` level of a same-group order fallback.
        target_order: Option<i64>,
        /// Every deployment `model` resolved to for this caller, the picked one included.
        candidate_ids: Vec<String>,
        retry: u32,
        fallback_depth: u32,
        fallback_level: u32,
        timeout: Option<f64>,
    },
    Sleep {
        seconds: f64,
    },
    AllowFallback {
        target: String,
    },
    Select {
        stage: Stage,
        model_group: String,
        candidates: Vec<String>,
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
    },
}

impl Ask {
    pub(crate) fn kind(&self) -> StepKind {
        match self {
            Self::Attempt { .. } => StepKind::Attempt,
            Self::Sleep { .. } => StepKind::Sleep,
            Self::AllowFallback { .. } => StepKind::AllowFallback,
            Self::Select { .. } => StepKind::Select,
            Self::Cache { .. } => StepKind::Cache,
        }
    }
}

/// Python's answer to an [`Ask`].
#[derive(Debug)]
pub(crate) enum Answer {
    /// The attempt's total tokens, or its failure.
    Attempt(Result<u64, AttemptError>),
    Slept,
    AllowFallback(bool),
    /// The candidate ids that remain, pick first.
    Select(Vec<String>),
    Cache(CacheOutcome),
}

impl Answer {
    fn kind(&self) -> StepKind {
        match self {
            Self::Attempt(_) => StepKind::Attempt,
            Self::Slept => StepKind::Sleep,
            Self::AllowFallback(_) => StepKind::AllowFallback,
            Self::Select(_) => StepKind::Select,
            Self::Cache(_) => StepKind::Cache,
        }
    }

    fn check(&self, ask: &Ask) -> Result<(), DriveError> {
        if self.kind() != ask.kind() {
            return Err(DriveError::WrongAnswer {
                expected: ask.kind(),
                got: self.kind(),
            });
        }
        if let (Self::Select(ids), Ask::Select { candidates, .. }) = (self, ask)
            && let Some(id) = ids.iter().find(|id| !candidates.contains(id))
        {
            return Err(DriveError::UnknownCandidate { id: id.clone() });
        }
        if let (Self::Cache(outcome), Ask::Cache { op, keys, .. }) = (self, ask) {
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
    },
}

/// One step of a call: what it is blocked on, or how it ended.
#[derive(Debug)]
pub(crate) enum Step {
    Ask(Ask),
    Done(Finish),
    Failed(Error),
}

/// A misuse of the step protocol by the caller.
#[derive(Debug, thiserror::Error)]
pub(crate) enum DriveError {
    #[error("the router call already finished")]
    Finished,
    #[error("the router call has not asked for anything yet, so the outcome must be None")]
    Unasked,
    #[error("the router call is waiting for an outcome of its `{expected}` step")]
    MissingAnswer { expected: StepKind },
    #[error("the router call is waiting for an outcome of its `{expected}` step, not `{got}`")]
    WrongAnswer { expected: StepKind, got: StepKind },
    #[error("deployment `{id}` was not one of the select step's candidates")]
    UnknownCandidate { id: String },
    #[error("the outcome does not answer a `{op}` cache step")]
    WrongCacheOutcome { op: CacheOp },
    #[error("a `batch_get` cache step for {expected} keys was answered with {got} values")]
    BatchLength { expected: usize, got: usize },
    #[error("the router call is pending on something other than a step")]
    Stalled,
}

impl DriveError {
    /// Whether the caller passed a bad outcome, rather than driving a call that cannot go on.
    pub(crate) fn is_request(&self) -> bool {
        match self {
            Self::Unasked
            | Self::MissingAnswer { .. }
            | Self::WrongAnswer { .. }
            | Self::UnknownCandidate { .. }
            | Self::WrongCacheOutcome { .. }
            | Self::BatchLength { .. } => true,
            Self::Finished | Self::Stalled => false,
        }
    }
}

/// The question a call is blocked on and the answer that releases it.
#[derive(Default)]
struct Exchange {
    asked: Option<Ask>,
    answer: Option<Answer>,
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

/// Park `question` in the bound exchange and wait for its answer.
async fn ask(question: Ask) -> Answer {
    let mut question = Some(question);
    poll_fn(move |_| {
        CURRENT.with_borrow(|current| {
            let Some(exchange) = current else {
                return Poll::Pending;
            };
            let mut exchange = lock(exchange);
            if let Some(question) = question.take() {
                exchange.asked = Some(question);
                return Poll::Pending;
            }
            exchange.answer.take().map_or(Poll::Pending, Poll::Ready)
        })
    })
    .await
}

const CHECKED: &str = "the driver checks each answer against its ask";

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
    let question = Ask::Cache {
        op,
        key: key.map(str::to_owned),
        keys: keys.map(<[String]>::to_vec),
        value,
        ttl: ttl_seconds(ttl),
    };
    let Answer::Cache(outcome) = ask(question).await else {
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
}

/// The [`Host`] every `RouterCoreHandle` router runs on.
pub(crate) struct StepHost;

#[async_trait]
impl Host for StepHost {
    async fn invoke(&self, attempt: &Attempt) -> Result<Output, AttemptError> {
        let question = Ask::Attempt {
            deployment_id: attempt.deployment.id.clone(),
            model: attempt.name.clone(),
            model_group: attempt.model_group.clone(),
            target_order: attempt.target_order,
            candidate_ids: attempt.candidate_ids.clone(),
            retry: attempt.retry,
            fallback_depth: attempt.fallback_depth,
            fallback_level: attempt.fallback_level,
            timeout: attempt.timeout.map(|timeout| timeout.as_secs_f64()),
        };
        let Answer::Attempt(result) = ask(question).await else {
            unreachable!("{CHECKED}");
        };
        result.map(|total_tokens| Output {
            value: Payload(Arc::new(())),
            total_tokens,
        })
    }

    async fn select(
        &self,
        stage: &SelectStage,
        ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error> {
        let question = Ask::Select {
            stage: match stage {
                SelectStage::Plugins => Stage::Plugins,
                SelectStage::Strategy(strategy) => Stage::Strategy(strategy.to_string()),
            },
            model_group: ctx.model_group.to_owned(),
            candidates: candidates.iter().map(|d| d.id.clone()).collect(),
        };
        let Answer::Select(ids) = ask(question).await else {
            unreachable!("{CHECKED}");
        };
        Ok(ids
            .iter()
            .filter_map(|id| candidates.iter().find(|d| d.id == *id).cloned())
            .collect())
    }

    async fn allow_fallback(&self, _request: &RouterRequest, target: &str) -> bool {
        let question = Ask::AllowFallback {
            target: target.to_owned(),
        };
        let Answer::AllowFallback(allowed) = ask(question).await else {
            unreachable!("{CHECKED}");
        };
        allowed
    }

    /// A zero sleep is asked too: Python's retry loop awaits `asyncio.sleep(0)` between sibling
    /// retries, which yields to the event loop.
    async fn sleep(&self, duration: Duration) {
        let question = Ask::Sleep {
            seconds: duration.as_secs_f64(),
        };
        let Answer::Slept = ask(question).await else {
            unreachable!("{CHECKED}");
        };
    }

    /// Kept for Python only when it acts on it: an exhausted retry loop is stamped on the
    /// error Python raises, and a fallback hop tells it which target a failed walk ended on.
    fn observe(&self, event: &RouterEvent) {
        if !matches!(
            event,
            RouterEvent::RetriesExhausted { .. } | RouterEvent::Fallback { .. }
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

    /// Each attempt runs through the Python router, whose callbacks record cooldowns,
    /// allowed-fails counters and usage.
    fn records_outcomes(&self) -> bool {
        true
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
    waiting: Option<Ask>,
}

impl Driver {
    pub(crate) fn new(call: impl Future<Output = Result<Finish, Error>> + Send + 'static) -> Self {
        Self {
            future: Some(Box::pin(call)),
            exchange: Shared::default(),
            waiting: None,
        }
    }

    /// The step the call is blocked on, if it asked for one.
    pub(crate) fn waiting(&self) -> Option<&Ask> {
        self.waiting.as_ref()
    }

    /// The events reported since the last call.
    pub(crate) fn take_events(&self) -> Vec<RouterEvent> {
        std::mem::take(&mut lock(&self.exchange).events)
    }

    /// Hand back the answer to the waiting step (`None` on the first call) and run the call to
    /// its next step or its end.
    pub(crate) fn resume(&mut self, answer: Option<Answer>) -> Result<Step, DriveError> {
        let Some(future) = self.future.as_mut() else {
            return Err(DriveError::Finished);
        };
        match (&self.waiting, answer) {
            (None, None) => {}
            (None, Some(_)) => return Err(DriveError::Unasked),
            (Some(ask), None) => {
                return Err(DriveError::MissingAnswer {
                    expected: ask.kind(),
                });
            }
            (Some(ask), Some(answer)) => {
                answer.check(ask)?;
                lock(&self.exchange).answer = Some(answer);
            }
        }
        self.waiting = None;
        match poll_bound(future.as_mut(), &self.exchange) {
            Poll::Ready(result) => {
                self.future = None;
                Ok(match result {
                    Ok(finish) => Step::Done(finish),
                    Err(error) => Step::Failed(error),
                })
            }
            Poll::Pending => {
                let Some(ask) = lock(&self.exchange).asked.take() else {
                    self.future = None;
                    return Err(DriveError::Stalled);
                };
                self.waiting = Some(ask.clone());
                Ok(Step::Ask(ask))
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
        Driver::new(async move { router.call(Op::Completion, request).await.map(Finish::Call) })
    }

    fn rate_limited() -> Answer {
        Answer::Attempt(Err(AttemptError::new(FailureKind::RateLimit, Some(429))))
    }

    /// A dict-backed stand-in for Python's `DualCache` that answers `cache` steps and keeps
    /// every one it answered.
    #[derive(Default)]
    struct Store {
        entries: HashMap<String, Value>,
        asked: Vec<Ask>,
    }

    impl Store {
        fn answer(&mut self, ask: &Ask) -> Answer {
            let Ask::Cache {
                op,
                key,
                keys,
                value,
                ..
            } = ask
            else {
                panic!("not a cache step: {ask:?}");
            };
            self.asked.push(ask.clone());
            let key = key.clone().unwrap_or_default();
            Answer::Cache(match op {
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

        /// Resume with `answer`, answering every cache step on the way, up to the next other
        /// step.
        fn resume(&mut self, driver: &mut Driver, answer: Option<Answer>) -> Step {
            let mut step = driver.resume(answer).unwrap();
            while let Step::Ask(ask @ Ask::Cache { .. }) = &step {
                let answer = self.answer(ask);
                step = driver.resume(Some(answer)).unwrap();
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
        let Step::Ask(Ask::Attempt {
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
            store.resume(&mut driver, Some(Answer::Attempt(Ok(12))))
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
        let Step::Ask(Ask::Cache {
            op: CacheOp::BatchGet,
            key: None,
            keys: Some(keys),
            value: None,
            ttl: None,
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
        let Step::Ask(Ask::Sleep { seconds }) = step else {
            panic!("a retry should sleep first, got {step:?}");
        };
        assert_eq!(seconds > 0.0, backs_off, "slept {seconds}s");
        let step = store.resume(&mut driver, Some(Answer::Slept));
        assert!(
            matches!(step, Step::Ask(Ask::Attempt { retry: 1, .. })),
            "expected the retry, got {step:?}"
        );
        let Step::Done(Finish::Call(response)) =
            store.resume(&mut driver, Some(Answer::Attempt(Ok(1))))
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
            let Step::Ask(Ask::Attempt { deployment_id, .. }) = store.resume(&mut call, None)
            else {
                panic!("the call should attempt the healthy deployment");
            };
            assert_eq!(deployment_id, "a2", "the cooled-down deployment was picked");
        }
    }

    /// Each attempt runs `PythonCore.call`, which records its own cooldowns, fail counters and
    /// usage, so the Rust call loop writes none of them.
    #[rstest]
    #[case::failure(rate_limited())]
    #[case::success(Answer::Attempt(Ok(7)))]
    fn attempt_outcomes_are_recorded_by_the_python_attempt(#[case] outcome: Answer) {
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
        let writes: Vec<&Ask> = store
            .asked
            .iter()
            .filter(|ask| {
                matches!(ask, Ask::Cache { op, .. } if !matches!(op, CacheOp::Get | CacheOp::BatchGet))
            })
            .collect();
        assert!(writes.is_empty(), "{writes:?}");
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
            Answer::Attempt(Err(AttemptError::new(FailureKind::BadRequest, Some(400))));
        let step = store.resume(&mut driver, Some(bad_request));
        assert!(
            matches!(&step, Step::Ask(Ask::AllowFallback { target }) if target == "backup"),
            "expected the fallback check, got {step:?}"
        );
        let step = store.resume(&mut driver, Some(Answer::AllowFallback(true)));
        assert!(
            matches!(&step, Step::Ask(Ask::Attempt { deployment_id, fallback_depth: 1, .. }) if deployment_id == "b1"),
            "expected the fallback attempt, got {step:?}"
        );
        let Step::Done(Finish::Call(response)) =
            store.resume(&mut driver, Some(Answer::Attempt(Ok(3))))
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
            Answer::Attempt(Err(AttemptError::new(FailureKind::BadRequest, Some(400))));
        store.resume(&mut driver, Some(bad_request));
        let Step::Failed(error) = store.resume(&mut driver, Some(Answer::AllowFallback(false)))
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
    #[case::wrong_kind(Some(Answer::Slept), StepKind::Attempt)]
    #[case::cache_outcome(Some(Answer::Cache(CacheOutcome::Stored)), StepKind::Attempt)]
    fn a_mismatched_outcome_is_rejected_and_the_step_kept(
        pair: Router,
        #[case] answer: Option<Answer>,
        #[case] expected: StepKind,
    ) {
        let mut store = Store::default();
        let mut driver = start(&pair, "primary");
        store.resume(&mut driver, None);
        let error = driver.resume(answer).unwrap_err();
        assert!(error.is_request(), "{error}");
        assert_eq!(driver.waiting().map(Ask::kind), Some(expected));
        assert!(matches!(
            store.resume(&mut driver, Some(Answer::Attempt(Ok(1)))),
            Step::Done(_)
        ));
    }

    #[rstest]
    #[case::wrong_op(Answer::Cache(CacheOutcome::Stored))]
    #[case::number_for_batch(Answer::Cache(CacheOutcome::Number(1.0)))]
    #[case::short_batch(Answer::Cache(CacheOutcome::Values(vec![None])))]
    #[case::not_a_cache_outcome(Answer::Slept)]
    fn a_mismatched_cache_outcome_is_rejected_and_the_step_kept(
        pair: Router,
        #[case] answer: Answer,
    ) {
        let mut store = Store::default();
        let mut driver = start(&pair, "primary");
        let step = driver.resume(None).unwrap();
        let Step::Ask(
            ask @ Ask::Cache {
                op: CacheOp::BatchGet,
                ..
            },
        ) = &step
        else {
            panic!("expected the cooldown read, got {step:?}");
        };
        let error = driver.resume(Some(answer)).unwrap_err();
        assert!(error.is_request(), "{error}");
        assert_eq!(driver.waiting(), Some(ask));
        let answer = store.answer(ask);
        assert_eq!(
            store.resume(&mut driver, Some(answer)).unwrap_kind(),
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
        let Step::Ask(Ask::Select {
            stage: Stage::Strategy(strategy),
            candidates,
            ..
        }) = step
        else {
            panic!("a host strategy should ask for a selection, got {step:?}");
        };
        assert_eq!(strategy, "least-busy");
        assert_eq!(candidates, ["a1", "a2"]);
        let step = store.resume(&mut driver, Some(Answer::Select(vec!["a2".to_owned()])));
        assert!(
            matches!(&step, Step::Ask(Ask::Attempt { deployment_id, .. }) if deployment_id == "a2"),
            "expected the picked deployment, got {step:?}"
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
            pair.select(&Op::Completion, &request)
                .await
                .map(|deployment| Finish::Selected {
                    deployment_id: deployment.id.clone(),
                    model_group: deployment.model_name.clone(),
                })
        });
        let Step::Done(Finish::Selected {
            deployment_id,
            model_group,
        }) = store.resume(&mut driver, None)
        else {
            panic!("a selection should finish with the pick");
        };
        assert!(["a1", "a2"].contains(&deployment_id.as_str()));
        assert_eq!(model_group, "primary");
        assert!(matches!(
            store.asked.as_slice(),
            [Ask::Cache {
                op: CacheOp::BatchGet,
                ..
            }]
        ));
    }

    trait StepKindOf {
        fn unwrap_kind(self) -> StepKind;
    }

    impl StepKindOf for Step {
        fn unwrap_kind(self) -> StepKind {
            match self {
                Step::Ask(ask) => ask.kind(),
                Step::Done(_) => StepKind::Done,
                Step::Failed(_) => StepKind::Failed,
            }
        }
    }
}
