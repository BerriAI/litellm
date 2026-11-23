use std::{collections::BTreeSet, sync::Arc, time::Duration};

use futures_util::future::join_all;
use litellm_redis_client::{CallTag, current_tag, tagged};
use litellm_redis_store::{
    Number, Pending, RequestRedis, Send as RedisSend, UpdateLock,
    rate_limits::{
        AcquireReply, AtomicCounter, AtomicReply, Counter, Entry, Gauge, GuardedIncrement,
        Increment, Kind, Local, RateLimits, Scope, WindowCount,
    },
};

use crate::{
    descriptor::{Declared, Descriptor, Limits, PROJECT_ITPM, PROJECT_OTPM},
    failure::{Failure, python_counter},
    status::{Backend, Code, LimitKind, Response, Status, WindowIdentity, identities},
};

pub const SLOT_TTL_SECONDS: i64 = 3600;

pub type Now = Arc<dyn Fn() -> f64 + Send + Sync>;

pub type Relabel = Arc<dyn Fn(&CallTag, &'static str) -> CallTag + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    pub window_size: i64,
    pub tpm_reservation_enabled: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Policy {
    pub fail_closed: bool,
    pub force_grouping: bool,
}

#[derive(Clone, Copy)]
pub struct Ctx<'a> {
    pub policy: Policy,
    pub request: Option<&'a RequestRedis>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the rate limit could not be verified against Redis")]
pub struct Unverifiable;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotAcquisition {
    pub slot_id: String,
    pub gauges: Vec<Scope>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GuardedOp {
    pub increment: Increment,
    pub expected_window_start: String,
    pub backend: Backend,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Plain(Increment),
    Guarded(GuardedOp),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Settlement {
    Plain(Vec<Increment>),
    Project(Vec<Op>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Refund {
    counter: Counter,
    window_start: String,
    increment: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct State {
    window: Option<String>,
    count: Option<i64>,
}

#[derive(Clone, Debug)]
struct WindowMeta {
    scope: Scope,
    requests_limit: Option<i64>,
    tokens_limit: Option<i64>,
    descriptor_key: String,
    descriptor_value: String,
}

#[derive(Clone, Debug)]
struct GaugeSpec {
    scope: Scope,
    limit: i64,
    descriptor_key: String,
}

#[derive(Clone, Debug)]
struct AtomicMeta {
    descriptor_key: String,
    descriptor_value: String,
    current_limit: i64,
    kind: Kind,
    counter: Counter,
    increment: i64,
    ttl: i64,
    window_size: i64,
}

type Group = (Vec<AtomicCounter>, Vec<AtomicMeta>);

type Registry = (Option<Vec<(String, f64)>>, i64);

pub struct Engine {
    store: Arc<RateLimits>,
    settings: Settings,
    now: Now,
    relabel: Option<Relabel>,
}

fn limit_kind(kind: Kind) -> LimitKind {
    match kind {
        Kind::Requests => LimitKind::Requests,
        Kind::Tokens => LimitKind::Tokens,
    }
}

fn seconds(ttl: i64) -> Option<Duration> {
    Some(Duration::from_secs(u64::try_from(ttl).unwrap_or(0)))
}

fn python_int(text: &str) -> Option<i64> {
    text.trim()
        .parse::<i64>()
        .ok()
        .or_else(|| text.trim().parse::<f64>().ok().map(|f| f.trunc() as i64))
}

fn python_str(number: Number) -> String {
    match number {
        Number::Int(value) => value.to_string(),
        Number::Float(value) if value.fract() == 0.0 && value.abs() < 1e16 => format!("{value:.1}"),
        Number::Float(value) => value.to_string(),
    }
}

fn scalar_int(local: &Local) -> Option<i64> {
    match local {
        Local::Number(number) => Some(number.truncate()),
        Local::Text(text) => python_int(text),
        Local::Slots(_) => None,
    }
}

fn scalar_text(local: &Local) -> Option<String> {
    match local {
        Local::Number(number) => Some(python_str(*number)),
        Local::Text(text) => Some(text.clone()),
        Local::Slots(_) => None,
    }
}

fn scope_of((descriptor, value): &(String, String)) -> Scope {
    Scope {
        descriptor: descriptor.clone(),
        value: value.clone(),
    }
}

fn counter_of(scope: &Scope, kind: Kind) -> Counter {
    Counter {
        scope: scope.clone(),
        kind,
    }
}

fn gauge_status(gauge: &GaugeSpec, in_flight: i64, code: Code) -> Status {
    Status {
        code,
        current_limit: gauge.limit,
        limit_remaining: (gauge.limit - in_flight).max(0),
        rate_limit_type: LimitKind::MaxParallelRequests,
        descriptor_key: gauge.descriptor_key.clone(),
        descriptor_value: None,
    }
}

fn atomic_status(meta: &AtomicMeta, code: Code, limit: i64, count: i64) -> Status {
    Status {
        code,
        current_limit: limit,
        limit_remaining: (limit - count).max(0),
        rate_limit_type: limit_kind(meta.kind),
        descriptor_key: meta.descriptor_key.clone(),
        descriptor_value: Some(meta.descriptor_value.clone()),
    }
}

impl Engine {
    pub fn new(store: Arc<RateLimits>, settings: Settings, now: Now) -> Self {
        Self {
            store,
            settings,
            now,
            relabel: None,
        }
    }

    pub fn with_relabel(mut self, relabel: Relabel) -> Self {
        self.relabel = Some(relabel);
        self
    }

    async fn labelled<F: Future>(&self, call_type: &'static str, operation: F) -> F::Output {
        match (self.relabel.as_ref(), current_tag()) {
            (Some(relabel), Some(tag)) => tagged(relabel(&tag, call_type), operation).await,
            _ => operation.await,
        }
    }

    pub async fn read(&self, entry: Entry, ctx: Ctx<'_>) -> Result<Option<Local>, Unverifiable> {
        let local = self.local(entry.clone());
        if local.is_some() || !self.store.has_redis() {
            return Ok(local);
        }
        self.labelled("async_get_cache", self.store.get(ctx.request, &entry, None))
            .await
            .map_err(|_| Unverifiable)
    }

    pub fn store(&self) -> &Arc<RateLimits> {
        &self.store
    }

    pub fn memory_updates(&self) -> &UpdateLock {
        self.store.memory_updates()
    }

    pub fn settings(&self) -> Settings {
        self.settings
    }

    pub fn now(&self) -> f64 {
        (self.now)()
    }

    pub fn has_redis(&self) -> bool {
        self.store.has_redis()
    }

    fn batch<'a>(&self, ctx: Ctx<'a>) -> Option<&'a RequestRedis> {
        if self.store.has_redis() {
            ctx.request
        } else {
            None
        }
    }

    fn local(&self, entry: Entry) -> Option<Local> {
        self.store.get_local(&entry).unwrap_or_else(|error| {
            tracing::warn!(%error, "rate limit memory read failed");
            None
        })
    }

    fn set_local(&self, entry: Entry, value: Local, ttl: Option<Duration>) {
        if let Err(error) = self.store.set_local(&entry, value, ttl) {
            tracing::warn!(%error, "rate limit memory write failed");
        }
    }

    fn slot_groups(&self, scopes: &[Scope], policy: Policy) -> Vec<Vec<usize>> {
        self.store.slot_groups(scopes, policy.force_grouping)
    }

    fn unverifiable(
        &self,
        policy: Policy,
        operation: &str,
        error: &dyn std::fmt::Display,
    ) -> Result<(), Unverifiable> {
        if !policy.fail_closed {
            return Ok(());
        }
        tracing::warn!(
            "fail_closed_rate_limit_enforcement: rejecting request, {operation} could not verify the counters against Redis: {error}"
        );
        Err(Unverifiable)
    }

    fn local_states(&self, counters: &[Counter]) -> Vec<State> {
        counters
            .iter()
            .map(|counter| State {
                window: self
                    .local(Entry::Window(counter.scope.clone()))
                    .as_ref()
                    .and_then(scalar_text),
                count: self
                    .local(Entry::Counter(counter.clone()))
                    .as_ref()
                    .and_then(scalar_int),
            })
            .collect()
    }

    fn collect(
        &self,
        descriptors: &[Descriptor],
        skip_tpm_check: bool,
    ) -> (Vec<Counter>, Vec<WindowMeta>, Vec<GaugeSpec>) {
        let mut counters = Vec::new();
        let mut metas: Vec<WindowMeta> = Vec::new();
        let mut gauges = Vec::new();
        for descriptor in descriptors {
            let scope = descriptor.scope();
            let limits = descriptor.limits();
            let requests_limit = limits.requests_per_unit;
            let tokens_limit = if skip_tpm_check {
                None
            } else {
                limits.tokens_per_unit
            };
            if let Some(limit) = limits.max_parallel_requests {
                gauges.push(GaugeSpec {
                    scope: scope.clone(),
                    limit,
                    descriptor_key: descriptor.key.clone(),
                });
            }
            if requests_limit.is_some() {
                counters.push(counter_of(&scope, Kind::Requests));
            }
            if tokens_limit.is_some() {
                counters.push(counter_of(&scope, Kind::Tokens));
            }
            if requests_limit.is_none() && tokens_limit.is_none() {
                continue;
            }
            let meta = WindowMeta {
                scope: scope.clone(),
                requests_limit,
                tokens_limit,
                descriptor_key: descriptor.key.clone(),
                descriptor_value: descriptor.value.clone(),
            };
            match metas.iter_mut().find(|existing| existing.scope == scope) {
                Some(existing) => *existing = meta,
                None => metas.push(meta),
            }
        }
        (counters, metas, gauges)
    }

    fn over_limit_list(
        counters: &[Counter],
        states: &[State],
        metas: &[WindowMeta],
        read_only: bool,
    ) -> Response {
        let pending = i64::from(read_only);
        let mut overall = Code::Ok;
        let mut statuses = Vec::new();
        for (counter, state) in counters.iter().zip(states) {
            let Some(meta) = metas.iter().find(|meta| meta.scope == counter.scope) else {
                continue;
            };
            let limit = match counter.kind {
                Kind::Requests => meta.requests_limit,
                Kind::Tokens => meta.tokens_limit,
            };
            let Some(limit) = limit else { continue };
            let mut code = Code::Ok;
            if let Some(count) = state.count
                && count + pending > limit
            {
                overall = Code::OverLimit;
                code = Code::OverLimit;
            }
            statuses.push(Status {
                code,
                current_limit: limit,
                limit_remaining: state.count.map_or(limit, |count| limit - count),
                rate_limit_type: limit_kind(counter.kind),
                descriptor_key: meta.descriptor_key.clone(),
                descriptor_value: Some(meta.descriptor_value.clone()),
            });
        }
        Response {
            overall,
            statuses,
            windows: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn should_rate_limit(
        &self,
        descriptors: &[Descriptor],
        read_only: bool,
        skip_tpm_check: bool,
        parallel_slot_id: Option<&str>,
        skip_parallel_check: bool,
        ctx: Ctx<'_>,
    ) -> Result<Response, Unverifiable> {
        let now = self.now();
        let now_int = now.trunc() as i64;
        let (counters, metas, gauges) = self.collect(descriptors, skip_tpm_check);
        let mut windowed = Response::ok(Vec::new());
        if !counters.is_empty() {
            let local =
                Self::over_limit_list(&counters, &self.local_states(&counters), &metas, read_only);
            if local.is_over() {
                return Ok(local);
            }
            let states = if read_only {
                self.read_without_incrementing(&counters, ctx).await?
            } else if self.store.has_redis() {
                let states = self.sliding_window_redis(&counters, now_int, ctx).await?;
                let window = seconds(self.settings.window_size);
                for (counter, state) in counters.iter().zip(&states) {
                    if let Some(count) = state.count {
                        self.set_local(
                            Entry::Counter(counter.clone()),
                            Local::Number(Number::Int(count)),
                            window,
                        );
                    }
                    if let Some(start) = &state.window {
                        self.set_local(
                            Entry::Window(counter.scope.clone()),
                            Local::Text(start.clone()),
                            window,
                        );
                    }
                }
                states
            } else {
                let _guard = self.store.memory_guard();
                self.sliding_window_memory(&counters, now_int, self.settings.window_size)
            };
            windowed = Self::over_limit_list(&counters, &states, &metas, read_only);
            if windowed.is_over() {
                return Ok(windowed);
            }
        }
        if gauges.is_empty() || skip_parallel_check {
            return Ok(windowed);
        }
        let slot_id = parallel_slot_id
            .map_or_else(|| uuid::Uuid::new_v4().simple().to_string(), str::to_owned);
        let gauge = self.check_gauges(&gauges, &slot_id, read_only, ctx).await?;
        let mut statuses = windowed.statuses;
        statuses.extend(gauge.statuses);
        Ok(Response {
            overall: gauge.overall,
            statuses,
            windows: None,
        })
    }

    fn sliding_window_memory(
        &self,
        counters: &[Counter],
        now_int: i64,
        window_size: i64,
    ) -> Vec<State> {
        let ttl = seconds(window_size);
        counters
            .iter()
            .map(|counter| {
                let start = self
                    .local(Entry::Window(counter.scope.clone()))
                    .as_ref()
                    .and_then(|local| scalar_text(local).map(|text| (text, scalar_int(local))));
                match start {
                    Some((text, Some(start))) if now_int - start < window_size => {
                        let current = self
                            .local(Entry::Counter(counter.clone()))
                            .as_ref()
                            .and_then(scalar_int)
                            .unwrap_or(0);
                        self.set_local(
                            Entry::Counter(counter.clone()),
                            Local::Number(Number::Int(current + 1)),
                            ttl,
                        );
                        State {
                            window: Some(text),
                            count: Some(current + 1),
                        }
                    }
                    _ => {
                        for kind in [Kind::Requests, Kind::Tokens] {
                            self.set_local(
                                Entry::Counter(counter_of(&counter.scope, kind)),
                                Local::Number(Number::Int(0)),
                                ttl,
                            );
                        }
                        self.set_local(
                            Entry::Window(counter.scope.clone()),
                            Local::Text(now_int.to_string()),
                            ttl,
                        );
                        self.set_local(
                            Entry::Counter(counter.clone()),
                            Local::Number(Number::Int(1)),
                            ttl,
                        );
                        State {
                            window: Some(now_int.to_string()),
                            count: Some(1),
                        }
                    }
                }
            })
            .collect()
    }

    async fn read_without_incrementing(
        &self,
        counters: &[Counter],
        ctx: Ctx<'_>,
    ) -> Result<Vec<State>, Unverifiable> {
        if !self.store.has_redis() {
            return Ok(self.local_states(counters));
        }
        let scopes: Vec<Scope> = counters
            .iter()
            .map(|counter| counter.scope.clone())
            .collect();
        let groups = self.slot_groups(&scopes, ctx.policy);
        let mut by_index: Vec<State> = vec![State::default(); counters.len()];
        for group in &groups {
            let members: Vec<Counter> =
                group.iter().map(|&index| counters[index].clone()).collect();
            let reply = match self.store.read_counters(RedisSend::Now, &members) {
                Some(pending) => self
                    .labelled("batch_counter_read_script", pending)
                    .await
                    .map_err(Failure::from),
                None => Err(Failure::no_redis()),
            };
            match reply {
                Ok(values) => {
                    for (offset, &index) in group.iter().enumerate() {
                        by_index[index] = State {
                            window: values.get(offset * 2).cloned().flatten(),
                            count: values
                                .get(offset * 2 + 1)
                                .cloned()
                                .flatten()
                                .as_deref()
                                .and_then(python_int),
                        };
                    }
                }
                Err(error) => {
                    self.unverifiable(ctx.policy, "batch_counter_read_script", &error)?;
                    tracing::warn!("batch_counter_read_script failed, using local mirror: {error}");
                    return Ok(self.local_states(counters));
                }
            }
        }
        Ok(by_index)
    }

    async fn sliding_window_redis(
        &self,
        counters: &[Counter],
        now_int: i64,
        ctx: Ctx<'_>,
    ) -> Result<Vec<State>, Unverifiable> {
        let scopes: Vec<Scope> = counters
            .iter()
            .map(|counter| counter.scope.clone())
            .collect();
        let slot_groups = self.slot_groups(&scopes, ctx.policy);
        let groups: Vec<Vec<Counter>> = slot_groups
            .iter()
            .map(|group| group.iter().map(|&index| counters[index].clone()).collect())
            .collect();
        let window_size = self.settings.window_size;
        let pipelined: Vec<Option<Pending<Vec<WindowCount>>>> = match self.batch(ctx) {
            Some(request) => groups
                .iter()
                .map(|group| {
                    self.store.sliding_window(
                        RedisSend::PreCall(request),
                        group,
                        now_int,
                        window_size,
                    )
                })
                .collect(),
            None => groups.iter().map(|_| None).collect(),
        };
        let mut pending = pipelined.into_iter();
        let mut states_by_group: Vec<Vec<State>> = Vec::new();
        for (index, group) in groups.iter().enumerate() {
            let declared = pending.next().flatten();
            let reply = match declared {
                Some(declared) => self
                    .labelled("batch_rate_limiter_script", declared)
                    .await
                    .map_err(Failure::from),
                None => match self
                    .store
                    .sliding_window(RedisSend::Now, group, now_int, window_size)
                {
                    Some(now) => self
                        .labelled("batch_rate_limiter_script", now)
                        .await
                        .map_err(Failure::from),
                    None => Err(Failure::no_redis()),
                },
            };
            match reply {
                Ok(counts) => states_by_group.push(
                    counts
                        .into_iter()
                        .map(|count| State {
                            window: Some(count.window_start),
                            count: Some(count.count),
                        })
                        .collect(),
                ),
                Err(error) => {
                    if ctx.policy.fail_closed {
                        let applied: Vec<Counter> =
                            groups[..index].iter().flatten().cloned().collect();
                        let values: Vec<State> =
                            states_by_group.iter().flatten().cloned().collect();
                        self.refund(Self::batch_refunds(&applied, &values)).await;
                        for (later, declared) in groups[index + 1..].iter().zip(pending.by_ref()) {
                            let Some(declared) = declared else { continue };
                            let Ok(counts) =
                                self.labelled("batch_rate_limiter_script", declared).await
                            else {
                                continue;
                            };
                            let values: Vec<State> = counts
                                .into_iter()
                                .map(|count| State {
                                    window: Some(count.window_start),
                                    count: Some(count.count),
                                })
                                .collect();
                            self.refund(Self::batch_refunds(later, &values)).await;
                        }
                    }
                    self.unverifiable(ctx.policy, "batch_rate_limiter_script", &error)?;
                    tracing::warn!("Redis Lua script failed for slot group {index}: {error}");
                    let _guard = self.store.memory_guard();
                    states_by_group.push(self.sliding_window_memory(group, now_int, window_size));
                }
            }
        }
        let mut by_index: Vec<State> = vec![State::default(); counters.len()];
        for (index, state) in slot_groups
            .iter()
            .flatten()
            .zip(states_by_group.into_iter().flatten())
        {
            by_index[*index] = state;
        }
        Ok(by_index)
    }

    fn batch_refunds(counters: &[Counter], states: &[State]) -> Vec<Refund> {
        counters
            .iter()
            .zip(states)
            .filter_map(|(counter, state)| {
                let start = state.window.as_deref().and_then(python_int)?;
                Some(Refund {
                    counter: counter.clone(),
                    window_start: start.to_string(),
                    increment: 1,
                })
            })
            .collect()
    }

    async fn refund(&self, refunds: Vec<Refund>) {
        if !self.store.has_redis() {
            return;
        }
        for refund in refunds {
            let increment = GuardedIncrement {
                counter: refund.counter.clone(),
                expected_window_start: refund.window_start,
                amount: Number::Int(-refund.increment),
                ttl: Some(Duration::ZERO),
            };
            let outcome = match self.store.guarded_increment(RedisSend::Now, &increment) {
                Some(pending) => self
                    .labelled("window_guarded_token_increment_script", pending)
                    .await
                    .map(|_| ())
                    .map_err(Failure::from),
                None => Err(Failure::no_redis()),
            };
            if let Err(error) = outcome {
                tracing::warn!(
                    "Failed to refund {} on rollback: {error}",
                    python_counter(&refund.counter)
                );
            }
        }
    }

    fn in_flight(&self, local: Option<Local>) -> i64 {
        match local {
            None => 0,
            Some(Local::Slots(slots)) => {
                let cutoff = self.now() - SLOT_TTL_SECONDS as f64;
                slots.iter().filter(|(_, at)| *at >= cutoff).count() as i64
            }
            Some(other) => scalar_int(&other).unwrap_or(0).max(0),
        }
    }

    fn local_gauge_counts(&self, scopes: &[Scope]) -> Vec<i64> {
        scopes
            .iter()
            .map(|scope| self.in_flight(self.local(Entry::Gauge(scope.clone()))))
            .collect()
    }

    async fn check_gauges(
        &self,
        gauges: &[GaugeSpec],
        slot_id: &str,
        read_only: bool,
        ctx: Ctx<'_>,
    ) -> Result<Response, Unverifiable> {
        let scopes: Vec<Scope> = gauges.iter().map(|gauge| gauge.scope.clone()).collect();
        let slot_ttl = Duration::from_secs(SLOT_TTL_SECONDS as u64);
        if read_only {
            let counts = if self.store.has_redis() {
                let reply = match self.store.parallel_count(RedisSend::Now, &scopes, slot_ttl) {
                    Some(pending) => self
                        .labelled("parallel_count_script", pending)
                        .await
                        .map_err(Failure::from),
                    None => Err(Failure::no_redis()),
                };
                match reply {
                    Ok(values) => values.into_iter().map(|value| value.max(0)).collect(),
                    Err(error) => {
                        self.unverifiable(ctx.policy, "parallel_count_script", &error)?;
                        tracing::warn!("parallel_count_script failed, using local mirror: {error}");
                        self.local_gauge_counts(&scopes)
                    }
                }
            } else {
                self.local_gauge_counts(&scopes)
            };
            let mut overall = Code::Ok;
            let statuses = gauges
                .iter()
                .zip(counts)
                .map(|(gauge, in_flight)| {
                    let code = if in_flight >= gauge.limit {
                        Code::OverLimit
                    } else {
                        Code::Ok
                    };
                    if code == Code::OverLimit {
                        overall = Code::OverLimit;
                    }
                    gauge_status(gauge, in_flight, code)
                })
                .collect();
            return Ok(Response {
                overall,
                statuses,
                windows: None,
            });
        }
        for (gauge, in_flight) in gauges.iter().zip(self.local_gauge_counts(&scopes)) {
            if in_flight >= gauge.limit {
                return Ok(Response::over(gauge_status(
                    gauge,
                    in_flight,
                    Code::OverLimit,
                )));
            }
        }
        if self.store.has_redis() {
            let specs: Vec<Gauge> = gauges
                .iter()
                .map(|gauge| Gauge {
                    scope: gauge.scope.clone(),
                    limit: gauge.limit,
                })
                .collect();
            let reply = match self
                .store
                .parallel_acquire(RedisSend::Now, &specs, slot_ttl, slot_id)
            {
                Some(pending) => self
                    .labelled("parallel_acquire_script", pending)
                    .await
                    .map_err(Failure::from),
                None => Err(Failure::no_redis()),
            };
            return match reply {
                Err(error) => {
                    self.unverifiable(ctx.policy, "parallel_acquire_script", &error)?;
                    tracing::warn!(
                        "parallel_acquire_script failed, falling back to in-memory gauge: {error}"
                    );
                    let now = self.now();
                    let _guard = self.store.memory_guard();
                    Ok(self.acquire_in_memory(gauges, slot_id, now))
                }
                Ok(AcquireReply::OverLimit {
                    index, in_flight, ..
                }) => Ok(Response::over(gauge_status(
                    &gauges[index],
                    in_flight,
                    Code::OverLimit,
                ))),
                Ok(AcquireReply::Acquired(counts)) => {
                    let statuses = gauges
                        .iter()
                        .zip(counts)
                        .map(|(gauge, in_flight)| {
                            self.set_local(
                                Entry::Gauge(gauge.scope.clone()),
                                Local::Number(Number::Int(in_flight)),
                                Some(slot_ttl),
                            );
                            gauge_status(gauge, in_flight, Code::Ok)
                        })
                        .collect();
                    Ok(Response::ok(statuses))
                }
            };
        }
        let now = self.now();
        let _guard = self.store.memory_guard();
        Ok(self.acquire_in_memory(gauges, slot_id, now))
    }

    fn acquire_in_memory(&self, gauges: &[GaugeSpec], slot_id: &str, now: f64) -> Response {
        let cutoff = now - SLOT_TTL_SECONDS as f64;
        let mut states: Vec<Registry> = Vec::new();
        for gauge in gauges {
            let state = match self.local(Entry::Gauge(gauge.scope.clone())) {
                Some(Local::Slots(slots)) => {
                    let live: Vec<(String, f64)> =
                        slots.into_iter().filter(|(_, at)| *at >= cutoff).collect();
                    let count = live.len() as i64;
                    (Some(live), count)
                }
                None => (Some(Vec::new()), 0),
                Some(other) => (None, scalar_int(&other).unwrap_or(0).max(0)),
            };
            if state.1 + 1 > gauge.limit {
                return Response::over(gauge_status(gauge, state.1, Code::OverLimit));
            }
            states.push(state);
        }
        let ttl = Some(Duration::from_secs(SLOT_TTL_SECONDS as u64));
        let statuses = gauges
            .iter()
            .zip(states)
            .map(|(gauge, (registry, in_flight))| {
                let value = match registry {
                    Some(mut slots) => {
                        match slots.iter_mut().find(|(id, _)| id == slot_id) {
                            Some(slot) => slot.1 = now,
                            None => slots.push((slot_id.to_owned(), now)),
                        }
                        Local::Slots(slots)
                    }
                    None => Local::Number(Number::Int(in_flight + 1)),
                };
                self.set_local(Entry::Gauge(gauge.scope.clone()), value, ttl);
                gauge_status(gauge, in_flight + 1, Code::Ok)
            })
            .collect();
        Response::ok(statuses)
    }

    pub async fn release_slots(&self, acquisition: &SlotAcquisition) {
        if acquisition.gauges.is_empty() || acquisition.slot_id.is_empty() {
            return;
        }
        if self.store.has_redis() {
            let reply = match self.store.parallel_release(
                RedisSend::Now,
                &acquisition.gauges,
                &acquisition.slot_id,
            ) {
                Some(pending) => self
                    .labelled("parallel_release_script", pending)
                    .await
                    .map_err(Failure::from),
                None => Err(Failure::no_redis()),
            };
            match reply {
                Ok(remaining) => {
                    let ttl = Some(Duration::from_secs(SLOT_TTL_SECONDS as u64));
                    for (scope, remaining) in acquisition.gauges.iter().zip(remaining) {
                        self.set_local(
                            Entry::Gauge(scope.clone()),
                            Local::Number(Number::Int(remaining.max(0))),
                            ttl,
                        );
                    }
                    return;
                }
                Err(error) => {
                    tracing::warn!(
                        "parallel_release_script failed, falling back to in-memory release: {error}"
                    );
                }
            }
        }
        self.release_in_memory(acquisition).await;
    }

    pub async fn defer_release(&self, acquisition: &SlotAcquisition, ctx: Ctx<'_>) -> bool {
        let Some(request) = self.batch(ctx) else {
            return false;
        };
        if acquisition.gauges.is_empty() || acquisition.slot_id.is_empty() {
            return false;
        }
        self.release_in_memory(acquisition).await;
        self.store
            .parallel_release_post_call(request, &acquisition.gauges, &acquisition.slot_id)
    }

    async fn release_in_memory(&self, acquisition: &SlotAcquisition) {
        let _guard = self.store.memory_guard();
        let ttl = Some(Duration::from_secs(SLOT_TTL_SECONDS as u64));
        for scope in &acquisition.gauges {
            let value = match self.local(Entry::Gauge(scope.clone())) {
                Some(Local::Slots(slots)) => {
                    if !slots.iter().any(|(id, _)| *id == acquisition.slot_id) {
                        continue;
                    }
                    Local::Slots(
                        slots
                            .into_iter()
                            .filter(|(id, _)| *id != acquisition.slot_id)
                            .collect(),
                    )
                }
                None => continue,
                Some(other) => {
                    Local::Number(Number::Int((scalar_int(&other).unwrap_or(0) - 1).max(0)))
                }
            };
            self.set_local(Entry::Gauge(scope.clone()), value, ttl);
        }
    }

    fn atomic_payload(&self, descriptor: &Descriptor, requests: i64, tokens: i64) -> Group {
        let scope = descriptor.scope();
        let limits: Limits = descriptor.limits();
        let window_size = descriptor.window_size(self.settings.window_size);
        let mut counters = Vec::new();
        let mut meta = Vec::new();
        for (kind, limit, increment) in [
            (Kind::Requests, limits.requests_per_unit, requests),
            (Kind::Tokens, limits.tokens_per_unit, tokens),
        ] {
            let Some(limit) = limit else { continue };
            if increment < 0 {
                continue;
            }
            let counter = counter_of(&scope, kind);
            counters.push(AtomicCounter {
                counter: counter.clone(),
                limit,
                increment,
                ttl_seconds: window_size,
                window_size_seconds: window_size,
            });
            meta.push(AtomicMeta {
                descriptor_key: descriptor.key.clone(),
                descriptor_value: descriptor.value.clone(),
                current_limit: limit,
                kind,
                counter,
                increment,
                ttl: window_size,
                window_size,
            });
        }
        (counters, meta)
    }

    pub async fn atomic_check_and_increment(
        &self,
        descriptors: &[Descriptor],
        increments: &[(i64, i64)],
        ctx: Ctx<'_>,
    ) -> Result<Response, Unverifiable> {
        let groups: Vec<Group> = descriptors
            .iter()
            .zip(increments)
            .map(|(descriptor, (requests, tokens))| {
                self.atomic_payload(descriptor, *requests, *tokens)
            })
            .filter(|(counters, _)| !counters.is_empty())
            .collect();
        if groups.is_empty() {
            return Ok(Response::ok(Vec::new()));
        }
        if self.store.has_redis() {
            return self.atomic_redis(groups, ctx).await;
        }
        let flat: Vec<AtomicMeta> = groups.into_iter().flat_map(|(_, meta)| meta).collect();
        let now = self.now();
        let _guard = self.store.memory_guard();
        Ok(self.atomic_in_memory(&flat, now))
    }

    fn atomic_response(reply: &AtomicReply, meta: &[AtomicMeta]) -> Result<Response, String> {
        match reply {
            AtomicReply::OverLimit {
                index,
                current,
                limit,
            } => {
                let meta = meta
                    .get(*index)
                    .ok_or("an over-limit index outside the counters")?;
                Ok(Response::over(atomic_status(
                    meta,
                    Code::OverLimit,
                    *limit,
                    *current,
                )))
            }
            AtomicReply::Applied(applied) => {
                if applied.len() != meta.len() {
                    return Err("a reply that does not match its counters".to_owned());
                }
                let statuses = applied
                    .iter()
                    .zip(meta)
                    .map(|(applied, meta)| {
                        atomic_status(meta, Code::Ok, meta.current_limit, applied.count)
                    })
                    .collect();
                let windows = applied
                    .iter()
                    .zip(meta)
                    .map(|(applied, meta)| WindowIdentity {
                        counter: meta.counter.clone(),
                        window_start: applied.window_start.to_string(),
                        backend: Backend::Redis,
                    })
                    .collect::<Vec<_>>();
                Ok(Response {
                    overall: Code::Ok,
                    statuses,
                    windows: Some(identities(windows)),
                })
            }
        }
    }

    fn atomic_refunds(reply: &AtomicReply, meta: &[AtomicMeta]) -> Vec<Refund> {
        let AtomicReply::Applied(applied) = reply else {
            return Vec::new();
        };
        applied
            .iter()
            .zip(meta)
            .map(|(applied, meta)| Refund {
                counter: meta.counter.clone(),
                window_start: applied.window_start.to_string(),
                increment: meta.increment,
            })
            .collect()
    }

    async fn atomic_redis(
        &self,
        groups: Vec<Group>,
        ctx: Ctx<'_>,
    ) -> Result<Response, Unverifiable> {
        if groups
            .iter()
            .any(|(_, meta)| meta.iter().any(|meta| meta.increment == i64::MAX))
        {
            let error = Failure::raised(
                "OverflowError",
                "an increment beyond a signed 64-bit integer",
            );
            self.unverifiable(ctx.policy, "check_and_increment_by_n_script", &error)?;
            self.log_atomic_failure("Refunded", 0, "prior", &error);
            let flat: Vec<AtomicMeta> = groups.into_iter().flat_map(|(_, meta)| meta).collect();
            let now = self.now();
            let _guard = self.store.memory_guard();
            return Ok(self.atomic_in_memory(&flat, now));
        }
        if let Some(request) = self.batch(ctx) {
            let declared: Vec<_> = groups
                .iter()
                .map(|(counters, _)| {
                    self.store
                        .check_and_increment(RedisSend::PreCall(request), counters)
                })
                .collect();
            return self.atomic_pipelined(groups, declared, ctx).await;
        }
        let single = groups.len() == 1;
        let mut applied: Vec<Refund> = Vec::new();
        let mut statuses = Vec::new();
        let mut windows = Vec::new();
        for (index, (counters, meta)) in groups.iter().enumerate() {
            let reply = match self.store.check_and_increment(RedisSend::Now, counters) {
                Some(pending) => self
                    .labelled("check_and_increment_by_n_script", pending)
                    .await
                    .map_err(Failure::from),
                None => Err(Failure::no_redis()),
            };
            let reply = match reply {
                Ok(reply) => reply,
                Err(error) => {
                    self.refund(applied).await;
                    self.unverifiable(ctx.policy, "check_and_increment_by_n_script", &error)?;
                    self.log_atomic_failure("Refunded", index, "prior", &error);
                    let flat: Vec<AtomicMeta> =
                        groups.iter().flat_map(|(_, meta)| meta.clone()).collect();
                    let now = self.now();
                    let _guard = self.store.memory_guard();
                    return Ok(self.atomic_in_memory(&flat, now));
                }
            };
            let response = match Self::atomic_response(&reply, meta) {
                Ok(response) => response,
                Err(error) => {
                    let error = Failure::raised("ValueError", error);
                    self.refund(applied).await;
                    self.unverifiable(ctx.policy, "check_and_increment_by_n_script", &error)?;
                    let flat: Vec<AtomicMeta> =
                        groups.iter().flat_map(|(_, meta)| meta.clone()).collect();
                    let now = self.now();
                    let _guard = self.store.memory_guard();
                    return Ok(self.atomic_in_memory(&flat, now));
                }
            };
            if response.is_over() {
                self.refund(applied).await;
                return Ok(response);
            }
            if single {
                return Ok(response);
            }
            applied.extend(Self::atomic_refunds(&reply, meta));
            windows.extend(response.windows().iter().cloned());
            statuses.extend(response.statuses);
        }
        Ok(Response {
            overall: Code::Ok,
            statuses,
            windows: Some(identities(windows)),
        })
    }

    async fn atomic_pipelined(
        &self,
        groups: Vec<Group>,
        declared: Vec<Option<Pending<AtomicReply>>>,
        ctx: Ctx<'_>,
    ) -> Result<Response, Unverifiable> {
        let replies: Vec<Result<AtomicReply, Failure>> =
            join_all(declared.into_iter().map(|declared| async move {
                match declared {
                    Some(pending) => self
                        .labelled("check_and_increment_by_n_script", pending)
                        .await
                        .map_err(Failure::from),
                    None => Err(Failure::no_redis()),
                }
            }))
            .await;
        let responses: Vec<Result<Response, Failure>> = replies
            .iter()
            .zip(&groups)
            .map(|(reply, (_, meta))| match reply {
                Ok(reply) => Self::atomic_response(reply, meta)
                    .map_err(|error| Failure::raised("ValueError", error)),
                Err(error) => Err(Failure::raised(error.exception(), error.to_string())),
            })
            .collect();
        let mut applied_groups = 0;
        let mut applied = Vec::new();
        let mut statuses = Vec::new();
        let mut windows = Vec::new();
        for ((reply, response), (_, meta)) in replies.iter().zip(&responses).zip(&groups) {
            let (Ok(reply @ AtomicReply::Applied(_)), Ok(response)) = (reply, response) else {
                continue;
            };
            if response.overall != Code::Ok {
                continue;
            }
            applied_groups += 1;
            applied.extend(Self::atomic_refunds(reply, meta));
            statuses.extend(response.statuses.clone());
            windows.extend(response.windows().iter().cloned());
        }
        if let Some(over) = responses
            .iter()
            .flatten()
            .find(|response| response.is_over())
        {
            self.refund(applied).await;
            return Ok(over.clone());
        }
        if let Some(Err(error)) = responses.iter().find(|response| response.is_err()) {
            self.refund(applied).await;
            self.unverifiable(ctx.policy, "check_and_increment_by_n_script", error)?;
            self.log_atomic_failure("Refunding", applied_groups, "pipelined", error);
            let flat: Vec<AtomicMeta> = groups.iter().flat_map(|(_, meta)| meta.clone()).collect();
            let now = self.now();
            let _guard = self.store.memory_guard();
            return Ok(self.atomic_in_memory(&flat, now));
        }
        if responses.len() == 1
            && let Some(Ok(response)) = responses.into_iter().next()
        {
            return Ok(response);
        }
        Ok(Response {
            overall: Code::Ok,
            statuses,
            windows: Some(identities(windows)),
        })
    }

    fn log_atomic_failure(&self, refund: &str, groups: usize, which: &str, error: &Failure) {
        tracing::error!(
            "atomic_check_and_increment_by_n: Redis Lua execution failed ({}). {refund} {groups} {which} descriptors and falling back to in-memory enforcement, counters will diverge from Redis until window expires (window_size={}s): {error}",
            error.exception(),
            self.settings.window_size,
        );
    }

    fn atomic_in_memory(&self, metas: &[AtomicMeta], now: f64) -> Response {
        let now_int = now.trunc() as i64;
        let mut states: Vec<(bool, i64, String)> = Vec::new();
        for meta in metas {
            let start = self
                .local(Entry::Window(meta.counter.scope.clone()))
                .as_ref()
                .and_then(scalar_int);
            let expired = start.is_none_or(|start| now_int - start >= meta.window_size);
            let current = if expired {
                0
            } else {
                self.local(Entry::Counter(meta.counter.clone()))
                    .as_ref()
                    .and_then(scalar_int)
                    .unwrap_or(0)
            };
            let over = if meta.increment > 0 {
                current.saturating_add(meta.increment) > meta.current_limit
            } else {
                current >= meta.current_limit
            };
            if over {
                return Response::over(atomic_status(
                    meta,
                    Code::OverLimit,
                    meta.current_limit,
                    current,
                ));
            }
            let window_start = match start {
                Some(start) if !expired => start,
                _ => now_int,
            };
            states.push((expired, current, window_start.to_string()));
        }
        let mut expired_scopes: Vec<(Scope, i64)> = Vec::new();
        for (meta, (expired, _, _)) in metas.iter().zip(&states) {
            if !*expired {
                continue;
            }
            match expired_scopes
                .iter_mut()
                .find(|(scope, _)| *scope == meta.counter.scope)
            {
                Some(existing) => existing.1 = meta.window_size,
                None => expired_scopes.push((meta.counter.scope.clone(), meta.window_size)),
            }
        }
        for (scope, window_size) in &expired_scopes {
            for kind in [Kind::Requests, Kind::Tokens] {
                self.set_local(
                    Entry::Counter(counter_of(scope, kind)),
                    Local::Number(Number::Int(0)),
                    seconds(*window_size),
                );
            }
        }
        let mut statuses = Vec::new();
        let mut windows = Vec::new();
        for (meta, (expired, current, window_start)) in metas.iter().zip(states) {
            let new_counter = if expired {
                meta.increment
            } else {
                current + meta.increment
            };
            if expired {
                self.set_local(
                    Entry::Window(meta.counter.scope.clone()),
                    Local::Text(now_int.to_string()),
                    seconds(meta.window_size),
                );
            }
            self.set_local(
                Entry::Counter(meta.counter.clone()),
                Local::Number(Number::Int(new_counter)),
                seconds(meta.ttl),
            );
            statuses.push(atomic_status(
                meta,
                Code::Ok,
                meta.current_limit,
                new_counter,
            ));
            windows.push(WindowIdentity {
                counter: meta.counter.clone(),
                window_start,
                backend: Backend::Local,
            });
        }
        Response {
            overall: Code::Ok,
            statuses,
            windows: Some(identities(windows)),
        }
    }

    pub fn tpm_descriptors(descriptors: &[Descriptor]) -> Vec<Descriptor> {
        descriptors
            .iter()
            .filter(|descriptor| {
                !descriptor.is_project_io() && descriptor.limits().tokens_per_unit.is_some()
            })
            .map(|descriptor| Descriptor {
                key: descriptor.key.clone(),
                value: descriptor.value.clone(),
                rate_limit: Some(Limits {
                    tokens_per_unit: descriptor.limits().tokens_per_unit,
                    window_size: descriptor.limits().window_size,
                    declared: Declared {
                        tokens_per_unit: true,
                        window_size: true,
                        ..Declared::default()
                    },
                    ..Limits::default()
                }),
            })
            .collect()
    }

    pub async fn reserve_tpm_tokens(
        &self,
        descriptors: &[Descriptor],
        estimated: i64,
        scope_estimates: &[(&str, i64)],
        ctx: Ctx<'_>,
    ) -> Result<Response, Unverifiable> {
        let tpm = Self::tpm_descriptors(descriptors);
        if tpm.is_empty() {
            return Ok(Response::ok(Vec::new()));
        }
        let increments: Vec<(i64, i64)> = tpm
            .iter()
            .map(|descriptor| {
                let amount = scope_estimates
                    .iter()
                    .find(|(key, _)| *key == descriptor.key)
                    .map_or(estimated, |(_, amount)| *amount);
                (0, amount)
            })
            .collect();
        self.atomic_check_and_increment(&tpm, &increments, ctx)
            .await
    }

    pub async fn reserve_io_tokens(
        &self,
        descriptors: &[Descriptor],
        input_tokens: i64,
        output_tokens: i64,
        ctx: Ctx<'_>,
    ) -> Result<(Response, i64, i64), Unverifiable> {
        let itpm: Vec<Descriptor> = descriptors
            .iter()
            .filter(|descriptor| descriptor.key == PROJECT_ITPM)
            .cloned()
            .collect();
        let otpm: Vec<Descriptor> = descriptors
            .iter()
            .filter(|descriptor| descriptor.key == PROJECT_OTPM)
            .cloned()
            .collect();
        if itpm.is_empty() && otpm.is_empty() {
            return Ok((Response::ok(Vec::new()), 0, 0));
        }
        let itpm_response = if itpm.is_empty() {
            None
        } else {
            let increments: Vec<(i64, i64)> = itpm.iter().map(|_| (0, input_tokens)).collect();
            Some(
                self.atomic_check_and_increment(&itpm, &increments, ctx)
                    .await?,
            )
        };
        if let Some(response) = &itpm_response
            && response.is_over()
        {
            return Ok((response.clone(), 0, 0));
        }
        let itpm_reserved = if itpm_response.is_some() {
            input_tokens
        } else {
            0
        };
        if otpm.is_empty() {
            let response = itpm_response.unwrap_or_else(|| Response::ok(Vec::new()));
            return Ok((response, itpm_reserved, 0));
        }
        let increments: Vec<(i64, i64)> = otpm.iter().map(|_| (0, output_tokens)).collect();
        let otpm_response = self
            .atomic_check_and_increment(&otpm, &increments, ctx)
            .await?;
        if otpm_response.is_over() {
            if itpm_reserved > 0 {
                let scopes: Vec<(String, String)> = itpm
                    .iter()
                    .map(|descriptor| (descriptor.key.clone(), descriptor.value.clone()))
                    .collect();
                let windows = itpm_response
                    .as_ref()
                    .map(|response| response.windows().to_vec())
                    .unwrap_or_default();
                self.refund_reserved(&scopes, itpm_reserved, &windows, ctx)
                    .await;
            }
            return Ok((otpm_response, 0, 0));
        }
        let mut statuses = Vec::new();
        let mut windows = Vec::new();
        if let Some(response) = itpm_response {
            windows.extend(response.windows().iter().cloned());
            statuses.extend(response.statuses);
        }
        windows.extend(otpm_response.windows().iter().cloned());
        statuses.extend(otpm_response.statuses);
        Ok((
            Response {
                overall: Code::Ok,
                statuses,
                windows: Some(identities(windows)),
            },
            itpm_reserved,
            output_tokens,
        ))
    }

    pub async fn refund_reserved(
        &self,
        scopes: &[(String, String)],
        amount: i64,
        windows: &[WindowIdentity],
        ctx: Ctx<'_>,
    ) {
        if amount <= 0 || scopes.is_empty() {
            return;
        }
        let reserved: BTreeSet<(String, String)> = scopes.iter().cloned().collect();
        if windows.is_empty() {
            let operations = self.reservation_ops(scopes, &reserved, 0, amount);
            self.increment_with_ttl_preservation(&operations, ctx).await;
            return;
        }
        let operations = self.project_ops(scopes, &reserved, 0, amount, windows);
        self.reservation_aware(&operations).await;
    }

    pub fn reservation_ops(
        &self,
        targets: &[(String, String)],
        reserved_scopes: &BTreeSet<(String, String)>,
        actual: i64,
        reserved: i64,
    ) -> Vec<Increment> {
        targets
            .iter()
            .filter_map(|target| {
                let amount = if reserved_scopes.contains(target) {
                    actual - reserved
                } else {
                    actual
                };
                (amount != 0).then(|| self.tokens_increment(&scope_of(target), amount))
            })
            .collect()
    }

    pub fn project_ops(
        &self,
        targets: &[(String, String)],
        reserved_scopes: &BTreeSet<(String, String)>,
        actual: i64,
        reserved: i64,
        windows: &[WindowIdentity],
    ) -> Vec<Op> {
        targets
            .iter()
            .filter_map(|target| {
                let is_reserved = reserved_scopes.contains(target);
                let amount = if is_reserved {
                    actual - reserved
                } else {
                    actual
                };
                if amount == 0 {
                    return None;
                }
                let increment = self.tokens_increment(&scope_of(target), amount);
                let identity = windows
                    .iter()
                    .find(|identity| identity.counter == increment.counter);
                Some(match identity {
                    Some(identity) if is_reserved => Op::Guarded(GuardedOp {
                        expected_window_start: identity.window_start.clone(),
                        backend: identity.backend,
                        increment,
                    }),
                    _ => Op::Plain(increment),
                })
            })
            .collect()
    }

    pub async fn settle(&self, settlement: Settlement, ctx: Ctx<'_>, post_call: bool) {
        match settlement {
            Settlement::Plain(operations) if operations.is_empty() => {}
            Settlement::Plain(operations) if post_call => {
                self.increment_many_post_call(&operations, ctx).await;
            }
            Settlement::Plain(operations) => self.increment_many(&operations).await,
            Settlement::Project(operations) if operations.is_empty() => {}
            Settlement::Project(operations) => self.reservation_aware(&operations).await,
        }
    }

    pub fn tokens_increment(&self, scope: &Scope, amount: i64) -> Increment {
        Increment {
            counter: counter_of(scope, Kind::Tokens),
            amount: Number::Int(amount),
            ttl: seconds(self.settings.window_size),
        }
    }

    pub async fn increment_with_ttl_preservation(&self, operations: &[Increment], ctx: Ctx<'_>) {
        if operations.is_empty() {
            return;
        }
        if !self.store.has_redis() {
            self.labelled(
                "async_increment_pipeline",
                self.store.increment_many(operations),
            )
            .await;
            return;
        }
        let scopes: Vec<Scope> = operations
            .iter()
            .map(|operation| operation.counter.scope.clone())
            .collect();
        for group in self.slot_groups(&scopes, ctx.policy) {
            let members: Vec<Increment> = group
                .iter()
                .map(|&index| operations[index].clone())
                .collect();
            if let Some(request) = self.batch(ctx)
                && self
                    .store
                    .token_increment_post_call(request, members.clone())
            {
                continue;
            }
            let outcome = match self.store.token_increment(RedisSend::Now, &members) {
                Some(pending) => self
                    .labelled("token_increment_script", pending)
                    .await
                    .map(|_| ())
                    .map_err(Failure::from),
                None => Err(Failure::no_redis()),
            };
            if let Err(error) = outcome {
                tracing::warn!(
                    "TTL preservation failed, falling back to regular pipeline: {error}"
                );
                self.labelled(
                    "async_increment_pipeline",
                    self.store.increment_many(operations),
                )
                .await;
                return;
            }
        }
    }

    pub async fn increment_many(&self, operations: &[Increment]) {
        self.labelled(
            "async_increment_pipeline",
            self.store.increment_many(operations),
        )
        .await;
    }

    pub async fn increment_many_post_call(&self, operations: &[Increment], ctx: Ctx<'_>) {
        match self.batch(ctx) {
            Some(request) => self.store.increment_many_post_call(request, operations),
            None => {
                self.labelled(
                    "async_increment_pipeline",
                    self.store.increment_many(operations),
                )
                .await
            }
        }
    }

    pub async fn apply_usage(&self, operations: &[Op], ctx: Ctx<'_>) {
        let plain: Vec<Increment> = operations
            .iter()
            .filter_map(|operation| match operation {
                Op::Plain(increment) => Some(increment.clone()),
                Op::Guarded(_) => None,
            })
            .collect();
        let guarded: Vec<Op> = operations
            .iter()
            .filter(|operation| matches!(operation, Op::Guarded(_)))
            .cloned()
            .collect();
        self.increment_with_ttl_preservation(&plain, ctx).await;
        if !guarded.is_empty() {
            self.reservation_aware(&guarded).await;
        }
    }

    pub async fn reservation_aware(&self, operations: &[Op]) {
        for operation in operations {
            if let Op::Plain(increment) = operation {
                let _ = self
                    .labelled("async_increment", self.store.increment(increment))
                    .await;
            }
        }
        let guarded: Vec<&GuardedOp> = operations
            .iter()
            .filter_map(|operation| match operation {
                Op::Guarded(guarded) => Some(guarded),
                Op::Plain(_) => None,
            })
            .collect();
        let local: Vec<&GuardedOp> = guarded
            .iter()
            .copied()
            .filter(|op| op.backend == Backend::Local)
            .collect();
        let remote: Vec<&GuardedOp> = guarded
            .iter()
            .copied()
            .filter(|op| op.backend != Backend::Local)
            .collect();
        if !local.is_empty() {
            let _guard = self.store.memory_guard();
            for operation in local {
                let counter = &operation.increment.counter;
                let active = self
                    .local(Entry::Window(counter.scope.clone()))
                    .as_ref()
                    .and_then(scalar_text);
                if active.as_deref() != Some(operation.expected_window_start.as_str()) {
                    continue;
                }
                let current = match self.local(Entry::Counter(counter.clone())) {
                    Some(Local::Number(number)) => number.as_f64(),
                    Some(Local::Text(text)) => text.trim().parse::<f64>().unwrap_or(0.0),
                    _ => 0.0,
                };
                self.set_local(
                    Entry::Counter(counter.clone()),
                    Local::Number(Number::Float(current + operation.increment.amount.as_f64())),
                    operation.increment.ttl,
                );
            }
        }
        for operation in remote {
            let increment = &operation.increment;
            if self.store.has_redis() {
                let guarded = GuardedIncrement {
                    counter: increment.counter.clone(),
                    expected_window_start: operation.expected_window_start.clone(),
                    amount: increment.amount,
                    ttl: increment.ttl,
                };
                let outcome = match self.store.guarded_increment(RedisSend::Now, &guarded) {
                    Some(pending) => self
                        .labelled("window_guarded_token_increment_script", pending)
                        .await
                        .map(|_| ())
                        .map_err(Failure::from),
                    None => Err(Failure::no_redis()),
                };
                match outcome {
                    Ok(()) => continue,
                    Err(error) => tracing::warn!(
                        "Window-guarded token adjustment failed for {}: {error}",
                        python_counter(&increment.counter)
                    ),
                }
            }
            if increment.amount.as_f64() > 0.0 {
                let _ = self
                    .labelled("async_increment", self.store.increment(increment))
                    .await;
            }
        }
    }
}
