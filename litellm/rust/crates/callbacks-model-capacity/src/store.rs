use std::{sync::Arc, time::Duration};

use litellm_callbacks_parallel_request_limiter::{
    Backend, Code, Ctx, Descriptor as LimiterDescriptor, Engine, GuardedOp, Op, Response,
    Status as LimiterStatus, WindowIdentity,
};
use litellm_python_compat::{number, repr::str_repr};
use litellm_redis_client::{CallTag, current_tag, tagged};
use litellm_redis_store::{
    Error as StoreError, Number as StoreNumber, RequestRedis, Send as RedisSend,
    rate_limits::{
        Counter as StoreCounter, Entry, GuardedIncrement, Increment as StoreIncrement, Kind, Local,
        Scope,
    },
};
use serde_json::{Number, Value};

use crate::{
    error::CapacityError,
    host::{CounterStore, SeamFuture},
    model::{
        Counter, CounterKind, Descriptor, Increment, LimiterResponse, ReservationOp, Status, Window,
    },
};

pub type Relabel = Arc<dyn Fn(&CallTag, &'static str) -> CallTag + Send + Sync>;

pub struct StoreLimiter {
    engine: Engine,
    relabel: Option<Relabel>,
}

fn store_counter(counter: &Counter) -> StoreCounter {
    StoreCounter {
        scope: Scope {
            descriptor: counter.key.clone(),
            value: counter.value.clone(),
        },
        kind: match counter.kind {
            CounterKind::Requests => Kind::Requests,
            CounterKind::Tokens => Kind::Tokens,
        },
    }
}

fn counter_of(counter: &StoreCounter) -> Counter {
    Counter {
        key: counter.scope.descriptor.clone(),
        value: counter.scope.value.clone(),
        kind: match counter.kind {
            Kind::Requests => CounterKind::Requests,
            Kind::Tokens => CounterKind::Tokens,
        },
    }
}

fn local_value(local: Option<Local>) -> Option<Value> {
    match local? {
        Local::Number(StoreNumber::Int(number)) => Some(Value::from(number)),
        Local::Number(StoreNumber::Float(number)) => Number::from_f64(number).map(Value::Number),
        Local::Text(text) => Some(Value::String(text)),
        Local::Slots(_) => None,
    }
}

fn store_error(error: StoreError) -> CapacityError {
    let kind: &'static str = error.kind().into();
    CapacityError::Store(format!(
        "({}, {})",
        str_repr(kind),
        str_repr(&litellm_error::Report::new(&error).to_string())
    ))
}

fn backfill(ttl: &Value) -> Result<Option<Duration>, CapacityError> {
    match ttl {
        Value::Null => Ok(None),
        Value::Number(number) => number
            .as_f64()
            .map(Duration::try_from_secs_f64)
            .transpose()
            .map_err(|error| CapacityError::Config(error.to_string())),
        other => Err(CapacityError::Config(format!("{other} is not a ttl"))),
    }
}

fn counted(text: &str) -> Result<Local, CapacityError> {
    let number = number::float(text).ok_or_else(|| {
        CapacityError::Store(format!(
            "could not convert string to float: {}",
            str_repr(text)
        ))
    })?;
    Ok(Local::Number(
        if number.is_finite() && number.fract() == 0.0 && number.abs() < 9.2e18 {
            StoreNumber::Int(number as i64)
        } else {
            StoreNumber::Float(number)
        },
    ))
}

fn limiter_descriptors(
    descriptors: &[Descriptor],
) -> Result<Vec<LimiterDescriptor>, CapacityError> {
    serde_json::to_value(descriptors)
        .and_then(serde_json::from_value)
        .map_err(|error| CapacityError::Type(error.to_string()))
}

fn status(status: LimiterStatus) -> Status {
    Status {
        code: match status.code {
            Code::Ok => "OK".into(),
            Code::OverLimit => "OVER_LIMIT".into(),
        },
        current_limit: Number::from(status.current_limit),
        limit_remaining: Some(Number::from(status.limit_remaining)),
        rate_limit_type: <&'static str>::from(status.rate_limit_type).to_owned(),
        descriptor_key: status.descriptor_key,
        descriptor_value: status.descriptor_value,
    }
}

fn window(identity: WindowIdentity) -> Window {
    Window {
        counter: counter_of(&identity.counter),
        start: identity.window_start,
        backend: match identity.backend {
            Backend::Redis => "redis".into(),
            Backend::Local => "local".into(),
        },
    }
}

fn response(response: Response) -> LimiterResponse {
    LimiterResponse {
        overall_code: if response.is_over() {
            "OVER_LIMIT".into()
        } else {
            "OK".into()
        },
        statuses: response.statuses.into_iter().map(status).collect(),
        reservation_windows: response
            .windows
            .unwrap_or_default()
            .into_iter()
            .map(window)
            .collect(),
    }
}

fn increment(counter: &Counter, amount: i64, ttl: Option<i64>) -> StoreIncrement {
    StoreIncrement {
        counter: store_counter(counter),
        amount: StoreNumber::Int(amount),
        ttl: ttl.map(|ttl| Duration::from_secs(u64::try_from(ttl).unwrap_or(0))),
    }
}

fn operation(operation: &ReservationOp) -> Op {
    match operation {
        ReservationOp::Increment {
            counter,
            amount,
            ttl,
        } => Op::Plain(increment(counter, *amount, *ttl)),
        ReservationOp::Guarded {
            counter,
            amount,
            ttl,
            expected_window_start,
            backend,
        } => Op::Guarded(GuardedOp {
            increment: increment(counter, *amount, *ttl),
            expected_window_start: expected_window_start.clone(),
            backend: if backend == "local" {
                Backend::Local
            } else {
                Backend::Redis
            },
        }),
    }
}

fn guarded_failure(counter: &StoreCounter) -> String {
    let kind: &'static str = counter.kind.into();
    format!(
        "Window-guarded token adjustment failed for RateLimitCounter(scope=RateLimitScope(descriptor_key={}, descriptor_value={}), kind={})",
        str_repr(&counter.scope.descriptor),
        str_repr(&counter.scope.value),
        str_repr(kind)
    )
}

impl StoreLimiter {
    pub fn new(engine: Engine, relabel: Option<Relabel>) -> Self {
        Self { engine, relabel }
    }

    async fn batched(&self, operations: &[Op], request: &RequestRedis) {
        let store = self.engine.store();
        let plain: Vec<StoreIncrement> = operations
            .iter()
            .filter_map(|operation| match operation {
                Op::Plain(increment) => Some(increment.clone()),
                Op::Guarded(_) => None,
            })
            .collect();
        if !plain.is_empty() {
            store.increment_many_post_call(request, &plain);
        }
        let mut local = Vec::new();
        for operation in operations {
            let Op::Guarded(guarded) = operation else {
                continue;
            };
            if guarded.backend == Backend::Local {
                local.push(operation.clone());
                continue;
            }
            let increment = &guarded.increment;
            store.guarded_increment_post_call(
                request,
                GuardedIncrement {
                    counter: increment.counter.clone(),
                    expected_window_start: guarded.expected_window_start.clone(),
                    amount: increment.amount,
                    ttl: increment.ttl,
                },
                guarded_failure(&increment.counter),
            );
        }
        if !local.is_empty() {
            self.engine.reservation_aware(&local).await;
        }
    }

    async fn labelled<F: Future>(&self, call_type: &'static str, operation: F) -> F::Output {
        match (self.relabel.as_ref(), current_tag()) {
            (Some(relabel), Some(tag)) => tagged(relabel(&tag, call_type), operation).await,
            _ => operation.await,
        }
    }

    async fn read(
        &self,
        counter: &Counter,
        backfill_ttl: &Value,
    ) -> Result<Option<Value>, CapacityError> {
        let store = self.engine.store();
        let entry = Entry::Counter(store_counter(counter));
        if !store.has_redis() {
            return Ok(local_value(store.get_local(&entry).map_err(store_error)?));
        }
        let pending = store
            .read_counters(RedisSend::Now, &[store_counter(counter)])
            .ok_or_else(|| {
                CapacityError::Store("the rate limiter has no Redis to run its scripts on".into())
            })?;
        let reply = self
            .labelled("batch_counter_read_script", pending)
            .await
            .map_err(store_error)?;
        let Some(count) = reply.get(1).cloned().flatten() else {
            return Ok(None);
        };
        let value = counted(&count)?;
        if store.get_local(&entry).map_err(store_error)?.is_none() {
            store
                .set_local(&entry, value.clone(), backfill(backfill_ttl)?)
                .map_err(store_error)?;
        }
        Ok(local_value(Some(value)))
    }
}

impl CounterStore for StoreLimiter {
    fn read_counter<'a>(
        &'a self,
        counter: &'a Counter,
        backfill_ttl: &'a Value,
    ) -> SeamFuture<'a, Option<Value>> {
        Box::pin(self.read(counter, backfill_ttl))
    }

    fn read_local_counter(&self, counter: &Counter) -> Result<Option<Value>, CapacityError> {
        self.engine
            .store()
            .get_local(&Entry::Counter(store_counter(counter)))
            .map(local_value)
            .map_err(store_error)
    }

    fn check_and_increment<'a>(
        &'a self,
        descriptors: &'a [Descriptor],
        increments: &'a [Increment],
        ctx: Ctx<'a>,
    ) -> SeamFuture<'a, LimiterResponse> {
        Box::pin(async move {
            let descriptors = limiter_descriptors(descriptors)?;
            let increments: Vec<(i64, i64)> = increments
                .iter()
                .map(|increment| (increment.requests, increment.tokens))
                .collect();
            self.engine
                .atomic_check_and_increment(&descriptors, &increments, ctx)
                .await
                .map(response)
                .map_err(|_| CapacityError::Unverifiable)
        })
    }

    fn should_rate_limit<'a>(
        &'a self,
        descriptors: &'a [Descriptor],
        read_only: bool,
        skip_parallel_check: bool,
        ctx: Ctx<'a>,
    ) -> SeamFuture<'a, LimiterResponse> {
        Box::pin(async move {
            let descriptors = limiter_descriptors(descriptors)?;
            self.engine
                .should_rate_limit(
                    &descriptors,
                    read_only,
                    false,
                    None,
                    skip_parallel_check,
                    ctx,
                )
                .await
                .map(response)
                .map_err(|_| CapacityError::Unverifiable)
        })
    }

    fn apply<'a>(
        &'a self,
        operations: &'a [ReservationOp],
        redis: Option<&'a RequestRedis>,
    ) -> SeamFuture<'a, ()> {
        Box::pin(async move {
            let operations: Vec<Op> = operations.iter().map(operation).collect();
            match redis.filter(|_| self.engine.has_redis()) {
                Some(request) => self.batched(&operations, request).await,
                None => self.engine.reservation_aware(&operations).await,
            }
            Ok(())
        })
    }
}
