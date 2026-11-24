use std::time::Duration;

use futures_util::future::join_all;
use litellm_redis_client::{Declared, Error as RedisError, Expiry, RedisBatch, commands};

use super::DualCache;
use crate::{Error, Number, RequestRedis, Send, failure_log::redis_failure, memory::Counter};

const INCREMENT_FAILED: &str =
    "Redis async_increment_cache failed, falling back to in-memory result";
const PIPELINE_FAILED: &str =
    "Redis async_increment_cache_pipeline failed, falling back to in-memory result";

/// A `RedisPipelineIncrementOperation`.
pub(crate) struct PipelineIncrement {
    pub(crate) key: String,
    pub(crate) amount: Number,
    pub(crate) ttl: Option<Duration>,
}

impl<V: Counter, R> DualCache<V, R> {
    /// `DualCache.async_increment_cache`: memory, then `INCRBYFLOAT` keeping an existing TTL, else
    /// setting `ttl` or Redis's default. `None` when memory refuses the increment, which then never
    /// reaches Redis, and the memory count when Redis fails. Both only log, as there.
    pub(crate) async fn increment(
        &self,
        key: &str,
        amount: Number,
        ttl: Option<Duration>,
    ) -> Option<Number> {
        let local = self
            .memory
            .increment(key, amount, ttl)
            .map_err(|error| redis_failure!(&Error::from(error), INCREMENT_FAILED))
            .ok()?;
        let Some(remote) = self.redis.get() else {
            return Some(local);
        };
        let expiry = Expiry::KeepExisting(ttl.unwrap_or(remote.default_ttl));
        let increment = commands::incr_by_float(key, amount.into(), Some(expiry));
        match remote.client.send(increment).await {
            Ok(total) => Some(Number::Float(total)),
            Err(error) => {
                redis_failure!(&error, INCREMENT_FAILED);
                Some(local)
            }
        }
    }

    /// `DualCache.async_increment_cache_pipeline`: memory, then one pipeline of `INCRBYFLOAT` with
    /// `EXPIRE` per increment. `None` when memory refuses an increment, which leaves the ones before
    /// it applied and sends nothing, and the memory totals when Redis fails.
    pub(crate) async fn increment_pipeline(
        &self,
        increments: &[PipelineIncrement],
    ) -> Option<Vec<Number>> {
        let local = increments
            .iter()
            .map(|increment| {
                self.memory
                    .increment(&increment.key, increment.amount, increment.ttl)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| redis_failure!(&Error::from(error), PIPELINE_FAILED))
            .ok()?;
        match self.increment_pipeline_remote(increments).await {
            None => Some(local),
            Some(Ok(totals)) => Some(totals),
            Some(Err(error)) => {
                redis_failure!(&error, PIPELINE_FAILED);
                Some(local)
            }
        }
    }

    /// `RedisCache.async_increment_pipeline`, Redis alone: one pipeline of `INCRBYFLOAT` with
    /// `EXPIRE` per increment. `None` without Redis or with nothing to increment.
    pub(crate) async fn increment_pipeline_remote(
        &self,
        increments: &[PipelineIncrement],
    ) -> Option<Result<Vec<Number>, RedisError>> {
        if increments.is_empty() {
            return None;
        }
        let pipeline = RedisBatch::pipelined(self.redis.client()?.clone());
        let totals = increments
            .iter()
            .map(|increment| {
                pipeline.add(commands::incr_by_float(
                    &increment.key,
                    increment.amount.into(),
                    increment.ttl.map(Expiry::Refresh),
                ))
            })
            .collect::<Vec<_>>();
        Some(
            Declared::all(totals)
                .await
                .into_iter()
                .map(|total| total.map(Number::Float))
                .collect(),
        )
    }

    /// `DualCache.async_increment_cache_pipeline_post_call`: memory now, stopping at the first
    /// increment memory refuses and only logging it, then each increment on the request's post-call
    /// batch, whose failure is logged once it settles.
    pub(crate) fn increment_pipeline_post_call(
        &self,
        request: &RequestRedis,
        increments: &[PipelineIncrement],
    ) {
        if let Err(error) = increments.iter().try_for_each(|increment| {
            self.memory
                .increment(&increment.key, increment.amount, increment.ttl)
                .map(drop)
        }) {
            tracing::warn!(%error, "in-memory increment failed");
        }
        let Some(batch) = self.redis.batch(Send::PostCall(request)) else {
            return;
        };
        let declared = increments
            .iter()
            .map(|increment| {
                batch.add(commands::incr_by_float(
                    &increment.key,
                    increment.amount.into(),
                    increment.ttl.map(Expiry::KeepExisting),
                ))
            })
            .collect::<Vec<_>>();
        request.spawn(async move {
            join_all(declared.into_iter().map(Declared::settled))
                .await
                .into_iter()
                .filter_map(Result::err)
                .for_each(|error| redis_failure!(&error, "deferred Redis increment failed"));
        });
    }
}
