//! router-core's [`RouterCache`] over the router's stores. Each method makes the store call Python's
//! `RoutingStores.answer` makes for the same cache step (`litellm/router_rust/routing_stores.py`),
//! so the gateway writes the bytes and sends the commands a Python pod does.

use std::{collections::HashMap, sync::Arc, time::Duration};

use litellm_redis_store::router_usage::Increment;
use litellm_redis_store::{
    Error, Number, Remote, RouterTiers, Send,
    deployment_health::{DeploymentHealth, HealthState as StoredHealth},
    router_cooldowns::{Cooldown, RouterCooldowns},
    router_usage::{COUNTER_TTL, Counter, Kind, RouterUsage, Scope},
};
use litellm_router_core::{
    HealthState, InMemoryRouterCache, LimitCounter, LimitKind, Outcomes, RouterCache, UsageCounts,
    UsageWindow, health::CooldownValue,
};

/// The deployments' routing state: cooldowns, usage and health checks in memory, shared through
/// the router's Redis when there is one, and this process's own success and failure counts, which
/// never leave it.
pub struct RoutingCache {
    cooldowns: Arc<RouterCooldowns>,
    usage: RouterUsage,
    health: DeploymentHealth,
    outcomes: InMemoryRouterCache,
}

impl RoutingCache {
    pub fn new(remote: Option<Remote>, tiers: RouterTiers) -> Self {
        Self {
            cooldowns: Arc::new(RouterCooldowns::new(remote.clone(), tiers.cooldowns())),
            usage: RouterUsage::new(remote.clone(), tiers.usage()),
            health: DeploymentHealth::new(remote, tiers.default_in_memory_ttl),
            outcomes: InMemoryRouterCache::default(),
        }
    }

    /// Each counter's value, `None` for one nothing counted.
    pub(crate) async fn counted(&self, counters: &[Counter]) -> Result<Vec<Option<Number>>, Error> {
        self.usage.get_many(counters).await
    }
}

fn cache_error(error: Error) -> litellm_cache::Error {
    match error {
        Error::Memory(error) => error,
        other => litellm_cache::Error::unavailable(other),
    }
}

/// A whole amount as an `int`, as router-core serializes it for the Python host, so the stored
/// cooldown and the `INCRBYFLOAT` argument are the bytes Python writes.
fn whole_as_int(amount: f64) -> Number {
    if amount.fract() == 0.0 && amount.abs() < i64::MAX as f64 {
        Number::Int(amount as i64)
    } else {
        Number::Float(amount)
    }
}

fn stored(cooldown: &CooldownValue) -> Cooldown {
    Cooldown {
        exception_received: cooldown.exception_received.clone(),
        status_code: cooldown.status_code.clone(),
        timestamp: Number::Float(cooldown.timestamp),
        cooldown_time: whole_as_int(cooldown.cooldown_time),
    }
}

fn read(cooldown: Cooldown) -> CooldownValue {
    CooldownValue {
        exception_received: cooldown.exception_received,
        status_code: cooldown.status_code,
        timestamp: cooldown.timestamp.as_f64(),
        cooldown_time: cooldown.cooldown_time.as_f64(),
    }
}

fn health(state: StoredHealth) -> HealthState {
    HealthState {
        is_healthy: state.is_healthy,
        timestamp: state.timestamp.as_f64(),
    }
}

pub(crate) fn counter(window: &UsageWindow, kind: Kind) -> Counter {
    Counter {
        scope: Scope::GlobalRouter,
        deployment_id: window.deployment_id.clone(),
        model: Some(window.model.clone()),
        window: window.minute.clone(),
        kind,
    }
}

/// The counter Python's `ModelRateLimitingCheck` keeps the same count in
/// (`UsageCounter.for_deployment`).
fn limit_counter(counter: &LimitCounter) -> Counter {
    Counter {
        scope: Scope::Deployment,
        deployment_id: counter.deployment_id.clone(),
        model: Some(counter.model.clone()),
        window: counter.window.clone(),
        kind: match counter.kind {
            LimitKind::Tpm => Kind::Tpm,
            LimitKind::Rpm => Kind::Rpm,
            LimitKind::Tpd => Kind::Tpd,
            LimitKind::Rpd => Kind::Rpd,
        },
    }
}

impl RouterCache for RoutingCache {
    async fn cooldowns(
        &self,
        ids: &[String],
    ) -> Result<Vec<Option<CooldownValue>>, litellm_cache::Error> {
        let active = self
            .cooldowns
            .active(Send::Now, ids)
            .map_err(cache_error)?
            .await
            .map_err(cache_error)?;
        let found: HashMap<String, Cooldown> = active.into_iter().collect();
        Ok(ids
            .iter()
            .map(|id| found.get(id).cloned().map(read))
            .collect())
    }

    async fn put_cooldown(
        &self,
        id: &str,
        cooldown: &CooldownValue,
        ttl: Duration,
    ) -> Result<(), litellm_cache::Error> {
        self.cooldowns
            .put(id, &stored(cooldown), ttl)
            .await
            .map_err(cache_error)
    }

    async fn increment_allowed_fails(
        &self,
        id: &str,
        suffix: Option<&str>,
        ttl: Duration,
    ) -> Result<f64, litellm_cache::Error> {
        self.cooldowns
            .increment_allowed_fails(id, suffix, ttl)
            .await
            .map(|count| count as f64)
            .map_err(cache_error)
    }

    async fn add_usage(
        &self,
        window: &UsageWindow,
        tokens: f64,
        requests: f64,
    ) -> Result<(), litellm_cache::Error> {
        let written = self
            .usage
            .increment_deployment(
                &window.deployment_id,
                Some(&window.model),
                &window.minute,
                whole_as_int(tokens),
                requests as i64,
            )
            .await;
        if let Err(error) = written {
            tracing::debug!(%error, "_increment_deployment_usage: usage was not recorded");
        }
        Ok(())
    }

    async fn add_request(&self, window: &UsageWindow) -> Result<(), litellm_cache::Error> {
        self.usage
            .increment(&counter(window, Kind::Rpm), Number::Int(1), COUNTER_TTL)
            .await;
        Ok(())
    }

    async fn usage(&self, window: &UsageWindow) -> Result<UsageCounts, litellm_cache::Error> {
        let read = self
            .usage
            .get_many(&[counter(window, Kind::Tpm), counter(window, Kind::Rpm)])
            .await
            .map_err(cache_error)?;
        let count = |index: usize| read.get(index).copied().flatten().map(Number::as_f64);
        Ok(UsageCounts {
            tokens: count(0),
            requests: count(1),
        })
    }

    async fn health_states(&self) -> Result<HashMap<String, HealthState>, litellm_cache::Error> {
        let states = self.health.get().await.unwrap_or_else(|error| {
            tracing::debug!(%error, "DeploymentHealthCache::async_get_states");
            None
        });
        Ok(states
            .unwrap_or_default()
            .0
            .into_iter()
            .map(|(id, state)| (id, health(state)))
            .collect())
    }

    async fn limit_usage(
        &self,
        counter: &LimitCounter,
        limit: f64,
    ) -> Result<Option<f64>, litellm_cache::Error> {
        self.usage
            .current(&limit_counter(counter), whole_as_int(limit))
            .await
            .map(|count| count.map(Number::as_f64))
            .map_err(cache_error)
    }

    async fn add_limit_usage(
        &self,
        counter: &LimitCounter,
        amount: i64,
        ttl: Duration,
    ) -> Result<Option<f64>, litellm_cache::Error> {
        Ok(self
            .usage
            .increment(&limit_counter(counter), Number::Int(amount), ttl)
            .await
            .map(Number::as_f64))
    }

    async fn add_limit_usages(
        &self,
        increments: &[(LimitCounter, i64, Duration)],
    ) -> Result<Option<Vec<f64>>, litellm_cache::Error> {
        let increments: Vec<Increment> = increments
            .iter()
            .map(|(counter, amount, ttl)| Increment {
                counter: limit_counter(counter),
                amount: Number::Int(*amount),
                ttl: Some(*ttl),
            })
            .collect();
        Ok(self
            .usage
            .increment_pipeline(&increments)
            .await
            .map(|counts| counts.into_iter().map(Number::as_f64).collect()))
    }

    async fn outcomes(&self, id: &str) -> Result<Outcomes, litellm_cache::Error> {
        self.outcomes.outcomes(id).await
    }

    async fn count_success(&self, id: &str) -> Result<(), litellm_cache::Error> {
        self.outcomes.count_success(id).await
    }

    async fn count_failure(&self, id: &str) -> Result<(), litellm_cache::Error> {
        self.outcomes.count_failure(id).await
    }
}
