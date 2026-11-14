use std::{
    sync::{Arc, LazyLock},
    time::{Duration, Instant},
};

use litellm_redis_store::{
    Number,
    rate_limits::{EnqueuedReply, Local, RateLimits},
};
use serde::{Deserialize, Serialize};

use crate::{
    engine::Now,
    failure::{Failure, redis_failure},
};

pub const ENQUEUED_TTL_SECONDS: i64 = 8 * 24 * 60 * 60;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnqueuedScope {
    pub key: String,
    pub value: String,
    pub limit: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReservationBackend {
    #[default]
    Redis,
    Memory,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnqueuedReservation {
    pub tokens: i64,
    pub scopes: Vec<EnqueuedScope>,
    #[serde(default)]
    pub backend: ReservationBackend,
    #[serde(default)]
    pub owner: String,
    #[serde(default)]
    pub reserved_at_monotonic: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnqueuedOutcome {
    Reservation(EnqueuedReservation),
    OverLimit { scope: EnqueuedScope, enqueued: i64 },
}

static STARTED: LazyLock<Instant> = LazyLock::new(Instant::now);

pub fn process_monotonic() -> Now {
    Arc::new(|| STARTED.elapsed().as_secs_f64())
}

fn counter_key(scope: &EnqueuedScope) -> String {
    format!("batch_enqueued_tokens:{}:{}", scope.key, scope.value)
}

fn record_key(batch_id: &str) -> String {
    format!("batch_enqueued_token_reservation:{batch_id}")
}

fn ttl() -> Duration {
    Duration::from_secs(ENQUEUED_TTL_SECONDS.unsigned_abs())
}

pub struct EnqueuedTokens {
    store: Arc<RateLimits>,
    owner: String,
    monotonic: Now,
}

impl EnqueuedTokens {
    pub fn new(store: Arc<RateLimits>, monotonic: Now) -> Self {
        Self {
            store,
            owner: uuid::Uuid::new_v4().simple().to_string(),
            monotonic,
        }
    }

    fn now(&self) -> f64 {
        (self.monotonic)()
    }

    pub async fn reserve(&self, tokens: i64, scopes: Vec<EnqueuedScope>) -> EnqueuedOutcome {
        if tokens <= 0 || scopes.is_empty() {
            return EnqueuedOutcome::Reservation(EnqueuedReservation {
                tokens: tokens.max(0),
                scopes,
                backend: ReservationBackend::Redis,
                owner: String::new(),
                reserved_at_monotonic: Some(self.now()),
            });
        }
        if self.store.has_redis() {
            match self.reserve_via_redis(tokens, &scopes).await {
                Ok(outcome) => return outcome,
                Err(error) => redis_failure!(
                    &error,
                    "Redis enqueued-token reserve failed, falling back to in-memory"
                ),
            }
        }
        self.reserve_in_memory(tokens, scopes)
    }

    async fn reserve_via_redis(
        &self,
        tokens: i64,
        scopes: &[EnqueuedScope],
    ) -> Result<EnqueuedOutcome, Failure> {
        let started = self.now();
        for (index, scope) in scopes.iter().enumerate() {
            let reply = match self.store.enqueued_reserve(
                &counter_key(scope),
                tokens,
                ENQUEUED_TTL_SECONDS,
                scope.limit,
            ) {
                Some(pending) => pending.await.map_err(Failure::from),
                None => Err(Failure::raised(
                    "RuntimeError",
                    "no Redis client for the enqueued-token store",
                )),
            };
            let reply: EnqueuedReply = match reply {
                Ok(reply) => reply,
                Err(error) => {
                    self.rollback(tokens, &scopes[..index]).await;
                    return Err(error);
                }
            };
            if !reply.reserved {
                self.rollback(tokens, &scopes[..index]).await;
                return Ok(EnqueuedOutcome::OverLimit {
                    scope: scope.clone(),
                    enqueued: reply.count,
                });
            }
        }
        Ok(EnqueuedOutcome::Reservation(EnqueuedReservation {
            tokens,
            scopes: scopes.to_vec(),
            backend: ReservationBackend::Redis,
            owner: String::new(),
            reserved_at_monotonic: Some(started),
        }))
    }

    async fn rollback(&self, tokens: i64, scopes: &[EnqueuedScope]) {
        if let Err(error) = self.refund_via_redis(tokens, scopes).await {
            tracing::warn!(
                "Rollback of partially reserved enqueued tokens failed; leaked increments expire with the TTL: {error}"
            );
        }
    }

    async fn refund_via_redis(&self, tokens: i64, scopes: &[EnqueuedScope]) -> Result<(), Failure> {
        for scope in scopes {
            match self.store.enqueued_refund(&counter_key(scope), tokens) {
                Some(pending) => pending.await.map_err(Failure::from)?,
                None => {
                    return Err(Failure::raised(
                        "RuntimeError",
                        "no Redis client for the enqueued-token store",
                    ));
                }
            }
        }
        Ok(())
    }

    fn local_counter(&self, scope: &EnqueuedScope) -> i64 {
        match self.store.get_raw_local(&counter_key(scope)) {
            Ok(Some(Local::Number(Number::Int(value)))) => value,
            Ok(Some(Local::Number(Number::Float(value)))) => value.trunc() as i64,
            Ok(_) => 0,
            Err(error) => {
                tracing::warn!(%error, "enqueued-token memory read failed");
                0
            }
        }
    }

    fn set_local_counter(&self, scope: &EnqueuedScope, value: i64) {
        if let Err(error) = self.store.set_raw_local(
            &counter_key(scope),
            Local::Number(Number::Int(value)),
            ttl(),
        ) {
            tracing::warn!(%error, "enqueued-token memory write failed");
        }
    }

    fn reserve_in_memory(&self, tokens: i64, scopes: Vec<EnqueuedScope>) -> EnqueuedOutcome {
        let started = self.now();
        {
            let _guard = self.store.enqueued_guard();
            let currents: Vec<i64> = scopes
                .iter()
                .map(|scope| self.local_counter(scope))
                .collect();
            if let Some((scope, current)) = scopes
                .iter()
                .zip(&currents)
                .find(|(scope, current)| **current + tokens > scope.limit)
            {
                return EnqueuedOutcome::OverLimit {
                    scope: scope.clone(),
                    enqueued: *current,
                };
            }
            for (scope, current) in scopes.iter().zip(&currents) {
                self.set_local_counter(scope, current + tokens);
            }
        }
        EnqueuedOutcome::Reservation(EnqueuedReservation {
            tokens,
            scopes,
            backend: ReservationBackend::Memory,
            owner: self.owner.clone(),
            reserved_at_monotonic: Some(started),
        })
    }

    pub async fn refund(&self, reservation: &EnqueuedReservation) {
        if reservation.tokens <= 0 || reservation.scopes.is_empty() {
            return;
        }
        if reservation.backend == ReservationBackend::Redis {
            if !self.store.has_redis() {
                tracing::warn!(
                    "No Redis client for a Redis-granted enqueued-token refund; leaked increments expire with the TTL"
                );
                return;
            }
            if let Err(error) = self
                .refund_via_redis(reservation.tokens, &reservation.scopes)
                .await
            {
                tracing::warn!(
                    "Redis enqueued-token refund failed; leaked increments expire with the TTL: {error}"
                );
            }
            return;
        }
        if reservation.owner != self.owner {
            tracing::warn!(
                "Skipping enqueued-token refund granted in another worker's memory; its counters expire with the TTL"
            );
            return;
        }
        let _guard = self.store.enqueued_guard();
        for scope in &reservation.scopes {
            let remaining = self.local_counter(scope) - reservation.tokens;
            if remaining <= 0 {
                if let Err(error) = self.store.delete_raw_local(&counter_key(scope)) {
                    tracing::warn!(%error, "enqueued-token memory delete failed");
                }
            } else {
                self.set_local_counter(scope, remaining);
            }
        }
    }

    pub async fn save(&self, batch_id: &str, reservation: &EnqueuedReservation) {
        let reserved_at = reservation
            .reserved_at_monotonic
            .unwrap_or_else(|| self.now());
        let serialized = serialized(reservation, reserved_at);
        let elapsed = self.now() - reserved_at;
        let ttl_seconds = (ENQUEUED_TTL_SECONDS - elapsed.ceil() as i64).max(1);
        let key = record_key(batch_id);
        if let Some(pending) = self
            .store
            .enqueued_record_save(&key, &serialized, ttl_seconds)
        {
            match pending.await {
                Ok(()) => return,
                Err(error) => redis_failure!(
                    &Failure::from(error),
                    "Redis enqueued-token reservation save failed, falling back to in-memory"
                ),
            }
        }
        if let Err(error) = self.store.set_raw_local(
            &key,
            Local::Text(serialized),
            Duration::from_secs(ttl_seconds.unsigned_abs()),
        ) {
            tracing::warn!(%error, "enqueued-token memory write failed");
        }
    }

    pub async fn pop(&self, batch_id: &str) -> Option<EnqueuedReservation> {
        let key = record_key(batch_id);
        let remote = match self.store.enqueued_record_pop(&key, ENQUEUED_TTL_SECONDS) {
            Some(pending) => pending.await.unwrap_or_else(|error| {
                redis_failure!(
                    &Failure::from(error),
                    "Redis enqueued-token reservation pop failed, falling back to in-memory"
                );
                None
            }),
            None => None,
        };
        let raw = match remote {
            Some(text) if text.is_empty() => {
                self.pop_local(&key);
                return None;
            }
            Some(text) => Some(Local::Text(text)),
            None => self.pop_local(&key),
        }?;
        let parsed = match raw {
            Local::Text(text) => serde_json::from_str::<EnqueuedReservation>(&text).ok(),
            Local::Number(_) | Local::Slots(_) => None,
        };
        match parsed {
            Some(mut reservation) => {
                if reservation.reserved_at_monotonic.is_none() {
                    reservation.reserved_at_monotonic = Some(self.now());
                }
                Some(reservation)
            }
            None => {
                tracing::warn!(
                    batch_id,
                    "Discarding malformed enqueued-token reservation record"
                );
                None
            }
        }
    }

    fn pop_local(&self, key: &str) -> Option<Local> {
        let _guard = self.store.enqueued_guard();
        let stored = self.store.get_raw_local(key).ok().flatten()?;
        if let Err(error) = self.store.delete_raw_local(key) {
            tracing::warn!(%error, "enqueued-token memory delete failed");
        }
        Some(stored)
    }
}

pub fn serialized(reservation: &EnqueuedReservation, reserved_at: f64) -> String {
    serde_json::to_string(&EnqueuedReservation {
        reserved_at_monotonic: Some(reserved_at),
        ..reservation.clone()
    })
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn tokens() -> EnqueuedTokens {
        EnqueuedTokens::new(Arc::new(RateLimits::new(None, None)), Arc::new(|| 100.0))
    }

    fn scope(key: &str, limit: i64) -> EnqueuedScope {
        EnqueuedScope {
            key: key.into(),
            value: "v".into(),
            limit,
        }
    }

    #[rstest]
    #[tokio::test]
    async fn memory_reservations_are_all_or_nothing_and_refund_to_zero() {
        let store = tokens();
        let both = vec![scope("api_key", 100), scope("team", 50)];
        let first = store.reserve(40, both.clone()).await;
        let EnqueuedOutcome::Reservation(granted) = first else {
            panic!("the first reservation fits");
        };
        assert_eq!(
            store.reserve(20, both.clone()).await,
            EnqueuedOutcome::OverLimit {
                scope: scope("team", 50),
                enqueued: 40,
            }
        );
        assert_eq!(store.local_counter(&both[0]), 40);
        store.refund(&granted).await;
        assert_eq!(store.local_counter(&both[0]), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn a_record_pops_once_and_serializes_like_pydantic() {
        let store = tokens();
        let EnqueuedOutcome::Reservation(granted) =
            store.reserve(10, vec![scope("api_key", 100)]).await
        else {
            panic!("fits");
        };
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&serialized(&granted, 100.0)).unwrap(),
            json!({"tokens": 10, "scopes": [{"key": "api_key", "value": "v", "limit": 100}],
                   "backend": "memory", "owner": store.owner, "reserved_at_monotonic": 100.0})
        );
        store.save("b", &granted).await;
        assert_eq!(store.pop("b").await, Some(granted));
        assert_eq!(store.pop("b").await, None);
    }

    #[rstest]
    #[tokio::test]
    async fn another_workers_memory_grant_is_not_refunded() {
        let store = tokens();
        let EnqueuedOutcome::Reservation(mut granted) =
            store.reserve(10, vec![scope("api_key", 100)]).await
        else {
            panic!("fits");
        };
        granted.owner = "other".into();
        store.refund(&granted).await;
        assert_eq!(store.local_counter(&scope("api_key", 100)), 10);
    }

    #[rstest]
    fn a_legacy_record_defaults_to_redis() {
        let parsed: EnqueuedReservation =
            serde_json::from_value(json!({"tokens": 5, "scopes": []})).unwrap();
        assert_eq!(parsed.backend, ReservationBackend::Redis);
    }
}
