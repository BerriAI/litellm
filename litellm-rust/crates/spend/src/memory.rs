use std::{collections::BTreeMap, convert::Infallible, sync::Mutex, time::Duration};

use tokio::time::Instant;

use crate::{Batch, BatchId, Buffer, Claimed, Key, Store, Tally};

#[derive(Debug)]
struct InFlight<K, V> {
    batch: Batch<K, V>,
    visible_at: Instant,
}

#[derive(Debug)]
struct State<K, V> {
    pending: Batch<K, V>,
    in_flight: BTreeMap<BatchId, InFlight<K, V>>,
    applied: BTreeMap<BatchId, Instant>,
}

#[derive(Debug)]
pub struct MemoryBuffer<K, V> {
    claim_ttl: Duration,
    remember_applied_for: Duration,
    state: Mutex<State<K, V>>,
}

impl<K: Key, V: Tally> MemoryBuffer<K, V> {
    pub fn new(claim_ttl: Duration, remember_applied_for: Duration) -> Self {
        Self {
            claim_ttl,
            remember_applied_for,
            state: Mutex::new(State {
                pending: Batch::default(),
                in_flight: BTreeMap::new(),
                applied: BTreeMap::new(),
            }),
        }
    }

    fn with_state<T>(&self, f: impl FnOnce(&mut State<K, V>) -> T) -> T {
        f(&mut self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()))
    }

    fn redeliver(&self, state: &mut State<K, V>, now: Instant) -> Option<Claimed<K, V>> {
        let (id, flight) = state
            .in_flight
            .iter_mut()
            .filter(|(_, flight)| flight.visible_at <= now)
            .min_by_key(|(_, flight)| flight.visible_at)?;
        flight.visible_at = now + self.claim_ttl;
        Some(Claimed::from_buffer(*id, flight.batch.clone()))
    }

    fn freeze(&self, state: &mut State<K, V>, max: usize, now: Instant) -> Option<Claimed<K, V>> {
        let (batch, kept) = std::mem::take(&mut state.pending).split_at(max);
        state.pending = kept;
        if batch.is_empty() {
            return None;
        }
        let id = BatchId::random();
        state.in_flight.insert(
            id,
            InFlight {
                batch: batch.clone(),
                visible_at: now + self.claim_ttl,
            },
        );
        Some(Claimed::from_buffer(id, batch))
    }
}

impl<K: Key, V: Tally> Buffer<K, V> for MemoryBuffer<K, V> {
    type Error = Infallible;

    async fn push(&self, batch: Batch<K, V>) -> Result<(), Self::Error> {
        self.with_state(|state| {
            state.pending = std::mem::take(&mut state.pending).merge(batch);
        });
        Ok(())
    }

    async fn claim(&self, max: usize) -> Result<Option<Claimed<K, V>>, Self::Error> {
        let now = Instant::now();
        Ok(self.with_state(|state| {
            self.redeliver(state, now)
                .or_else(|| self.freeze(state, max, now))
        }))
    }

    async fn ack(&self, claimed: Claimed<K, V>) -> Result<(), Self::Error> {
        self.with_state(|state| state.in_flight.remove(&claimed.id()));
        Ok(())
    }

    async fn release(&self, claimed: Claimed<K, V>) -> Result<(), Self::Error> {
        let now = Instant::now();
        self.with_state(|state| {
            if let Some(flight) = state.in_flight.get_mut(&claimed.id()) {
                flight.visible_at = now;
            }
        });
        Ok(())
    }
}

impl<K: Key, V: Tally> Store<K, V> for MemoryBuffer<K, V> {
    type Error = Infallible;

    async fn commit(&self, claimed: &Claimed<K, V>) -> Result<(), Self::Error> {
        let now = Instant::now();
        self.with_state(|state| {
            state.applied.retain(|_, applied_at| {
                now.duration_since(*applied_at) < self.remember_applied_for
            });
            if state.applied.insert(claimed.id(), now).is_some() {
                return;
            }
            state.pending = std::mem::take(&mut state.pending).merge(claimed.batch().clone());
        });
        Ok(())
    }
}
