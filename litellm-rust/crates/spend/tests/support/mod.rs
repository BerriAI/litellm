#![allow(
    dead_code,
    reason = "each test binary uses a different part of the shared support"
)]

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use litellm_spend::{BatchId, Buffer, Claimed, Cost, EntityKey, MemoryBuffer, Store};
use litellm_spend_testing::Ledger;

pub const CLAIM_TTL: Duration = Duration::from_secs(30);

pub fn buffer() -> MemoryBuffer<EntityKey, Cost> {
    MemoryBuffer::new(CLAIM_TTL, Duration::from_secs(3_600))
}

#[derive(Debug, thiserror::Error)]
#[error("injected failure")]
pub struct Injected;

#[derive(Clone, Default)]
pub struct Faults {
    pub latency: Duration,
    pub fail_without_applying_every: Option<usize>,
    pub fail_after_applying_every: Option<usize>,
    pub poison: Arc<Mutex<Option<EntityKey>>>,
}

#[derive(Default)]
struct Applied {
    totals: HashMap<EntityKey, f64>,
    batches: HashSet<BatchId>,
}

#[derive(Default)]
pub struct Database {
    faults: Faults,
    applied: Mutex<Applied>,
    attempts: AtomicUsize,
    in_flight: Arc<AtomicUsize>,
    pub max_in_flight: AtomicUsize,
    pub failed_without_applying: AtomicUsize,
    pub failed_after_applying: AtomicUsize,
    pub repeats_skipped: AtomicUsize,
}

struct InFlight(Arc<AtomicUsize>);

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn due(every: Option<usize>, attempt: usize) -> bool {
    every.is_some_and(|every| attempt.is_multiple_of(every))
}

impl Database {
    pub fn new(faults: Faults) -> Self {
        Self {
            faults,
            ..Self::default()
        }
    }

    pub fn totals(&self) -> HashMap<EntityKey, f64> {
        self.applied.lock().unwrap().totals.clone()
    }

    fn enter(&self) -> InFlight {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        InFlight(self.in_flight.clone())
    }

    fn apply(&self, claimed: &Claimed<EntityKey, Cost>) {
        let mut applied = self.applied.lock().unwrap();
        if !applied.batches.insert(claimed.id()) {
            self.repeats_skipped.fetch_add(1, Ordering::SeqCst);
            return;
        }
        for (entity, cost) in claimed.batch().iter() {
            *applied.totals.entry(entity.clone()).or_default() += cost.0;
        }
    }
}

impl Store<EntityKey, Cost> for Database {
    type Error = Injected;

    async fn commit(&self, claimed: &Claimed<EntityKey, Cost>) -> Result<(), Injected> {
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        let _in_flight = self.enter();
        let poison = self.faults.poison.lock().unwrap().clone();
        if poison.is_some_and(|poison| claimed.batch().get(&poison).is_some()) {
            return Err(Injected);
        }
        if due(self.faults.fail_without_applying_every, attempt) {
            tokio::time::sleep(self.faults.latency).await;
            self.failed_without_applying.fetch_add(1, Ordering::SeqCst);
            return Err(Injected);
        }
        self.apply(claimed);
        tokio::time::sleep(self.faults.latency).await;
        if due(self.faults.fail_after_applying_every, attempt) {
            self.failed_after_applying.fetch_add(1, Ordering::SeqCst);
            return Err(Injected);
        }
        Ok(())
    }
}

impl Ledger for Database {
    async fn total(&self, entity: &EntityKey) -> Option<Cost> {
        self.totals().get(entity).copied().map(Cost)
    }
}

pub struct BufferLedger(pub MemoryBuffer<EntityKey, Cost>);

impl Store<EntityKey, Cost> for BufferLedger {
    type Error = std::convert::Infallible;

    async fn commit(&self, claimed: &Claimed<EntityKey, Cost>) -> Result<(), Self::Error> {
        self.0.commit(claimed).await
    }
}

impl Ledger for BufferLedger {
    async fn total(&self, entity: &EntityKey) -> Option<Cost> {
        let mut held = Vec::new();
        while let Some(claimed) = self.0.claim(usize::MAX).await.unwrap() {
            held.push(claimed);
        }
        let costs: Vec<_> = held
            .iter()
            .filter_map(|claimed| claimed.batch().get(entity).copied())
            .collect();
        for claimed in held {
            self.0.release(claimed).await.unwrap();
        }
        (!costs.is_empty()).then(|| Cost(costs.iter().map(|cost| cost.0).sum()))
    }
}
