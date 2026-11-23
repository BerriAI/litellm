use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use rstest::fixture;

use crate::InMemoryCache;

pub(crate) type Clock = Arc<AtomicU64>;

#[fixture]
pub(crate) fn clock() -> Clock {
    Arc::new(AtomicU64::new(100))
}

pub(crate) fn cache<V: Clone>(clock: &Clock, capacity: usize) -> InMemoryCache<V> {
    let clock = clock.clone();
    InMemoryCache::with_clock(Some(capacity), Some(Duration::from_secs(60)), move || {
        Duration::from_secs(clock.load(Ordering::SeqCst))
    })
}

pub(crate) fn at(clock: &Clock, seconds: u64) {
    clock.store(seconds, Ordering::SeqCst);
}

pub(crate) fn secs(seconds: u64) -> Option<Duration> {
    Some(Duration::from_secs(seconds))
}
