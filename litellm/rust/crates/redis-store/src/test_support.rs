use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use crate::Clock;

pub(crate) const START: u64 = 1_000;

/// A clock at `START` seconds that a test moves by adding to the counter.
pub(crate) fn clock() -> (Clock, Arc<AtomicU64>) {
    let seconds = Arc::new(AtomicU64::new(START));
    let read = seconds.clone();
    let clock: Clock = Arc::new(move || Duration::from_secs(read.load(Ordering::SeqCst)));
    (clock, seconds)
}
