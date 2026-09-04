use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const RECENT_WINDOW_SECS: u64 = 30;

static GIL_ACQUISITIONS: AtomicU64 = AtomicU64::new(0);
static LAST_GIL_UNIX_SECS: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, PartialEq, Eq)]
pub struct GilSnapshot {
    pub total_acquisitions: u64,
    pub seconds_since_last: Option<u64>,
    pub acquired_last_30s: bool,
}

pub fn snapshot() -> GilSnapshot {
    let total_acquisitions = GIL_ACQUISITIONS.load(Ordering::Relaxed);
    let last_acquisition = LAST_GIL_UNIX_SECS.load(Ordering::Relaxed);
    let seconds_since_last =
        (last_acquisition != 0).then(|| now_unix_secs().saturating_sub(last_acquisition));

    GilSnapshot {
        total_acquisitions,
        seconds_since_last,
        acquired_last_30s: seconds_since_last.is_some_and(|seconds| seconds <= RECENT_WINDOW_SECS),
    }
}

#[cfg(feature = "python")]
pub(crate) fn record_acquisition() {
    GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
    LAST_GIL_UNIX_SECS.store(now_unix_secs(), Ordering::Relaxed);
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
