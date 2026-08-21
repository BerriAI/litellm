//! GIL-activity tracking.
//!
//! Every acquisition of the Python GIL is recorded here so the `/health/gil`
//! endpoint can report whether Python was touched recently. The design goal is
//! that the GIL is acquired **only at load time** (config read) and never on the
//! realtime hot path — polling this endpoint during traffic should show the
//! count holding steady and `acquired_last_30s` falling to `false`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Window (seconds) for the "recently acquired" signal.
pub const RECENT_WINDOW_SECS: u64 = 30;

static GIL_ACQUISITIONS: AtomicU64 = AtomicU64::new(0);
/// Unix seconds of the last acquisition; `0` means "never".
static LAST_GIL_UNIX_SECS: AtomicU64 = AtomicU64::new(0);

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Record that the GIL was just acquired. Call immediately before taking the GIL.
///
/// Only invoked under the `python-config` feature; without it the gateway never
/// touches Python, so the recorder is unused (and the endpoint reports zero).
#[cfg_attr(not(feature = "python-config"), allow(dead_code))]
pub fn record_acquisition() {
    GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
    LAST_GIL_UNIX_SECS.store(now_unix_secs(), Ordering::Relaxed);
}

/// Point-in-time view of GIL activity.
pub struct GilSnapshot {
    pub total_acquisitions: u64,
    pub seconds_since_last: Option<u64>,
    pub acquired_last_30s: bool,
}

/// Read the current GIL-activity snapshot.
pub fn snapshot() -> GilSnapshot {
    let total = GIL_ACQUISITIONS.load(Ordering::Relaxed);
    let last = LAST_GIL_UNIX_SECS.load(Ordering::Relaxed);
    let seconds_since_last = if last == 0 {
        None
    } else {
        Some(now_unix_secs().saturating_sub(last))
    };
    let acquired_last_30s = seconds_since_last.is_some_and(|secs| secs <= RECENT_WINDOW_SECS);
    GilSnapshot {
        total_acquisitions: total,
        seconds_since_last,
        acquired_last_30s,
    }
}
