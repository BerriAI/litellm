//! GIL accounting.
//!
//! A single chokepoint for releasing the GIL around blocking work. Every
//! blocking call in the bridge goes through [`release_gil`] instead of calling
//! `Python::allow_threads` directly, so the release count stays accurate and we
//! have one place to extend later (timing histograms, per-call labels, etc.).

use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::prelude::*;

/// Number of times the bridge has released the GIL since process start.
static GIL_RELEASES: AtomicU64 = AtomicU64::new(0);

/// Release the GIL around `f`, recording the release.
///
/// `f` must not touch any Python state — that is what makes releasing the GIL
/// safe. Returning the value back to Python re-acquires the GIL at the call
/// site, after `f` has finished.
pub fn release_gil<T, F>(py: Python<'_>, f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    GIL_RELEASES.fetch_add(1, Ordering::Relaxed);
    py.allow_threads(f)
}

/// Record an off-GIL offload without wrapping a closure.
///
/// The async `aocr` path hands work to the Tokio runtime, which runs without the
/// GIL by construction, so there is no `allow_threads` closure to wrap — this
/// keeps the release count accurate across both entry points.
pub fn note_offload() {
    GIL_RELEASES.fetch_add(1, Ordering::Relaxed);
}

/// Total GIL releases performed by the bridge so far.
pub fn release_count() -> u64 {
    GIL_RELEASES.load(Ordering::Relaxed)
}
