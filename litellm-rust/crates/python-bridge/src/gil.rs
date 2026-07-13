//! GIL accounting.
//!
//! A single chokepoint for releasing the GIL around blocking work. Every
//! blocking call in the bridge goes through [`release_gil`] instead of calling
//! `Python::detach` directly, so the release count stays accurate and we
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
    py.detach(f)
}

/// Total GIL releases performed by the bridge so far.
pub fn release_count() -> u64 {
    GIL_RELEASES.load(Ordering::Relaxed)
}
