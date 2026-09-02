use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::prelude::*;

static GIL_RELEASES: AtomicU64 = AtomicU64::new(0);

/// Runs work detached from the interpreter and records the release.
///
/// `f` must not access Python state while the interpreter is detached.
pub fn release_gil<T, F>(py: Python<'_>, f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    GIL_RELEASES.fetch_add(1, Ordering::Relaxed);
    py.detach(f)
}

pub fn release_count() -> u64 {
    GIL_RELEASES.load(Ordering::Relaxed)
}
