use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::prelude::*;

static GIL_ACQUISITIONS: AtomicU64 = AtomicU64::new(0);
static GIL_RELEASES: AtomicU64 = AtomicU64::new(0);

/// Runs work attached to the interpreter and records the acquisition.
pub fn attach<T, F>(f: F) -> T
where
    F: for<'py> FnOnce(Python<'py>) -> T,
{
    GIL_ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
    Python::attach(f)
}

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

pub fn acquisition_count() -> u64 {
    GIL_ACQUISITIONS.load(Ordering::Relaxed)
}

pub fn release_count() -> u64 {
    GIL_RELEASES.load(Ordering::Relaxed)
}
