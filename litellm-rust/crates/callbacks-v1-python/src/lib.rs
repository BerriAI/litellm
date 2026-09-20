//! The Python adapter of the v1 callback contract: the per-call subscriber snapshot and
//! one [`PythonLifecycle`](litellm_host_python::PythonLifecycle) that marshals
//! `litellm-callbacks-v1` envelopes and patches to Python handlers. The contract itself
//! lives in `litellm-callbacks-v1`.

mod adapter;
mod call;
mod python;
mod subscribers;

pub use adapter::V1PythonLifecycle;
pub use call::V1PythonSurface;

#[cfg(test)]
#[path = "../tests/lifecycle.rs"]
mod lifecycle_tests;

#[cfg(test)]
#[path = "../tests/support.rs"]
mod test_support;
