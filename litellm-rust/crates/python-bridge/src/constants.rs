pub(crate) const FUNCTION_TRACE_TARGET: &str = "litellm::function_trace";

/// Maximum concurrent in-flight provider calls process-wide; unset = unlimited.
pub(crate) const MAX_IN_FLIGHT_ENV: &str = "LITELLM_RUST_MAX_IN_FLIGHT";
/// When truthy, over-limit calls raise `RustBridgeDeclined` instead of queueing.
pub(crate) const SHED_ON_LIMIT_ENV: &str = "LITELLM_RUST_SHED_ON_LIMIT";
/// Worker threads for the shared Tokio runtime; unset = CPU count. Must be
/// applied at module init — the runtime is built lazily on first use and
/// `pyo3_async_runtimes::tokio::init` is a silent no-op afterwards.
pub(crate) const WORKER_THREADS_ENV: &str = "LITELLM_RUST_WORKER_THREADS";
