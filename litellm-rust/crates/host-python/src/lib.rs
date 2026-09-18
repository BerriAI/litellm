//! The CPython runtime adapter: value marshalling, interpreter detachment, the tokio and
//! asyncio glue, and the driver that runs a native [`Machine`](litellm_callbacks::machine::Machine)
//! against a Python route host and a Python lifecycle. Everything here is Python-specific by
//! construction; another host language gets its own crate of the same shape.

mod adapter;
mod argument;
mod callable;
mod driver;
mod execution;
mod gil;
mod handle;
mod marshal;

pub use adapter::{
    HostOpError, LifecycleEvent, LifecycleStep, PythonLifecycle, RouteHost, missing_state,
};
pub use argument::lookup;
pub use callable::wrap_failure;
pub use driver::run_call;
pub use execution::{poll_async_value, run_async, run_async_value, run_sync, run_sync_value};
pub use gil::{release_count, release_gil};
pub use handle::{Execution, ExecutionBody, ExecutionStep};
pub use marshal::{Pythonized, from_py, from_py_argument, panic_to_pyerr, to_py};

/// Starts the interpreter and imports the standard modules the tests share, once, so
/// parallel test threads never race a first import of `asyncio`.
#[cfg(test)]
pub(crate) fn initialize_python() {
    static IMPORTED: std::sync::Once = std::sync::Once::new();
    pyo3::Python::initialize();
    IMPORTED.call_once(|| {
        pyo3::Python::attach(|py| {
            py.import("asyncio").expect("asyncio imports");
        });
    });
}
