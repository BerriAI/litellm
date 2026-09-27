//! The CPython runtime adapter: value marshalling, interpreter detachment, the tokio and
//! asyncio glue, and the driver that runs a native [`Machine`](litellm_host::machine::Machine)
//! against a Python protocol host and a Python lifecycle. Everything here is Python-specific by
//! construction; another host language gets its own crate of the same shape.

mod adapter;
mod argument;
mod callable;
mod driver;
mod execution;
mod file_reader;
mod fork_gate;
mod gil;
mod handle;
mod marshal;

pub use adapter::{
    InvokeError, LifecycleEvent, LifecycleStep, Preflight, ProtocolHost, PythonLifecycle,
    missing_state,
};
pub use argument::{json_fields, lookup};
pub use callable::wrap_failure;
pub use driver::run_call;
pub use execution::{
    ForkedAfterNativeRuntimeStarted, ProcessReservedForForking, enter_native, poll_async_value,
    reserve_process_for_forking, run_async, run_async_value, run_sync, run_sync_value,
    runtime_started,
};
pub use file_reader::{FileContent, PythonFileReader, py_bytes};
pub use fork_gate::RuntimeAlreadyStarted;
pub use gil::{PythonContext, attach_blocking, release_count, release_gil};
pub use handle::{Execution, ExecutionBody, ExecutionStep};
pub use marshal::{
    Pythonized, from_py, from_py_argument, json_loads, json_object_field, panic_to_pyerr, to_py,
};

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

#[cfg(test)]
pub(crate) struct InitializedPython;

#[cfg(test)]
impl InitializedPython {
    pub(crate) fn attach<F, R>(&self, f: F) -> R
    where
        F: for<'py> FnOnce(pyo3::Python<'py>) -> R,
    {
        pyo3::Python::attach(f)
    }
}

#[cfg(test)]
#[rstest::fixture]
#[once]
pub(crate) fn initialized_python() -> InitializedPython {
    initialize_python();
    InitializedPython
}
