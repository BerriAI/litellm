use std::num::NonZeroUsize;

use litellm_python_interop::callback_runtime::{AsyncContext, CallbackRuntime, SyncContext};
use pyo3::prelude::*;

use crate::constants::OCR_CALLBACK_CAPACITY;
litellm_core::streaming_observer_catalog!(crate::bind_python_hooks,
    pub struct PythonStreamingSession;
    trait litellm_core::provider_callbacks::StreamingObserver;
);
litellm_core::session_observer_catalog!(crate::bind_python_hooks,
    pub struct PythonSession;
    trait litellm_core::provider_callbacks::SessionObserver;
);

#[pyclass(frozen)]
struct PythonCallbackRuntime(CallbackRuntime);

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _streaming_constructor = PythonStreamingSession::<SyncContext>::new;
    let _session_constructor = PythonSession::<AsyncContext>::new;
    let capacity = NonZeroUsize::new(OCR_CALLBACK_CAPACITY)
        .expect("Python callback capacity is a positive constant");
    module.add(
        "__python_callback_runtime__",
        PythonCallbackRuntime(CallbackRuntime::new(module, capacity)?),
    )
}

pub(crate) fn python_async_session(
    adapter: Py<PyAny>,
    py: Python<'_>,
) -> PyResult<PythonSession<AsyncContext>> {
    let module = py.import("litellm.rust_bridge._native")?;
    let runtime = module
        .getattr("__python_callback_runtime__")?
        .extract::<PyRef<'_, PythonCallbackRuntime>>()?
        .0
        .clone();
    PythonSession::new(adapter.bind(py), runtime.async_context(py)?)
}
