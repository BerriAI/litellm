use std::num::NonZeroUsize;

use litellm_core::provider_callbacks::{
    CallbackDecision, ProviderAttemptObserver, ProviderError, ProviderPostCall, ProviderPreCall,
};
use litellm_python_interop::callback_runtime::{AsyncContext, CallbackRuntime, SyncContext};
use pyo3::prelude::*;

use crate::constants::OCR_CALLBACK_CAPACITY;
use crate::execution::PythonCallContext;

litellm_core::provider_attempt_observer_catalog!(crate::bind_python_hooks,
    pub(crate) struct PythonProviderSession;
    trait ProviderAttemptObserver;
);
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

pub(crate) enum PythonProviderObserver {
    Disabled,
    Sync(PythonProviderSession<SyncContext>),
    Async(PythonProviderSession<AsyncContext>),
}

impl PythonProviderObserver {
    pub(crate) fn new(
        adapter: Option<Py<PyAny>>,
        context: PythonCallContext<'_>,
    ) -> PyResult<Self> {
        let Some(adapter) = adapter else {
            return Ok(Self::Disabled);
        };
        let py = context.py;
        let module = py.import("litellm.rust_bridge._native")?;
        let runtime = module
            .getattr("__python_callback_runtime__")?
            .extract::<PyRef<'_, PythonCallbackRuntime>>()?
            .0
            .clone();
        if context.asynchronous {
            Ok(Self::Async(PythonProviderSession::new(
                adapter.bind(py),
                runtime.async_context(py)?,
            )?))
        } else {
            Ok(Self::Sync(PythonProviderSession::new(
                adapter.bind(py),
                runtime.sync_context(py)?,
            )?))
        }
    }
}

impl ProviderAttemptObserver for PythonProviderObserver {
    type Error = PyErr;

    async fn pre_call(&mut self, input: &ProviderPreCall) -> PyResult<CallbackDecision> {
        match self {
            Self::Disabled => Ok(CallbackDecision::Unchanged),
            Self::Sync(session) => session.pre_call(input).await,
            Self::Async(session) => session.pre_call(input).await,
        }
    }

    async fn post_call(&mut self, input: &ProviderPostCall) -> PyResult<CallbackDecision> {
        match self {
            Self::Disabled => Ok(CallbackDecision::Unchanged),
            Self::Sync(session) => session.post_call(input).await,
            Self::Async(session) => session.post_call(input).await,
        }
    }

    async fn error(&mut self, input: &ProviderError) -> PyResult<()> {
        match self {
            Self::Disabled => Ok(()),
            Self::Sync(session) => session.error(input).await,
            Self::Async(session) => session.error(input).await,
        }
    }
}
