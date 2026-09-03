use std::num::NonZeroUsize;

use litellm_core::ocr::observers::{OcrObserver, OcrPostCall, OcrPreCall};
use litellm_python_interop::callback_runtime::{AsyncContext, CallbackRuntime, SyncContext};
use pyo3::prelude::*;

use crate::constants::OCR_CALLBACK_CAPACITY;
use crate::execution::PythonCallContext;

litellm_core::ocr_observer_catalog!(crate::bind_python_hooks,
    pub(crate) struct PythonOcrSession;
    trait OcrObserver;
);

#[pyclass(frozen)]
struct OcrRuntime(CallbackRuntime);

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let capacity = NonZeroUsize::new(OCR_CALLBACK_CAPACITY)
        .expect("OCR callback capacity is a positive constant");
    module.add(
        "__ocr_callback_runtime__",
        OcrRuntime(CallbackRuntime::new(module, capacity)?),
    )
}

pub(crate) enum PythonOcrObserver {
    Disabled,
    Sync(PythonOcrSession<SyncContext>),
    Async(PythonOcrSession<AsyncContext>),
}

impl PythonOcrObserver {
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
            .getattr("__ocr_callback_runtime__")?
            .extract::<PyRef<'_, OcrRuntime>>()?
            .0
            .clone();
        if context.asynchronous {
            Ok(Self::Async(PythonOcrSession::new(
                adapter.bind(py),
                runtime.async_context(py)?,
            )?))
        } else {
            Ok(Self::Sync(PythonOcrSession::new(
                adapter.bind(py),
                runtime.sync_context(py)?,
            )?))
        }
    }
}

impl OcrObserver for PythonOcrObserver {
    type Error = PyErr;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn pre_call(&mut self, input: &OcrPreCall) -> PyResult<()> {
        match self {
            Self::Disabled => Ok(()),
            Self::Sync(session) => session.pre_call(input).await,
            Self::Async(session) => session.pre_call(input).await,
        }
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn post_call(&mut self, input: &OcrPostCall) -> PyResult<()> {
        match self {
            Self::Disabled => Ok(()),
            Self::Sync(session) => session.post_call(input).await,
            Self::Async(session) => session.post_call(input).await,
        }
    }
}
