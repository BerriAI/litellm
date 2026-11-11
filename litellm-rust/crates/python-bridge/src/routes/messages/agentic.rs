use litellm_host::{
    hooks::CallHooks,
    interceptors::{RequestContext, WireRequest},
    lifecycle::Timing,
};
use litellm_host_python::{HookStep, PythonOwned, PythonRuntime, to_py};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use super::host::ROUTE_HOST_MODULE;

/// What the provider was sent, which Python's agentic hooks read back as the handler's own
/// model, provider and request params.
struct Sent {
    body: Py<PyAny>,
    model: String,
    provider: String,
    requested: Py<PyAny>,
}

/// Python's Messages handler hands the provider's answer to the callbacks overriding
/// `async_should_run_agentic_loop` before `@client` sees it, and wraps a stream so they run
/// once it ends. This hook asks the route host to do both at the same points.
pub(super) struct AgenticLoop {
    asynchronous: bool,
    request: Option<Py<PyDict>>,
    sent: Option<Sent>,
}

impl AgenticLoop {
    pub(super) fn new(asynchronous: bool) -> Self {
        Self {
            asynchronous,
            request: None,
            sent: None,
        }
    }

    fn host_call<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        subject: &Bound<'py, PyAny>,
        sent: &Sent,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(request) = &self.request else {
            return Ok(None);
        };
        py.import(ROUTE_HOST_MODULE)?
            .getattr(name)?
            .call1((
                subject,
                request.bind(py),
                sent.body.bind(py),
                &sent.model,
                &sent.provider,
                sent.requested.bind(py),
            ))
            .map(Some)
    }

    fn resume_response(
        &mut self,
        _: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        result.map(HookStep::Ready)
    }
}

impl CallHooks<PythonRuntime> for AgenticLoop {
    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
        if !self.asynchronous {
            return Ok(());
        }
        let runs = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("runs_agentic_hooks")?
            .call1((arguments.bind(py),))?
            .extract::<bool>()?;
        self.request = runs.then(|| arguments.clone_ref(py));
        Ok(())
    }

    fn before_provider_request(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
        let Some(request) = &self.request else {
            return Ok(HookStep::Ready(wire));
        };
        if self.sent.is_none() {
            py.import(ROUTE_HOST_MODULE)?
                .getattr("record_agentic_loop")?
                .call1((request.bind(py), &context.custom_llm_provider))?;
            self.sent = Some(Sent {
                body: to_py(py, &wire.body)?,
                model: context.model.clone(),
                provider: context.custom_llm_provider.clone(),
                requested: to_py(py, &context.optional_params)?,
            });
        }
        Ok(HookStep::Ready(wire))
    }

    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        _: Timing,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        let Some(sent) = &self.sent else {
            return Ok(HookStep::Ready(response));
        };
        match self.host_call(py, "agentic_response", response.bind(py), sent)? {
            Some(awaitable) => Ok(HookStep::Await(awaitable.unbind(), Self::resume_response)),
            None => Ok(HookStep::Ready(response)),
        }
    }

    fn on_stream_open(&mut self, py: Python<'_>, head: &Py<PyAny>) -> PyResult<()> {
        if let Some(sent) = &self.sent {
            self.host_call(py, "agentic_stream", head.bind(py), sent)?;
        }
        Ok(())
    }
}

impl PythonOwned for AgenticLoop {
    fn close(&mut self, _: Python<'_>) {
        self.request = None;
        self.sent = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.request)?;
        if let Some(sent) = &self.sent {
            visit.call(&sent.body)?;
            visit.call(&sent.requested)?;
        }
        Ok(())
    }
}
