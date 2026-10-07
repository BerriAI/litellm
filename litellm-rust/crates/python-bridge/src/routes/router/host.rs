use litellm_host::machine::HostServices;
use litellm_host_python::{
    InvokeError, PythonBinding, PythonHostCalls, PythonOwned, missing_state,
};
use litellm_router::{
    engine::Routed,
    host::{Attempt, FallbackCheck, Invoked, RouterHost},
};
use pyo3::{
    exceptions::PyRuntimeError,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use super::{BridgeError, PyObj, RouterHostCall, RouterProtocol, RouterRequest, python};
use litellm_host::protocol::Reply;

pub(super) struct BridgeHost(pub(super) HostServices<RouterProtocol>);

impl RouterHost for BridgeHost {
    type Response = PyObj;
    type Error = PyObj;
    type Fault = BridgeError;

    async fn invoke(&self, attempt: Attempt<PyObj>) -> Result<Invoked<PyObj, PyObj>, BridgeError> {
        self.0
            .call(|reply| RouterHostCall::Invoke { attempt, reply })
            .await
    }

    async fn sleep(&self, seconds: f64) -> Result<(), BridgeError> {
        self.0
            .call(|reply| RouterHostCall::Sleep { seconds, reply })
            .await
    }

    async fn allow_fallback(&self, check: FallbackCheck<PyObj>) -> Result<bool, BridgeError> {
        self.0
            .call(|reply| RouterHostCall::AllowFallback { check, reply })
            .await
    }
}

enum Pending {
    Invoke(Reply<Invoked<PyObj, PyObj>>),
    Sleep(Reply<()>),
    AllowFallback(Reply<bool>),
}

pub(super) struct RouterBinding {
    driver: Option<Py<PyAny>>,
    pending: Option<Pending>,
    asynchronous: bool,
}

impl RouterBinding {
    pub(super) fn new(driver: Py<PyAny>, asynchronous: bool) -> Self {
        Self {
            driver: Some(driver),
            pending: None,
            asynchronous,
        }
    }

    fn driver<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.driver
            .as_ref()
            .map(|driver| driver.bind(py).clone())
            .ok_or_else(missing_state)
    }

    fn settle(&mut self, py: Python<'_>, result: &Bound<'_, PyAny>) -> PyResult<()> {
        match self.pending.take() {
            Some(Pending::Invoke(reply)) => reply.send(python::invoked(py, result)?),
            Some(Pending::Sleep(reply)) => reply.send(()),
            Some(Pending::AllowFallback(reply)) => reply.send(result.extract()?),
            None => {
                return Err(PyRuntimeError::new_err(
                    "router host reply without a pending op",
                ));
            }
        }
        Ok(())
    }
}

impl PythonBinding for RouterBinding {
    type Protocol = RouterProtocol;
    type Failure = PyErr;

    fn decode_request(
        &mut self,
        _py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<RouterRequest, InvokeError<BridgeError>> {
        python::router_request(arguments).map_err(InvokeError::Python)
    }

    fn encode_response(
        &mut self,
        py: Python<'_>,
        routed: Routed<PyObj, PyObj>,
    ) -> PyResult<Py<PyAny>> {
        let outcome = python::outcome(py, &routed.outcome)?;
        let ops = python::ops(py, &routed.ops)?;
        self.driver(py)?
            .call_method1("success", (routed.response.clone_ref(py), outcome, ops))
            .map(Bound::unbind)
    }

    fn encode_stream_head(
        &mut self,
        _: Python<'_>,
        head: std::convert::Infallible,
    ) -> PyResult<Py<PyAny>> {
        match head {}
    }

    fn encode_chunk(
        &mut self,
        _: Python<'_>,
        chunk: std::convert::Infallible,
    ) -> PyResult<Py<PyAny>> {
        match chunk {}
    }

    fn map_error(&self, py: Python<'_>, error: BridgeError) -> PyResult<PyErr> {
        match error {
            BridgeError::Failed(failed) => {
                let raised = python::raised(py, &failed.error)?;
                let ops = python::ops(py, &failed.ops)?;
                let exception = self.driver(py)?.call_method1("failure", (raised, ops))?;
                Ok(PyErr::from_value(exception))
            }
            other => Ok(PyRuntimeError::new_err(other.to_string())),
        }
    }

    fn host_error(error: &PyErr) -> BridgeError {
        BridgeError::Request(error.to_string())
    }
}

impl PythonHostCalls<RouterProtocol> for RouterBinding {
    fn handle_host_call(
        &mut self,
        py: Python<'_>,
        call: RouterHostCall,
    ) -> Result<(), InvokeError<BridgeError>> {
        self.begin_host_call(py, call).map(|_| ())
    }

    fn begin_host_call(
        &mut self,
        py: Python<'_>,
        call: RouterHostCall,
    ) -> Result<Option<Py<PyAny>>, InvokeError<BridgeError>> {
        let driver = self.driver(py)?;
        let (method, argument, pending) = match call {
            RouterHostCall::Invoke { attempt, reply } => (
                if self.asynchronous {
                    "invoke"
                } else {
                    "invoke_sync"
                },
                python::attempt(py, &attempt)?,
                Pending::Invoke(reply),
            ),
            RouterHostCall::Sleep { seconds, reply } => (
                if self.asynchronous {
                    "sleep"
                } else {
                    "sleep_sync"
                },
                pyo3::types::PyFloat::new(py, seconds).into_any().unbind(),
                Pending::Sleep(reply),
            ),
            RouterHostCall::AllowFallback { check, reply } => (
                if self.asynchronous {
                    "allow_fallback"
                } else {
                    "allow_fallback_sync"
                },
                python::fallback_check(py, &check)?,
                Pending::AllowFallback(reply),
            ),
        };
        self.pending = Some(pending);
        let result = driver.call_method1(method, (argument,));
        if self.asynchronous {
            return result
                .map(|awaitable| Some(awaitable.unbind()))
                .map_err(|error| {
                    self.pending = None;
                    InvokeError::Python(error)
                });
        }
        self.resume_host_call(py, result.map(Bound::unbind))
    }

    fn resume_host_call(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> Result<Option<Py<PyAny>>, InvokeError<BridgeError>> {
        match result {
            Ok(value) => {
                self.settle(py, value.bind(py))?;
                Ok(None)
            }
            Err(error) => {
                self.pending = None;
                Err(InvokeError::Python(error))
            }
        }
    }
}

impl PythonOwned for RouterBinding {
    fn close(&mut self, _: Python<'_>) {
        self.pending = None;
        self.driver = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.driver)
    }
}
