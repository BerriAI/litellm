use std::marker::PhantomData;

use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use serde::Serialize;

use litellm_core::call_lifecycle::host::{HostPhase, HostStep};
use litellm_core::call_lifecycle::provider::{
    CompletedCall, CompletedOperation, CompletedReply, CompletedRoute, ProviderRequest,
    ProviderResponse,
};
use litellm_core::call_lifecycle::workflow::LifecycleOperation;
use litellm_python_interop::{
    from_py_preserving_errors as from_py, to_py_preserving_errors as to_py,
};

use super::contract::{AdapterOperation, CallMode, PythonCallType};
use super::{OperationClass, PythonCallState, PythonRoute, missing_state, now, run_call};
use crate::errors::execution_error_to_pyerr;

pub(crate) trait PythonCompletedRoute: CompletedRoute {
    const SYNC_CALL_TYPE: PythonCallType;
    const ASYNC_CALL_TYPE: PythonCallType;

    fn admit(request: &Bound<'_, PyDict>) -> PyResult<()>;
    fn project(request: &Bound<'_, PyDict>) -> PyResult<Self::Request>;
}

struct PythonCompletedHost<R: PythonCompletedRoute> {
    state: PythonCallState,
    request: Py<PyDict>,
    adapter: Py<PyAny>,
    pending: Option<AdapterOperation>,
    route: PhantomData<R>,
}

impl<R: PythonCompletedRoute> PythonRoute for PythonCompletedHost<R>
where
    R::Response: Serialize,
{
    type Call = CompletedCall<R>;

    fn state(&self) -> &PythonCallState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut PythonCallState {
        &mut self.state
    }

    fn classify(operation: &CompletedOperation<R::Response>) -> OperationClass {
        match operation {
            CompletedOperation::Lifecycle(LifecycleOperation::Phase(
                HostPhase::Prepare | HostPhase::PostProcess | HostPhase::CacheStore,
            )) => OperationClass::Route,
            CompletedOperation::Lifecycle(LifecycleOperation::Phase(phase)) => {
                OperationClass::Phase(*phase)
            }
            CompletedOperation::Lifecycle(LifecycleOperation::Success { .. }) => {
                OperationClass::Phase(HostPhase::Success)
            }
            CompletedOperation::Lifecycle(LifecycleOperation::Failure { .. }) => {
                OperationClass::Phase(HostPhase::Failure)
            }
            _ => OperationClass::Route,
        }
    }

    fn lifecycle_result() -> CompletedReply<R::Request> {
        CompletedReply::Lifecycle(Ok(()))
    }
    fn map_error(error: litellm_core::Error) -> PyErr {
        execution_error_to_pyerr(error)
    }

    fn invoke(
        &mut self,
        py: Python<'_>,
        operation: CompletedOperation<R::Response>,
    ) -> PyResult<CompletedReply<R::Request>> {
        match self.invoke_step(py, operation)? {
            HostStep::Ready(reply) => Ok(reply),
            HostStep::Suspend(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "route operation requires async execution",
            )),
        }
    }

    fn invoke_step(
        &mut self,
        py: Python<'_>,
        operation: CompletedOperation<R::Response>,
    ) -> PyResult<HostStep<CompletedReply<R::Request>, Py<PyAny>>> {
        let (operation, payload) = match operation {
            CompletedOperation::Lifecycle(LifecycleOperation::Phase(HostPhase::Prepare)) => {
                self.state.prepare(py)?;
                return Ok(HostStep::Ready(CompletedReply::Prepared(Ok(
                    crate::cache::plan(
                        py,
                        self.state.call_type.as_str(),
                        self.state.kwargs.bind(py),
                    )?,
                ))));
            }
            CompletedOperation::Lifecycle(LifecycleOperation::Phase(HostPhase::PostProcess)) => (
                AdapterOperation::PostProcess,
                self.state
                    .response
                    .as_ref()
                    .ok_or_else(missing_state)?
                    .clone_ref(py),
            ),
            CompletedOperation::Lifecycle(LifecycleOperation::Phase(HostPhase::CacheStore)) => (
                AdapterOperation::CacheStore,
                self.state
                    .response
                    .as_ref()
                    .ok_or_else(missing_state)?
                    .clone_ref(py),
            ),
            CompletedOperation::Lifecycle(LifecycleOperation::ConstructCachedResponse(
                response,
            )) => {
                self.state.end = Some(now(py)?);
                (AdapterOperation::CachedResponse, to_py(py, &response)?)
            }
            CompletedOperation::Lifecycle(LifecycleOperation::ProjectRequest) => (
                AdapterOperation::ProjectRequest,
                self.request.clone_ref(py).into_any(),
            ),
            CompletedOperation::BeforeRequest(request) => {
                (AdapterOperation::BeforeRequest, to_py(py, &request)?)
            }
            CompletedOperation::AfterResponse(response) => {
                (AdapterOperation::AfterResponse, to_py(py, &response)?)
            }
            CompletedOperation::Lifecycle(LifecycleOperation::ConstructResponse(response)) => {
                self.state.end = Some(now(py)?);
                (
                    AdapterOperation::ConstructResponse,
                    to_py(py, response.as_ref())?,
                )
            }
            CompletedOperation::Lifecycle(LifecycleOperation::MapFailure(error)) => {
                if self.state.error.is_none() {
                    self.state.retain_error(py, execution_error_to_pyerr(error));
                }
                if self.state.end.is_none() {
                    self.state.end = Some(now(py)?);
                }
                (
                    AdapterOperation::MapFailure,
                    self.state
                        .error
                        .as_ref()
                        .ok_or_else(missing_state)?
                        .clone_ref(py)
                        .into_any(),
                )
            }
            _ => return Err(missing_state()),
        };
        self.pending = Some(operation);
        let step = self.adapter.bind(py).call_method1(
            "invoke",
            (
                operation.as_str(),
                payload,
                &self.request,
                &self.state.kwargs,
                self.state.logger()?.object(py),
            ),
        )?;
        let protocol = py.import("litellm.rust_bridge.lifecycle")?;
        if step.is_instance(&protocol.getattr("Await")?)? {
            if !self.state.mode.is_async() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "sync route operation suspended",
                ));
            }
            return Ok(HostStep::Suspend(step.getattr("awaitable")?.unbind()));
        }
        if !step.is_instance(&protocol.getattr("Complete")?)? {
            return Err(missing_state());
        }
        self.accept_route(py, step.getattr("value")?.unbind())
            .map(HostStep::Ready)
    }

    fn accept_route(
        &mut self,
        py: Python<'_>,
        value: Py<PyAny>,
    ) -> PyResult<CompletedReply<R::Request>> {
        Ok(match self.pending.take().ok_or_else(missing_state)? {
            AdapterOperation::PostProcess => Self::lifecycle_result(),
            AdapterOperation::CacheStore => CompletedReply::CacheStore(Ok(if value.is_none(py) {
                None
            } else {
                Some(from_py(value.bind(py))?)
            })),
            AdapterOperation::ProjectRequest => {
                let request = value.into_bound(py).cast_into::<PyDict>()?;
                CompletedReply::Request(Ok(R::project(&request)?))
            }
            AdapterOperation::BeforeRequest => {
                CompletedReply::BeforeRequest(Ok(from_py::<ProviderRequest>(value.bind(py))?))
            }
            AdapterOperation::AfterResponse => {
                CompletedReply::AfterResponse(Ok(from_py::<ProviderResponse>(value.bind(py))?))
            }
            AdapterOperation::CachedResponse | AdapterOperation::ConstructResponse => {
                self.state.response = Some(value);
                Self::lifecycle_result()
            }
            AdapterOperation::MapFailure => {
                self.state
                    .retain_error(py, PyErr::from_value(value.into_bound(py)));
                Self::lifecycle_result()
            }
        })
    }

    fn cleanup(&mut self) {
        self.pending = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.request)?;
        visit.call(&self.adapter)
    }
}

pub(crate) fn run<R: PythonCompletedRoute>(
    py: Python<'_>,
    request: Bound<'_, PyDict>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
    host: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>>
where
    R::Response: Serialize,
{
    R::admit(&request)?;
    let controls = crate::cache::snapshot(
        py,
        if asynchronous {
            R::ASYNC_CALL_TYPE.as_str()
        } else {
            R::SYNC_CALL_TYPE.as_str()
        },
        &request,
    )
    .map_err(crate::errors::terminal_pyerr)?;
    crate::errors::admit(
        litellm_core::call_lifecycle::cache::ResponseCachePlan {
            controls,
            ..Default::default()
        }
        .admit(),
    )?;
    let call = R::operation(asynchronous);
    let host = PythonCompletedHost::<R> {
        state: PythonCallState::new(
            py,
            args.unbind(),
            kwargs.copy()?.unbind(),
            CallMode::from_async(asynchronous),
            if asynchronous {
                R::ASYNC_CALL_TYPE
            } else {
                R::SYNC_CALL_TYPE
            },
        )?,
        request: request.unbind(),
        adapter: host.unbind(),
        pending: None,
        route: PhantomData,
    };
    run_call(py, call, host).map_err(crate::errors::terminal_pyerr)
}
