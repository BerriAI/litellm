//! The legacy `Logging` contract as one adapter: every event and interception the driver
//! raises is answered with the same `Logging` calls, in the same order, as the Python
//! `@client` path makes them.

use litellm_callbacks::event::{
    AttemptInfo, CallEvent, FailureOrigin, RequestContext, Timing, WireRequest, epoch_seconds,
};
use litellm_callbacks::failure::FailureClass;
use litellm_host_python::{
    AdapterStep, CallbackAdapter, PublicValue, from_py, missing_state, to_py,
};
use pyo3::exceptions::{PyBaseException, PyException};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::deferred::{PendingLogging, PendingSuccess};
use crate::{
    DeploymentHooks, LegacyCallbacks, PublicCall, PythonLogger, finalize, is_internal_call,
    prepare, setup,
};

/// What the legacy contract needs to know about the route it is logging.
#[derive(Clone, Copy, Debug)]
pub struct LegacySurface {
    pub call_type: &'static str,
    /// What `Logging.pre_call` is told the input was.
    pub input_description: &'static str,
}

enum Pending {
    DeploymentPreCall,
    DeploymentPostCall,
    DeploymentFailure,
    AsyncFailure,
}

pub struct LegacyLogging {
    surface: LegacySurface,
    call: PublicCall,
    logger: Option<PythonLogger>,
    start: Py<PyAny>,
    end: Option<Py<PyAny>>,
    response: Option<Py<PyAny>>,
    error: Option<Py<PyBaseException>>,
    body: Option<Py<PyDict>>,
    headers: Option<Py<PyDict>>,
    asynchronous: bool,
    internal: bool,
    pending: Option<Pending>,
    /// The current attempt's failure already reached the failure callbacks, so the
    /// terminal failure that follows it dispatches nothing more.
    attempt_dispatched: bool,
}

fn datetime(py: Python<'_>, epoch_seconds: f64) -> PyResult<Py<PyAny>> {
    py.import("datetime")?
        .getattr("datetime")?
        .call_method1("fromtimestamp", (epoch_seconds,))
        .map(Bound::unbind)
}

fn is_cancellation(py: Python<'_>, error: &PyErr) -> bool {
    !error.is_instance_of::<PyException>(py)
}

impl LegacyLogging {
    pub fn new(
        py: Python<'_>,
        surface: LegacySurface,
        call: PublicCall,
        asynchronous: bool,
    ) -> Self {
        Self {
            surface,
            call,
            logger: None,
            start: py.None(),
            end: None,
            response: None,
            error: None,
            body: None,
            headers: None,
            asynchronous,
            internal: false,
            pending: None,
            attempt_dispatched: false,
        }
    }

    /// A later attempt of the same call. In Python each attempt is its own `@client`
    /// call: the previous attempt's context is restored, `function_setup` runs again on
    /// the caller's keywords with a fresh call id and the shared trace id, and the
    /// deployment hook and limits run for the new attempt.
    fn restart(&mut self, py: Python<'_>, attempt: &AttemptInfo) -> PyResult<AdapterStep> {
        if let Some(logger) = self.logger.take() {
            logger.restore_context(py)?;
        }
        let fresh = self.call.kwargs().bind(py).copy()?;
        for stale in ["litellm_call_id", "litellm_logging_obj"] {
            if fresh.contains(stale)? {
                fresh.del_item(stale)?;
            }
        }
        fresh.set_item("litellm_trace_id", &attempt.trace_id)?;
        self.response = None;
        self.error = None;
        self.body = None;
        self.headers = None;
        self.attempt_dispatched = false;
        self.begin(py, fresh.unbind(), epoch_seconds())
    }

    /// Deployment hooks are awaited, and Python's synchronous `@client` wrapper never
    /// runs them.
    fn deployment_hooks(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.asynchronous && DeploymentHooks::needed(py)?)
    }

    fn logger(&self) -> PyResult<&PythonLogger> {
        self.logger.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("call logging is not initialized")
        })
    }

    fn prepare(&mut self, py: Python<'_>) -> PyResult<AdapterStep> {
        let prepared = prepare(py, self.call.kwargs().bind(py), self.logger()?)?.unbind();
        self.call.set_kwargs(prepared);
        Ok(AdapterStep::Arguments(self.call.kwargs().clone_ref(py)))
    }

    fn finalize(&mut self, py: Python<'_>) -> PyResult<AdapterStep> {
        finalize(
            py,
            &self.response,
            self.logger()?,
            self.call.kwargs(),
            &self.start,
            &self.end,
        )?;
        self.response
            .as_ref()
            .map(|response| AdapterStep::Response(response.clone_ref(py)))
            .ok_or_else(missing_state)
    }

    fn dispatch_success(&self, py: Python<'_>) -> PyResult<()> {
        match self.try_dispatch_success(py) {
            Err(error) if error.is_instance_of::<PyException>(py) => {
                error.write_unraisable(py, self.logger.as_ref().map(|logger| logger.object(py)));
                Ok(())
            }
            result => result,
        }
    }

    fn try_dispatch_success(&self, py: Python<'_>) -> PyResult<()> {
        let logger = self.logger()?;
        let pending = || PendingSuccess {
            logger: logger.clone_ref(py),
            response: self.response.as_ref().map(|value| value.clone_ref(py)),
            start: self.start.clone_ref(py),
            end: self.end.as_ref().map(|value| value.clone_ref(py)),
        };
        if !self.asynchronous {
            return pending().sync(py);
        }
        if !self.internal
            && self
                .call
                .kwargs()
                .bind(py)
                .get_item("fallbacks")?
                .is_none_or(|value| value.is_none())
        {
            if !logger.callbacks_needed(py, "async_success")? {
                logger.success_bookkeeping(py, &self.response, &self.start, &self.end, true)?;
            } else if logger.defers_async_logging(py) {
                let pending = Py::new(
                    py,
                    PendingLogging {
                        pending: Some(pending()),
                    },
                )?;
                logger.defer_success(py, pending.bind(py).as_any())?;
            } else {
                pending().asynchronous(py)?;
            }
        }
        logger.sync_success_for_async_call(py, &self.response, &self.start, &self.end)
    }

    /// The sync failure handler, then the async one for async calls. Ordinary handler
    /// errors never replace the selected failure or suppress the other family; a
    /// cancellation does end the call.
    fn dispatch_failure(&mut self, py: Python<'_>) -> PyResult<AdapterStep> {
        let (Some(logger), Some(error)) = (&self.logger, &self.error) else {
            return Ok(AdapterStep::Done);
        };
        if self.asynchronous && self.internal {
            return Ok(AdapterStep::Done);
        }
        if let Err(failure) = logger.failure(py, error, &self.start, &self.end, false)
            && is_cancellation(py, &failure)
        {
            return Err(failure);
        }
        if !self.asynchronous {
            return Ok(AdapterStep::Done);
        }
        match logger.failure(py, error, &self.start, &self.end, true) {
            Ok(Some(awaitable)) => {
                self.pending = Some(Pending::AsyncFailure);
                Ok(AdapterStep::Await(awaitable))
            }
            Ok(None) => Ok(AdapterStep::Done),
            Err(failure) if is_cancellation(py, &failure) => Err(failure),
            Err(_) => Ok(AdapterStep::Done),
        }
    }
}

impl CallbackAdapter for LegacyLogging {
    fn begin(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<AdapterStep> {
        self.call.set_kwargs(arguments);
        self.start = datetime(py, started_at)?;
        self.internal = is_internal_call(py)?;
        let result = setup(
            py,
            self.surface.call_type,
            self.call.args(),
            self.call.kwargs(),
            &self.start,
            self.asynchronous,
        )?;
        self.logger = Some(result.logger()?);
        self.call.set_kwargs(result.kwargs()?);
        if self.deployment_hooks(py)? {
            self.pending = Some(Pending::DeploymentPreCall);
            return Ok(AdapterStep::Await(DeploymentHooks::before_call(
                py,
                self.call.kwargs(),
                self.surface.call_type,
            )?));
        }
        self.prepare(py)
    }

    fn before_send(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<AdapterStep> {
        let logger = self.logger()?;
        logger.update_from_kwargs(py, self.call.kwargs(), &wire, context)?;
        if !logger.callbacks_needed(py, "payload")? {
            logger.record_api_call_start(py)?;
            return Ok(AdapterStep::Wire(wire));
        }
        let body = to_py(py, &wire.body)?
            .into_bound(py)
            .cast_into::<PyDict>()?;
        for name in context.passthrough_fields.iter() {
            if let Some(value) = self.call.lookup(py, name)? {
                body.set_item(name, value)?;
            }
        }
        let headers = PyDict::new(py);
        for (name, value) in &wire.headers {
            headers.set_item(name, value)?;
        }
        self.body = Some(body.clone().unbind());
        self.headers = Some(headers.clone().unbind());
        let api_key = self.call.lookup(py, "api_key")?;
        self.logger()?.pre_call(
            py,
            self.surface.input_description,
            api_key.as_ref(),
            &body,
            &headers,
            &wire.url,
        )?;
        let headers = headers
            .iter()
            .map(|(name, value)| Ok((name.extract::<String>()?, value.extract::<String>()?)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(AdapterStep::Wire(Box::new(WireRequest {
            body: from_py(&body)?,
            headers,
            ..*wire
        })))
    }

    fn after_success(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<AdapterStep> {
        self.end = Some(datetime(py, timing.end_time)?);
        self.response = Some(response);
        if self.deployment_hooks(py)? {
            self.pending = Some(Pending::DeploymentPostCall);
            return Ok(AdapterStep::Await(DeploymentHooks::after_success(
                py,
                self.call.kwargs(),
                &self.response,
                self.surface.call_type,
            )?));
        }
        self.finalize(py)
    }

    fn emit(
        &mut self,
        py: Python<'_>,
        event: &CallEvent,
        public: Option<PublicValue<'_>>,
    ) -> PyResult<AdapterStep> {
        match (event, public) {
            (CallEvent::AttemptStarted { attempt }, _) if attempt.index == 0 => {
                Ok(AdapterStep::Done)
            }
            (CallEvent::AttemptStarted { attempt }, _) => self.restart(py, attempt),
            (CallEvent::Failed { timing, .. }, Some(PublicValue::Error(_)))
                if self.attempt_dispatched =>
            {
                self.end = Some(datetime(py, timing.end_time)?);
                Ok(AdapterStep::Done)
            }
            (CallEvent::ResponseReceived { raw }, _) => {
                let logger = self.logger()?;
                if logger.callbacks_needed(py, "payload")? {
                    logger.post_call(py, &raw.body, self.body.as_ref(), self.headers.as_ref())?;
                }
                Ok(AdapterStep::Done)
            }
            (CallEvent::Succeeded { timing }, Some(PublicValue::Response(response))) => {
                self.end = Some(datetime(py, timing.end_time)?);
                self.response = Some(response.clone_ref(py));
                self.dispatch_success(py)?;
                Ok(AdapterStep::Done)
            }
            (CallEvent::Failed { timing, origin }, Some(PublicValue::Error(error))) => {
                self.end = Some(datetime(py, timing.end_time)?);
                self.error = Some(error.clone_ref(py).into_value(py));
                if *origin == FailureOrigin::Call
                    && self.logger.is_some()
                    && self.deployment_hooks(py)?
                {
                    let error = self.error.as_ref().ok_or_else(missing_state)?;
                    self.pending = Some(Pending::DeploymentFailure);
                    return Ok(AdapterStep::Await(DeploymentHooks::after_failure(
                        py,
                        self.call.kwargs(),
                        error,
                        self.surface.call_type,
                    )?));
                }
                self.dispatch_failure(py)
            }
            _ => Err(missing_state()),
        }
    }

    /// The attempt's failure families, exactly as the terminal failure of a lone call
    /// runs them: the deployment failure hook, then the sync and async handlers.
    fn attempt_failed(
        &mut self,
        py: Python<'_>,
        _attempt: &AttemptInfo,
        _class: FailureClass,
        error: &PyErr,
    ) -> PyResult<AdapterStep> {
        self.end = Some(datetime(py, epoch_seconds())?);
        self.error = Some(error.clone_ref(py).into_value(py));
        self.attempt_dispatched = true;
        if self.logger.is_some() && self.deployment_hooks(py)? {
            let error = self.error.as_ref().ok_or_else(missing_state)?;
            self.pending = Some(Pending::DeploymentFailure);
            return Ok(AdapterStep::Await(DeploymentHooks::after_failure(
                py,
                self.call.kwargs(),
                error,
                self.surface.call_type,
            )?));
        }
        self.dispatch_failure(py)
    }

    fn resume(&mut self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<AdapterStep> {
        match self.pending.take().ok_or_else(missing_state)? {
            Pending::DeploymentPreCall => {
                self.call
                    .set_kwargs(result?.into_bound(py).cast_into::<PyDict>()?.unbind());
                self.prepare(py)
            }
            Pending::DeploymentPostCall => {
                self.response = Some(result?);
                self.finalize(py)
            }
            Pending::DeploymentFailure => self.dispatch_failure(py),
            Pending::AsyncFailure => match result {
                Err(failure) if is_cancellation(py, &failure) => Err(failure),
                _ => Ok(AdapterStep::Done),
            },
        }
    }

    fn close(&mut self, py: Python<'_>) {
        if let Some(logger) = self.logger.take()
            && let Err(error) = logger.restore_context(py)
        {
            error.write_unraisable(py, None);
        }
        self.body = None;
        self.headers = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.call.traverse(visit)?;
        if let Some(logger) = &self.logger {
            logger.traverse(visit)?;
        }
        visit.call(&self.start)?;
        visit.call(&self.end)?;
        visit.call(&self.response)?;
        visit.call(&self.error)?;
        visit.call(&self.body)?;
        visit.call(&self.headers)
    }
}

#[cfg(test)]
#[path = "../tests/attempts.rs"]
mod attempts_tests;
#[cfg(test)]
#[path = "../tests/deployment_hooks.rs"]
mod deployment_hooks_tests;
#[cfg(test)]
#[path = "../tests/payload.rs"]
mod payload_tests;
#[cfg(test)]
#[path = "../tests/terminal.rs"]
mod terminal_tests;
