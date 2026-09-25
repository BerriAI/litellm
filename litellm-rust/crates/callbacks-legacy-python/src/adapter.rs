//! The legacy `Logging` contract as one adapter: every event and interception the driver
//! raises is answered with the same `Logging` calls, in the same order, as the Python
//! `@client` path makes them.

use litellm_host::event::{
    FailureOrigin, MachineEvent, PublicRequest, RequestContext, Timing, WireRequest, epoch_seconds,
};
use litellm_host_python::{
    LifecycleEvent, LifecycleStep, PythonLifecycle, from_py, json_fields, missing_state, to_py,
};
use pyo3::{
    exceptions::{PyBaseException, PyException},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDateTime, PyDict, PyList},
};
use serde_json::Value;

use crate::{
    CallbackHooks, LegacyCallbacks, PublicCall, PythonLogger,
    deferred::{PendingLogging, PendingSuccess},
    finalize, is_internal_call,
    python::Streaming,
    setup,
};

/// What the legacy contract needs to know about the route it is logging.
#[derive(Clone, Copy, Debug)]
pub struct LegacySurface {
    pub call_type: &'static str,
    /// What `Logging.pre_call` is told the input was.
    pub input_description: &'static str,
    /// How a streamed response is billed; `None` for a route that never streams.
    pub stream: Option<PassThroughStream>,
}

/// The pass-through billing a streamed response goes through once its chunks are in.
#[derive(Clone, Copy, Debug)]
pub struct PassThroughStream {
    pub url_route: &'static str,
    /// A value of Python's `EndpointType`.
    pub endpoint_type: &'static str,
}

/// What the Messages stream iterator keeps for its end-of-stream billing.
struct DeliveredStream {
    chunks: Py<PyList>,
    first_chunk: Option<Py<PyAny>>,
}

enum Pending {
    PreRequest(Box<PublicRequest>),
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
    context: Option<RequestContext>,
    stream: Option<DeliveredStream>,
    asynchronous: bool,
    internal: bool,
    pending: Option<Pending>,
}

fn datetime(py: Python<'_>, epoch_seconds: f64) -> PyResult<Py<PyAny>> {
    PyDateTime::from_timestamp(py, epoch_seconds, None).map(|value| value.into_any().unbind())
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
            context: None,
            stream: None,
            asynchronous,
            internal: false,
            pending: None,
        }
    }

    /// Deployment hooks are awaited, and Python's synchronous `@client` wrapper never
    /// runs them.
    fn runs_deployment_hooks(&self) -> bool {
        self.asynchronous
    }

    fn logger(&self) -> PyResult<&PythonLogger> {
        self.logger.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("call logging is not initialized")
        })
    }

    /// The keyword view the rest of the call reads: a copy, so the deployment hook's own
    /// dict is left as the hook returned it, carrying the logger as `@client` injects it.
    /// The driver's preflight rewrites this same dict before the host projects from it.
    fn prepare(&mut self, py: Python<'_>) -> PyResult<LifecycleStep> {
        let prepared = self.call.kwargs().bind(py).copy()?;
        prepared.set_item("litellm_logging_obj", self.logger()?.object(py))?;
        self.call.set_kwargs(prepared.unbind());
        Ok(LifecycleStep::Arguments(self.call.kwargs().clone_ref(py)))
    }

    fn finalize(&mut self, py: Python<'_>) -> PyResult<LifecycleStep> {
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
            .map(|response| LifecycleStep::Response(response.clone_ref(py)))
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
            if logger.defers_async_logging(py) {
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

    fn stream_success(&self, py: Python<'_>, stream: &DeliveredStream) -> PyResult<()> {
        let logger = self.logger()?;
        let billing = self.surface.stream.ok_or_else(missing_state)?;
        let billed = Streaming::Success.call(
            py,
            (
                logger.object(py),
                billing.url_route,
                billing.endpoint_type,
                &self.body,
                &stream.chunks,
                &self.start,
                &self.end,
                &stream.first_chunk,
            ),
        );
        match billed {
            Err(error) if error.is_instance_of::<PyException>(py) => {
                error.write_unraisable(py, Some(logger.object(py)));
                Ok(())
            }
            result => result.map(|_| ()),
        }
    }

    /// A failure after the stream reached the caller bills the delivered chunks as
    /// partial usage. The sync path has no loop to schedule that on, so it falls back to
    /// the plain failure handler.
    fn stream_failure(&mut self, py: Python<'_>) -> PyResult<LifecycleStep> {
        let (Some(logger), Some(error), Some(stream), Some(billing)) =
            (&self.logger, &self.error, &self.stream, self.surface.stream)
        else {
            return Ok(LifecycleStep::Done);
        };
        if !self.asynchronous {
            return self.dispatch_failure(py);
        }
        let scheduled = Streaming::Failure.call(
            py,
            (
                logger.object(py),
                billing.endpoint_type,
                &self.body,
                &stream.chunks,
                error,
            ),
        );
        match scheduled {
            Ok(awaitable) => {
                self.pending = Some(Pending::AsyncFailure);
                Ok(LifecycleStep::Await(awaitable.unbind()))
            }
            Err(failure) if is_cancellation(py, &failure) => Err(failure),
            Err(_) => Ok(LifecycleStep::Done),
        }
    }

    /// The sync failure handler, then the async one for async calls. Ordinary handler
    /// errors never replace the selected failure or suppress the other family; a
    /// cancellation does end the call.
    fn dispatch_failure(&mut self, py: Python<'_>) -> PyResult<LifecycleStep> {
        let (Some(logger), Some(error)) = (&self.logger, &self.error) else {
            return Ok(LifecycleStep::Done);
        };
        if self.asynchronous && self.internal {
            return Ok(LifecycleStep::Done);
        }
        if let Err(failure) = logger.failure(py, error, &self.start, &self.end, false)
            && is_cancellation(py, &failure)
        {
            return Err(failure);
        }
        if !self.asynchronous {
            return Ok(LifecycleStep::Done);
        }
        match logger.failure(py, error, &self.start, &self.end, true) {
            Ok(Some(awaitable)) => {
                self.pending = Some(Pending::AsyncFailure);
                Ok(LifecycleStep::Await(awaitable))
            }
            Ok(None) => Ok(LifecycleStep::Done),
            Err(failure) if is_cancellation(py, &failure) => Err(failure),
            Err(_) => Ok(LifecycleStep::Done),
        }
    }
}

impl PythonLifecycle for LegacyLogging {
    fn begin(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<LifecycleStep> {
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
        self.call
            .kwargs()
            .bind(py)
            .set_item("litellm_logging_obj", self.logger()?.object(py))?;
        if self.runs_deployment_hooks() {
            self.pending = Some(Pending::DeploymentPreCall);
            return Ok(LifecycleStep::Await(CallbackHooks::before_call(
                py,
                self.logger()?,
                self.call.kwargs(),
                self.surface.call_type,
            )?));
        }
        self.prepare(py)
    }

    fn pre_request(&mut self, py: Python<'_>, request: PublicRequest) -> PyResult<LifecycleStep> {
        if !self.asynchronous {
            return Ok(LifecycleStep::Params(request.params));
        }
        let kwargs = self.call.kwargs().bind(py).copy()?;
        for name in request.fields {
            if !kwargs.contains(name)?
                && let Some(value) = self.call.lookup(py, name)?
            {
                kwargs.set_item(name, value)?;
            }
        }
        let params = PyDict::new(py);
        params.set_item("custom_llm_provider", &request.custom_llm_provider)?;
        kwargs.set_item("litellm_params", params)?;
        let messages = match self.call.lookup(py, "messages")? {
            Some(messages) => messages,
            None => to_py(py, &request.messages)?.into_bound(py),
        };
        let awaitable = CallbackHooks::pre_request(py, &request.model, &messages, &kwargs)?;
        self.pending = Some(Pending::PreRequest(Box::new(request)));
        Ok(LifecycleStep::Await(awaitable))
    }

    fn before_send(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<LifecycleStep> {
        let logger = self.logger()?;
        logger.update_from_kwargs(py, self.call.kwargs(), &wire, context)?;
        let body = to_py(py, &wire.body)?
            .into_bound(py)
            .cast_into::<PyDict>()?;
        for (name, sent) in wire.body.as_object().into_iter().flatten() {
            if let Some(value) = self.call.lookup(py, name)?
                && from_py::<Value>(&value).is_ok_and(|caller| caller == *sent)
            {
                body.set_item(name, value)?;
            }
        }
        let headers = PyDict::new(py);
        for (name, value) in &wire.headers {
            headers.set_item(name, value)?;
        }
        self.body = Some(body.clone().unbind());
        self.headers = Some(headers.clone().unbind());
        self.context = Some(context.clone());
        self.logger()?.pre_call(
            py,
            self.surface.input_description,
            context.api_key.as_ref().map(|api_key| api_key.expose()),
            &body,
            &headers,
            &wire.url,
        )?;
        let headers = headers
            .iter()
            .map(|(name, value)| Ok((name.extract::<String>()?, value.extract::<String>()?)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(LifecycleStep::Wire(Box::new(WireRequest {
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
    ) -> PyResult<LifecycleStep> {
        self.end = Some(datetime(py, timing.end_time)?);
        self.response = Some(response);
        if self.runs_deployment_hooks() {
            self.pending = Some(Pending::DeploymentPostCall);
            return Ok(LifecycleStep::Await(CallbackHooks::after_success(
                py,
                self.call.kwargs(),
                &self.response,
                self.surface.call_type,
            )?));
        }
        self.finalize(py)
    }

    fn emit(&mut self, py: Python<'_>, event: LifecycleEvent<'_>) -> PyResult<LifecycleStep> {
        match event {
            LifecycleEvent::Started { .. } => Ok(LifecycleStep::Done),
            LifecycleEvent::Machine(MachineEvent::ResponseReceived { raw }) => {
                let api_key = self
                    .context
                    .as_ref()
                    .and_then(|context| context.api_key.as_ref())
                    .map(|api_key| api_key.expose());
                self.logger()?.post_call(
                    py,
                    &raw.body,
                    api_key,
                    self.body.as_ref(),
                    self.headers.as_ref(),
                )?;
                Ok(LifecycleStep::Done)
            }
            LifecycleEvent::Succeeded { timing, response } => {
                self.end = Some(datetime(py, timing.end_time)?);
                self.response = Some(response.clone_ref(py));
                match &self.stream {
                    Some(stream) => self.stream_success(py, stream)?,
                    None => self.dispatch_success(py)?,
                }
                Ok(LifecycleStep::Done)
            }
            LifecycleEvent::Failed {
                timing,
                origin,
                error,
            } => {
                self.end = Some(datetime(py, timing.end_time)?);
                self.error = Some(error.clone_ref(py).into_value(py));
                if self.stream.is_some() {
                    return self.stream_failure(py);
                }
                if origin == FailureOrigin::Call
                    && self.logger.is_some()
                    && self.runs_deployment_hooks()
                {
                    let error = self.error.as_ref().ok_or_else(missing_state)?;
                    self.pending = Some(Pending::DeploymentFailure);
                    return Ok(LifecycleStep::Await(CallbackHooks::after_failure(
                        py,
                        self.call.kwargs(),
                        error,
                        self.surface.call_type,
                    )?));
                }
                self.dispatch_failure(py)
            }
        }
    }

    fn opened(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.surface.stream.is_none() {
            return Err(missing_state());
        }
        Streaming::Opened.call(py, (self.logger()?.object(py),))?;
        self.stream = Some(DeliveredStream {
            chunks: PyList::empty(py).unbind(),
            first_chunk: None,
        });
        Ok(())
    }

    fn delivered(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()> {
        let stream = self.stream.as_mut().ok_or_else(missing_state)?;
        if stream.first_chunk.is_none() {
            stream.first_chunk = Some(datetime(py, epoch_seconds())?);
        }
        stream.chunks.bind(py).append(chunk)
    }

    fn resume(&mut self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<LifecycleStep> {
        match self.pending.take().ok_or_else(missing_state)? {
            Pending::PreRequest(request) => {
                let returned = result?;
                if returned.is_none(py) {
                    return Ok(LifecycleStep::Params(request.params));
                }
                let view = returned.into_bound(py).cast_into::<PyDict>()?.copy()?;
                if view.contains("litellm_params")? {
                    view.del_item("litellm_params")?;
                }
                let params =
                    json_fields(request.fields.iter().copied(), |name| view.get_item(name))?;
                self.call.set_kwargs(view.unbind());
                Ok(LifecycleStep::Params(params))
            }
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
                _ => Ok(LifecycleStep::Done),
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
        self.context = None;
        self.stream = None;
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
        if let Some(stream) = &self.stream {
            visit.call(&stream.chunks)?;
            visit.call(&stream.first_chunk)?;
        }
        visit.call(&self.body)?;
        visit.call(&self.headers)
    }
}

#[cfg(test)]
mod deployment_hooks_tests {
    use std::ffi::CStr;

    use litellm_host::event::{FailureOrigin, Timing};
    use litellm_host_python::{LifecycleEvent, LifecycleStep, PythonLifecycle};
    use pyo3::exceptions::asyncio::CancelledError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use rstest::rstest;

    use super::LegacyLogging;
    use crate::test_support::{legacy_call, local, namespace, run};

    const CALL: &CStr = c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
kwargs = {'logger': logger, 'document': document}
";

    const TIMING: Timing = Timing {
        start_time: 0.0,
        end_time: 1.0,
    };

    fn begin<'py>(
        py: Python<'py>,
        locals: &Bound<'py, PyDict>,
        asynchronous: bool,
    ) -> (LegacyLogging, LifecycleStep) {
        let mut logging = legacy_call(py, locals, asynchronous);
        let kwargs = local(locals, "kwargs")
            .cast_into::<PyDict>()
            .unwrap()
            .unbind();
        let step = logging.begin(py, kwargs, 0.0).unwrap();
        (logging, step)
    }

    fn arguments<'py>(py: Python<'py>, step: LifecycleStep) -> Bound<'py, PyDict> {
        let LifecycleStep::Arguments(arguments) = step else {
            panic!("expected the prepared arguments");
        };
        arguments.into_bound(py)
    }

    fn awaits_deployment_hook(step: &LifecycleStep) -> bool {
        matches!(step, LifecycleStep::Await(_))
    }

    #[rstest]
    #[case::synchronous(false)]
    #[case::asynchronous(true)]
    fn deployment_pre_call_hook_runs_only_for_asynchronous_calls(#[case] asynchronous: bool) {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, CALL);
            let (_, step) = begin(py, &locals, asynchronous);
            assert_eq!(awaits_deployment_hook(&step), asynchronous);
            let names: Vec<String> = local(&locals, "logger")
                .call_method0("names")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(names.contains(&"pre_hook".to_string()), asynchronous);
        });
    }

    #[test]
    fn kwargs_returned_by_the_pre_call_hook_are_what_the_call_prepares() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(
                py,
                c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
replacement = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,ZWRpdGVk'}
kwargs = {'logger': logger, 'document': document}
replaced_kwargs = {'logger': logger, 'document': replacement, 'pages': [0]}
",
            );
            let (mut logging, step) = begin(py, &locals, true);
            assert!(awaits_deployment_hook(&step));
            let step = logging
                .resume(py, Ok(local(&locals, "replaced_kwargs").unbind()))
                .unwrap();
            locals.set_item("prepared", arguments(py, step)).unwrap();
            run(
                py,
                &locals,
                c"
assert prepared['document'] is replacement
assert prepared['pages'] is replaced_kwargs['pages']
assert prepared['litellm_logging_obj'] is logger
assert 'litellm_logging_obj' not in replaced_kwargs
",
            );
        });
    }

    #[rstest]
    #[case::synchronous(false)]
    #[case::asynchronous(true)]
    fn a_keyword_the_bridge_never_reads_reaches_every_reader_as_the_callers_object(
        #[case] asynchronous: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(
                py,
                c"
opaque = object()
hooked = []
logger.hooks = {'pre': lambda kwargs: hooked.append(kwargs['vendor_extension']) or kwargs}
kwargs = {'logger': logger, 'vendor_extension': opaque}
",
            );
            let (mut logging, step) = begin(py, &locals, asynchronous);
            let step = match step {
                LifecycleStep::Await(hook_result) => logging.resume(py, Ok(hook_result)).unwrap(),
                step => step,
            };
            locals.set_item("prepared", arguments(py, step)).unwrap();
            locals.set_item("asynchronous", asynchronous).unwrap();
            run(
                py,
                &locals,
                c"
assert prepared['vendor_extension'] is opaque
assert hooked == ([opaque] if asynchronous else []), hooked
",
            );
        });
    }

    #[test]
    fn response_returned_by_the_post_call_hook_is_finalized_and_returned() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(
                py,
                c"
kwargs = {'logger': logger}
response = object()
replacement = object()
logger.hooks = {'pre': lambda kwargs: kwargs}
",
            );
            let (mut logging, _) = begin(py, &locals, true);
            logging
                .resume(py, Ok(local(&locals, "kwargs").unbind()))
                .unwrap();
            let step = logging
                .after_success(py, local(&locals, "response").unbind(), TIMING)
                .unwrap();
            assert!(awaits_deployment_hook(&step));
            let step = logging
                .resume(py, Ok(local(&locals, "replacement").unbind()))
                .unwrap();
            let LifecycleStep::Response(returned) = step else {
                panic!("expected the finalized response");
            };
            assert!(returned.bind(py).is(local(&locals, "replacement")));
            run(
                py,
                &locals,
                c"
[finalized] = [value for name, value in logger.calls if name == 'finalize']
assert finalized is replacement
",
            );
        });
    }

    #[rstest]
    #[case::pre_call(false)]
    #[case::post_call(true)]
    fn cancelling_a_deployment_hook_ends_the_call_with_that_cancellation(#[case] post_call: bool) {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, c"kwargs = {'logger': logger}\nresponse = object()");
            let (mut logging, _) = begin(py, &locals, true);
            if post_call {
                logging
                    .resume(py, Ok(local(&locals, "kwargs").unbind()))
                    .unwrap();
                logging
                    .after_success(py, local(&locals, "response").unbind(), TIMING)
                    .unwrap();
            }
            let cancellation = CancelledError::new_err("cancelled");
            let cancelled = cancellation.value(py).clone();
            let error = logging.resume(py, Err(cancellation)).err().unwrap();
            assert!(error.value(py).is(&cancelled));
            let names: Vec<String> = local(&locals, "logger")
                .call_method0("names")
                .unwrap()
                .extract()
                .unwrap();
            assert!(!names.iter().any(|name| name.contains("handler")));
        });
    }

    #[rstest]
    #[case::hook_completed(false)]
    #[case::hook_cancelled(true)]
    fn failure_callbacks_run_after_the_failure_hook_however_it_ends(#[case] cancelled: bool) {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(
                py,
                c"kwargs = {'logger': logger}\nfailure = ValueError('provider')",
            );
            let (mut logging, _) = begin(py, &locals, true);
            logging
                .resume(py, Ok(local(&locals, "kwargs").unbind()))
                .unwrap();
            let failure = PyErr::from_value(local(&locals, "failure"));
            let failed = LifecycleEvent::Failed {
                timing: TIMING,
                origin: FailureOrigin::Call,
                error: &failure,
            };
            let step = logging.emit(py, failed).unwrap();
            assert!(awaits_deployment_hook(&step));
            let hook_result = if cancelled {
                Err(CancelledError::new_err("cancelled"))
            } else {
                Ok(py.None())
            };
            assert!(matches!(
                logging.resume(py, hook_result).unwrap(),
                LifecycleStep::Await(_)
            ));
            run(
                py,
                &locals,
                c"
assert logger.names()[-3:] == ['failure_hook', 'failure_handler', 'async_failure_handler'], logger.calls
assert all(value is failure for name, value in logger.calls if name.endswith('_handler'))
",
            );
        });
    }
}

#[cfg(test)]
mod payload_tests {
    use std::ffi::CStr;

    use litellm_auth::SecretValue;
    use litellm_host::event::{MachineEvent, RawResponse, RequestContext, WireRequest};
    use litellm_host_python::{LifecycleEvent, LifecycleStep, PythonLifecycle, to_py};
    use proptest::prelude::*;
    use pyo3::gc::{PyTraverseError, PyVisit};
    use pyo3::prelude::*;
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::LegacyLogging;
    use crate::PythonLogger;
    use crate::test_support::{legacy_call, local, namespace, run};

    /// The payload phases of `Logging` on top of `StubLogger`, with `pre_call` handing the
    /// payload to the case's `on_pre_call`.
    const PAYLOAD_LOGGER: &CStr = c"
class Request:
    pass

class PayloadLogger(StubLogger):
    def update_from_kwargs(self, **update):
        self.update = update

    def pre_call(self, input, api_key, additional_args):
        self.record('pre_call', None)
        self.pre = additional_args
        self.pre_api_key = api_key
        on_pre_call(additional_args)

    def post_call(self, original_response, api_key, additional_args):
        self.record('post_call', None)
        self.post = (original_response, api_key, additional_args)

request = Request()
kwargs = {}
logger = PayloadLogger()
on_pre_call = lambda additional_args: None
check = lambda: None
";

    const DOCUMENT: &str = "data:application/pdf;base64,YWJj";
    const EDITED: &str = "data:application/pdf;base64,ZWRpdGVk";

    fn document(source: &str) -> Value {
        json!({"type": "document_url", "document_url": source})
    }

    fn before_send(script: &CStr, body: Value) -> WireRequest {
        before_send_with_secrets(script, json!({}), body, &[])
    }

    /// Runs `before_send` over `body` for a route whose parameters are `optional_params`, with
    /// the Python objects `script` binds, then delivers the provider's raw response the way the
    /// driver does and runs the script's `check()`.
    fn before_send_with_secrets(
        script: &CStr,
        optional_params: Value,
        body: Value,
        secret_fields: &[&str],
    ) -> WireRequest {
        before_send_bound(&[], script, optional_params, body, secret_fields)
    }

    /// [`before_send_with_secrets`] with `bindings` placed in the namespace before `script` runs.
    fn before_send_bound(
        bindings: &[(&str, &Value)],
        script: &CStr,
        optional_params: Value,
        body: Value,
        secret_fields: &[&str],
    ) -> WireRequest {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, PAYLOAD_LOGGER);
            for &(name, value) in bindings {
                locals.set_item(name, to_py(py, value).unwrap()).unwrap();
            }
            run(py, &locals, script);
            let mut logging = LegacyLogging {
                logger: Some(PythonLogger::new(local(&locals, "logger").unbind())),
                ..legacy_call(py, &locals, false)
            };
            let context = RequestContext {
                model: "model".into(),
                custom_llm_provider: "provider".into(),
                optional_params,
                secret_fields: secret_fields.iter().map(|name| name.to_string()).collect(),
                api_key: Some(SecretValue::new("route-key")),
            };
            let wire = WireRequest {
                url: "https://provider.invalid/ocr".into(),
                headers: vec![("x-route".into(), "route".into())],
                body,
            };
            let (_, step) = send_and_receive(py, &mut logging, wire, &context);
            run(py, &locals, c"check()");
            let LifecycleStep::Wire(wire) = step else {
                panic!("before_send did not hand back the wire request");
            };
            *wire
        })
    }

    /// `before_send` over `wire`, then the provider's raw response the way the driver
    /// delivers it, so `pre_call` and `post_call` have both seen the retained payload.
    fn send_and_receive<'a>(
        py: Python<'_>,
        logging: &'a mut LegacyLogging,
        wire: WireRequest,
        context: &RequestContext,
    ) -> (&'a mut LegacyLogging, LifecycleStep) {
        let step = logging.before_send(py, Box::new(wire), context).unwrap();
        let raw = MachineEvent::ResponseReceived {
            raw: RawResponse {
                body: "raw response".into(),
            },
        };
        assert!(matches!(
            logging.emit(py, LifecycleEvent::Machine(&raw)).unwrap(),
            LifecycleStep::Done
        ));
        (logging, step)
    }

    fn route_context() -> RequestContext {
        RequestContext {
            model: "model".into(),
            custom_llm_provider: "provider".into(),
            optional_params: json!({}),
            secret_fields: vec![],
            api_key: Some(SecretValue::new("route-key")),
        }
    }

    fn route_wire() -> WireRequest {
        WireRequest {
            url: "https://provider.invalid/ocr".into(),
            headers: vec![("x-route".into(), "route".into())],
            body: json!({}),
        }
    }

    /// A Python object owning one `LegacyLogging`, so the interpreter's collector sees the
    /// edges the adapter reports and clears them the way the driver's `Execution` does.
    #[pyclass(weakref)]
    struct Retained {
        logging: Option<LegacyLogging>,
    }

    #[pymethods]
    impl Retained {
        fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
            match &self.logging {
                Some(logging) => logging.traverse(&visit),
                None => Ok(()),
            }
        }

        fn __clear__(slf: &Bound<'_, Self>) {
            drop(slf.borrow_mut().logging.take());
        }
    }

    #[test]
    fn a_cycle_through_the_retained_headers_is_collected() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, PAYLOAD_LOGGER);
            let mut logging = LegacyLogging {
                logger: Some(PythonLogger::new(local(&locals, "logger").unbind())),
                ..legacy_call(py, &locals, false)
            };
            send_and_receive(py, &mut logging, route_wire(), &route_context());
            let retained = Py::new(
                py,
                Retained {
                    logging: Some(logging),
                },
            )
            .unwrap();
            locals.set_item("retained", retained).unwrap();
            run(
                py,
                &locals,
                c"
import gc
import weakref

logger.post[2]['headers']['owner'] = retained
logger.pre = logger.post = None
reference = weakref.ref(retained)
del retained
gc.collect()
assert reference() is None
",
            );
        });
    }

    #[test]
    fn close_releases_the_retained_headers() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, PAYLOAD_LOGGER);
            let mut logging = LegacyLogging {
                logger: Some(PythonLogger::new(local(&locals, "logger").unbind())),
                ..legacy_call(py, &locals, false)
            };
            send_and_receive(py, &mut logging, route_wire(), &route_context());
            run(
                py,
                &locals,
                c"
import weakref

class Sentinel:
    pass

sentinel = Sentinel()
logger.post[2]['headers']['sentinel'] = sentinel
logger.pre = logger.post = None
reference = weakref.ref(sentinel)
del sentinel
assert reference() is not None
",
            );
            logging.close(py);
            run(py, &locals, c"assert reference() is None");
        });
    }

    #[rstest]
    #[case::caller_keyword(c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
pages = [0]
kwargs = {'document': document, 'pages': pages}
observed = []
on_pre_call = lambda args: observed.append(
    (args['complete_input_dict']['document'] is document, args['complete_input_dict']['pages'] is pages)
)
def check():
    assert observed == [(True, True)], observed
")]
    #[case::request_attribute_behind_an_omitted_keyword(c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
pages = [0]
request.document = document
kwargs = {'pages': pages}
observed = []
on_pre_call = lambda args: observed.append(
    (args['complete_input_dict']['document'] is document, args['complete_input_dict']['pages'] is pages)
)
def check():
    assert observed == [(True, True)], observed
")]
    fn passthrough_keys_reach_pre_call_as_the_callers_own_objects(#[case] script: &CStr) {
        let body = json!({"model": "model", "document": document(DOCUMENT), "pages": [0]});
        let wire = before_send(script, body.clone());
        assert_eq!(wire.body, body);
    }

    #[test]
    fn pre_call_edit_of_a_passthrough_object_reaches_the_caller_and_the_wire() {
        let wire = before_send(
            c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
kwargs = {'document': document}
def on_pre_call(args):
    args['complete_input_dict']['document']['document_url'] = 'data:application/pdf;base64,ZWRpdGVk'
def check():
    assert document['document_url'] == 'data:application/pdf;base64,ZWRpdGVk'
",
            json!({"document": document(DOCUMENT)}),
        );
        assert_eq!(wire.body["document"], document(EDITED));
    }

    #[test]
    fn a_body_key_the_route_rewrote_is_not_the_callers_object() {
        let wire = before_send(
            c"
document = {'type': 'document_url', 'document_url': 'https://example.invalid/scan.pdf'}
kwargs = {'document': document}
observed = []
def on_pre_call(args):
    observed.append(args['complete_input_dict']['document'] is document)
    args['complete_input_dict']['document']['document_name'] = 'edited.pdf'
def check():
    assert observed == [False], observed
    assert document == {'type': 'document_url', 'document_url': 'https://example.invalid/scan.pdf'}
",
            json!({"document": document(DOCUMENT)}),
        );
        assert_eq!(
            wire.body["document"],
            json!({"type": "document_url", "document_url": DOCUMENT, "document_name": "edited.pdf"})
        );
    }

    #[test]
    fn a_caller_value_with_no_json_form_is_left_out_of_realiasing() {
        let body = json!({"pages": [0]});
        let wire = before_send(
            c"
opaque = object()
kwargs = {'pages': opaque}
observed = []
on_pre_call = lambda args: observed.append(args['complete_input_dict']['pages'])
def check():
    assert observed == [[0]], observed
",
            body.clone(),
        );
        assert_eq!(wire.body, body);
    }

    #[rstest]
    #[case::body(
        c"
def on_pre_call(args):
    args['complete_input_dict'] = {'replacement': True}
"
    )]
    #[case::headers(
        c"
def on_pre_call(args):
    args['headers'] = {'x-replacement': 'yes'}
"
    )]
    fn rebinding_the_payload_envelope_does_not_reach_the_wire(#[case] script: &CStr) {
        let body = json!({"document": document(DOCUMENT)});
        let wire = before_send(script, body.clone());
        assert_eq!(wire.body, body);
        assert_eq!(wire.headers, [("x-route".to_string(), "route".to_string())]);
    }

    #[test]
    fn pre_call_header_edit_reaches_the_wire() {
        let wire = before_send(
            c"
def on_pre_call(args):
    args['headers']['x-callback'] = 'edited'
",
            json!({}),
        );
        assert_eq!(
            wire.headers,
            [
                ("x-route".to_string(), "route".to_string()),
                ("x-callback".to_string(), "edited".to_string()),
            ]
        );
    }

    #[test]
    fn pre_call_receives_the_wire_request_and_the_logger_its_redacted_request() {
        let body = json!({"model": "model", "document": document(DOCUMENT)});
        before_send_with_secrets(
            c"
logger_fn = lambda *args: None
kwargs = {
    'litellm_call_id': 'call-1',
    'client_secret': 'shh',
    'proxy_server_request': {'body': {}},
    'logger_fn': logger_fn,
    'litellm_request_debug': True,
    'ocr_cost_per_page': 0.05,
}
observed = []
on_pre_call = observed.append
def check():
    [args] = observed
    assert args['api_base'] == 'https://provider.invalid/ocr', args
    assert args['complete_input_dict'] == {
        'model': 'model',
        'document': {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'},
    }, args
    update = logger.update
    assert update['model'] == 'model' and update['custom_llm_provider'] == 'provider', update
    assert update['litellm_params']['litellm_call_id'] == 'call-1', update
    assert update['litellm_params']['api_base'] == 'https://provider.invalid/ocr', update
    assert update['litellm_params']['logger_fn'] is logger_fn, update
    assert update['litellm_params']['litellm_request_debug'] is True, update
    assert update['litellm_params']['ocr_cost_per_page'] == 0.05, update
    assert update['kwargs']['client_secret'] == '****', update
    assert 'proxy_server_request' not in update['kwargs'], update
    assert update['optional_params']['client_secret'] == '****', update
",
            json!({"client_secret": "shh"}),
            body,
            &["client_secret"],
        );
    }

    #[rstest]
    #[case::added_key(
        c"
def on_pre_call(args):
    args['complete_input_dict']['include_image_base64'] = True
",
        json!({"document": document(DOCUMENT), "include_image_base64": true})
    )]
    #[case::replaced_document(
        c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
kwargs = {'document': document}
def on_pre_call(args):
    args['complete_input_dict']['document'] = {
        'type': 'document_url', 'document_url': 'data:application/pdf;base64,ZWRpdGVk'
    }
def check():
    assert document['document_url'] == 'data:application/pdf;base64,YWJj', document
",
        json!({"document": document(EDITED)})
    )]
    #[case::retained_body_edited_after_rebinding(
        c"
def on_pre_call(args):
    retained = args['complete_input_dict']
    args['complete_input_dict'] = {'rebound': True}
    retained['include_image_base64'] = True
",
        json!({"document": document(DOCUMENT), "include_image_base64": true})
    )]
    fn pre_call_body_edits_reach_the_wire(#[case] script: &CStr, #[case] expected: Value) {
        let body = json!({"document": document(DOCUMENT)});
        let wire = before_send(script, body);
        assert_eq!(wire.body, expected);
    }

    #[test]
    fn retained_headers_edited_after_rebinding_reach_the_wire() {
        let wire = before_send(
            c"
def on_pre_call(args):
    retained = args['headers']
    args['headers'] = {'x-rebound': 'rebound'}
    retained['x-retained'] = 'sent'
",
            json!({}),
        );
        assert_eq!(
            wire.headers,
            [
                ("x-route".to_string(), "route".to_string()),
                ("x-retained".to_string(), "sent".to_string()),
            ]
        );
    }

    #[test]
    fn post_call_receives_the_raw_response_the_route_key_and_the_body_and_headers_pre_call_saw() {
        before_send(
            c"
def check():
    original_response, api_key, additional_args = logger.post
    assert original_response == 'raw response', original_response
    assert api_key == logger.pre_api_key == 'route-key', (api_key, logger.pre_api_key)
    assert additional_args == {
        'complete_input_dict': logger.pre['complete_input_dict'],
        'headers': logger.pre['headers'],
    }, additional_args
    assert additional_args['complete_input_dict'] is logger.pre['complete_input_dict']
    assert additional_args['headers'] is logger.pre['headers']
",
            json!({"document": document(DOCUMENT)}),
        );
    }

    #[test]
    fn every_request_runs_the_full_pre_call_and_post_call() {
        let wire = before_send(
            c"
def on_pre_call(args):
    args['complete_input_dict']['include_image_base64'] = True
def check():
    assert logger.names() == ['pre_call', 'post_call'], logger.calls
",
            json!({"document": document(DOCUMENT)}),
        );
        assert_eq!(
            wire.body,
            json!({"document": document(DOCUMENT), "include_image_base64": true})
        );
    }

    /// What one pre-call callback does to the payload it is handed.
    #[derive(Clone, Debug)]
    enum Edit {
        Nothing,
        Set(String, Value),
        Remove(String),
        Rebind(Value),
        RebindThenSetRetained(String, Value),
    }

    impl Edit {
        fn script(&self) -> Value {
            match self {
                Self::Nothing => json!({"kind": "nothing"}),
                Self::Set(key, value) => json!({"kind": "set", "key": key, "value": value}),
                Self::Remove(key) => json!({"kind": "remove", "key": key}),
                Self::Rebind(value) => json!({"kind": "rebind", "value": value}),
                Self::RebindThenSetRetained(key, value) => {
                    json!({"kind": "rebind_then_set_retained", "key": key, "value": value})
                }
            }
        }

        /// The legacy contract: the provider is sent the body object `pre_call` received, as
        /// the callback left it. Rebinding the envelope's key points the envelope elsewhere and
        /// leaves that object alone.
        fn sent(&self, body: &Map<String, Value>) -> Value {
            let mut sent = body.clone();
            match self {
                Self::Nothing | Self::Rebind(_) => {}
                Self::Set(key, value) | Self::RebindThenSetRetained(key, value) => {
                    sent.insert(key.clone(), value.clone());
                }
                Self::Remove(key) => {
                    sent.remove(key);
                }
            }
            Value::Object(sent)
        }
    }

    /// How the caller's keyword for a body key relates to what the route sends under it.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Caller {
        PassedUnchanged,
        RewrittenByTheRoute,
        NotPassed,
    }

    const MODEL: &CStr = c"
aliased = {}
def on_pre_call(args):
    body = args['complete_input_dict']
    aliased.update({name: body[name] is kwargs[name] for name in unchanged})
    kind = edit['kind']
    if kind == 'set':
        body[edit['key']] = edit['value']
    elif kind == 'remove':
        body.pop(edit['key'], None)
    elif kind == 'rebind':
        args['complete_input_dict'] = edit['value']
    elif kind == 'rebind_then_set_retained':
        args['complete_input_dict'] = {}
        body[edit['key']] = edit['value']
def check():
    assert aliased == {name: True for name in unchanged}, aliased
    assert logger.names() == ['pre_call', 'post_call'], logger.calls
";

    fn json_value() -> impl Strategy<Value = Value> {
        let leaf = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::from),
            any::<i64>().prop_map(Value::from),
            any::<f64>()
                .prop_filter("JSON has no NaN or infinity", |number| number.is_finite())
                .prop_map(Value::from),
            ".{0,8}".prop_map(Value::from),
        ];
        leaf.prop_recursive(3, 24, 4, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..4).prop_map(Value::from),
                prop::collection::btree_map(key(), inner, 0..4)
                    .prop_map(|fields| Value::Object(fields.into_iter().collect())),
            ]
        })
    }

    fn key() -> impl Strategy<Value = String> {
        "[a-z]{1,6}"
    }

    fn caller() -> impl Strategy<Value = Caller> {
        prop_oneof![
            Just(Caller::PassedUnchanged),
            Just(Caller::RewrittenByTheRoute),
            Just(Caller::NotPassed),
        ]
    }

    fn edit() -> impl Strategy<Value = Edit> {
        prop_oneof![
            Just(Edit::Nothing),
            (key(), json_value()).prop_map(|(key, value)| Edit::Set(key, value)),
            key().prop_map(Edit::Remove),
            json_value().prop_map(Edit::Rebind),
            (key(), json_value()).prop_map(|(key, value)| Edit::RebindThenSetRetained(key, value)),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// For any body, any caller keywords and any callback edit: every keyword the route
        /// sends unchanged reaches `pre_call` as the caller's own object, and the provider is
        /// sent exactly what the model says, so a callback that edits nothing changes nothing.
        #[test]
        fn the_wire_is_the_body_pre_call_received_as_the_callback_left_it(
            fields in prop::collection::btree_map(key(), (json_value(), caller()), 0..5),
            edit in edit(),
        ) {
            let body: Map<String, Value> = fields
                .iter()
                .map(|(name, (value, _))| (name.clone(), value.clone()))
                .collect();
            let kwargs: Map<String, Value> = fields
                .iter()
                .filter_map(|(name, (value, caller))| match caller {
                    Caller::PassedUnchanged => Some((name.clone(), value.clone())),
                    Caller::RewrittenByTheRoute => Some((name.clone(), json!([value]))),
                    Caller::NotPassed => None,
                })
                .collect();
            let unchanged: Value = fields
                .iter()
                .filter(|(_, (_, caller))| *caller == Caller::PassedUnchanged)
                .map(|(name, _)| Value::from(name.clone()))
                .collect();

            let wire = before_send_bound(
                &[
                    ("kwargs", &Value::Object(kwargs)),
                    ("unchanged", &unchanged),
                    ("edit", &edit.script()),
                ],
                MODEL,
                json!({}),
                Value::Object(body.clone()),
                &[],
            );

            prop_assert_eq!(wire.body, edit.sent(&body));
            prop_assert_eq!(wire.headers, [("x-route".to_string(), "route".to_string())]);
        }
    }
}

#[cfg(test)]
mod terminal_tests {
    use std::ffi::CStr;

    use litellm_host::event::{FailureOrigin, Timing};
    use litellm_host_python::{LifecycleEvent, LifecycleStep, PythonLifecycle};
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::exceptions::asyncio::CancelledError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use rstest::rstest;

    use super::LegacyLogging;
    use crate::PythonLogger;
    use crate::test_support::{legacy_call, local, namespace, run};

    const TIMING: Timing = Timing {
        start_time: 0.0,
        end_time: 1.0,
    };

    fn logged(py: Python<'_>, locals: &Bound<'_, PyDict>, asynchronous: bool) -> LegacyLogging {
        LegacyLogging {
            logger: Some(PythonLogger::new(local(locals, "logger").unbind())),
            ..legacy_call(py, locals, asynchronous)
        }
    }

    fn succeed(
        py: Python<'_>,
        locals: &Bound<'_, PyDict>,
        logging: &mut LegacyLogging,
    ) -> LifecycleStep {
        let response = local(locals, "response").unbind();
        logging
            .emit(
                py,
                LifecycleEvent::Succeeded {
                    timing: TIMING,
                    response: &response,
                },
            )
            .unwrap()
    }

    fn fail(
        py: Python<'_>,
        locals: &Bound<'_, PyDict>,
        logging: &mut LegacyLogging,
    ) -> LifecycleStep {
        let failure = PyErr::from_value(local(locals, "failure"));
        logging
            .emit(
                py,
                LifecycleEvent::Failed {
                    timing: TIMING,
                    origin: FailureOrigin::Host,
                    error: &failure,
                },
            )
            .unwrap()
    }

    #[rstest]
    #[case::sync_listened(false, c"", &["submit"])]
    #[case::async_listened(
        true,
        c"",
        &["async_success_handler", "enqueued", "sync_success_for_async_call"]
    )]
    #[case::async_deferred(true, c"logger._defer_async_logging = True", &["sync_success_for_async_call"])]
    #[case::async_with_fallbacks(true, c"kwargs = {'fallbacks': ['other']}", &["sync_success_for_async_call"])]
    fn success_reaches_the_logging_handlers(
        #[case] asynchronous: bool,
        #[case] script: &CStr,
        #[case] expected: &[&str],
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, c"response = object()");
            run(py, &locals, script);
            let mut logging = logged(py, &locals, asynchronous);
            assert!(matches!(
                succeed(py, &locals, &mut logging),
                LifecycleStep::Done
            ));
            let names: Vec<String> = local(&locals, "logger")
                .call_method0("names")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(names, expected);
            run(
                py,
                &locals,
                c"
assert all(value is response for name, value in logger.calls if name.endswith('_handler'))
assert hasattr(logger, '_native_pending_logging') == getattr(logger, '_defer_async_logging', False)
",
            );
        });
    }

    #[rstest]
    #[case::synchronous(false, &["failure_handler"])]
    #[case::asynchronous(true, &[])]
    fn internal_calls_skip_failure_callbacks_only_when_asynchronous(
        #[case] asynchronous: bool,
        #[case] expected: &[&str],
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, c"failure = ValueError('provider')");
            let mut logging = LegacyLogging {
                internal: true,
                ..logged(py, &locals, asynchronous)
            };
            assert!(matches!(
                fail(py, &locals, &mut logging),
                LifecycleStep::Done
            ));
            let names: Vec<String> = local(&locals, "logger")
                .call_method0("names")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(names, expected);
        });
    }

    #[test]
    fn internal_async_calls_skip_the_async_success_fan_out() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, c"response = object()");
            let mut logging = LegacyLogging {
                internal: true,
                ..logged(py, &locals, true)
            };
            succeed(py, &locals, &mut logging);
            run(
                py,
                &locals,
                c"assert logger.names() == ['sync_success_for_async_call'], logger.calls",
            );
        });
    }

    #[test]
    fn a_failing_success_callback_is_reported_without_replacing_the_response() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(
                py,
                c"
response = object()
failure = ValueError('terminal diagnostic')

class FailingLogger(StubLogger):
    def handle_sync_success_callbacks_for_async_calls(self, *args):
        raise failure

logger = FailingLogger()
",
            );
            let mut logging = logged(py, &locals, true);
            assert!(matches!(
                succeed(py, &locals, &mut logging),
                LifecycleStep::Done
            ));
            assert!(
                logging
                    .response
                    .as_ref()
                    .unwrap()
                    .bind(py)
                    .is(local(&locals, "response"))
            );
            run(py, &locals, c"assert unraisable_from(logger) == [failure]");
        });
    }

    #[rstest]
    #[case::sync_listened(false, c"", &["failure_handler"])]
    #[case::async_listened(true, c"", &["failure_handler", "async_failure_handler"])]
    fn failure_reaches_the_logging_handlers(
        #[case] asynchronous: bool,
        #[case] script: &CStr,
        #[case] expected: &[&str],
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, c"failure = ValueError('provider')");
            run(py, &locals, script);
            let mut logging = logged(py, &locals, asynchronous);
            let step = fail(py, &locals, &mut logging);
            let awaits_async_handler = expected.contains(&"async_failure_handler");
            assert_eq!(
                matches!(step, LifecycleStep::Await(_)),
                awaits_async_handler
            );
            let names: Vec<String> = local(&locals, "logger")
                .call_method0("names")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(names, expected);
            run(
                py,
                &locals,
                c"assert all(value is failure for name, value in logger.calls if name.endswith('_handler'))",
            );
        });
    }

    #[test]
    fn a_failing_sync_failure_callback_keeps_the_error_and_still_runs_the_async_family() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(
                py,
                c"
failure = ValueError('selected')

class FailingLogger(StubLogger):
    def failure_handler(self, error, trace, start, end):
        self.record('failure_handler', error)
        raise RuntimeError('handler failed')

logger = FailingLogger()
",
            );
            let mut logging = logged(py, &locals, true);
            assert!(matches!(
                fail(py, &locals, &mut logging),
                LifecycleStep::Await(_)
            ));
            assert!(
                logging
                    .error
                    .as_ref()
                    .unwrap()
                    .bind(py)
                    .is(local(&locals, "failure"))
            );
            run(
                py,
                &locals,
                c"assert logger.names() == ['failure_handler', 'async_failure_handler'], logger.calls",
            );
        });
    }

    #[rstest]
    #[case::completed(None, true)]
    #[case::handler_error(Some(false), true)]
    #[case::cancelled(Some(true), false)]
    fn the_async_failure_handler_ends_the_call_unless_it_was_cancelled(
        #[case] error: Option<bool>,
        #[case] done: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, c"failure = ValueError('provider')");
            let mut logging = logged(py, &locals, true);
            fail(py, &locals, &mut logging);
            let result = match error {
                None => Ok(py.None()),
                Some(false) => Err(PyRuntimeError::new_err("handler failed")),
                Some(true) => Err(CancelledError::new_err("cancelled")),
            };
            let expected = result.as_ref().err().map(|error| error.value(py).clone());
            match logging.resume(py, result) {
                Ok(step) => assert!(done && matches!(step, LifecycleStep::Done)),
                Err(propagated) => {
                    assert!(!done);
                    assert!(propagated.value(py).is(expected.unwrap()));
                }
            }
        });
    }

    #[test]
    fn closing_restores_the_correlation_context_once() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, c"");
            let mut logging = logged(py, &locals, true);
            logging.close(py);
            logging.close(py);
            run(
                py,
                &locals,
                c"assert logger.names() == ['restore'], logger.calls",
            );
        });
    }
}

#[cfg(test)]
mod pre_request_hooks_tests {
    use std::ffi::CStr;

    use litellm_host::event::PublicRequest;
    use litellm_host_python::{LifecycleStep, PythonLifecycle};
    use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::LegacyLogging;
    use crate::test_support::{legacy_call, local, namespace, run};

    const CALL: &CStr = c"
tool = {'name': 'original_tool', 'input_schema': {'type': 'object'}}
tools = [tool]
messages = [{'role': 'user', 'content': 'hi'}]
kwargs = {'logger': logger, 'model': 'claude', 'messages': messages, 'tools': tools, 'max_tokens': 16}
logger.hooks = {'pre': lambda kwargs: kwargs}
";

    fn object(value: Value) -> Map<String, Value> {
        let Value::Object(map) = value else {
            panic!("expected a json object, got {value}");
        };
        map
    }

    fn request() -> PublicRequest {
        PublicRequest {
            model: "claude".into(),
            custom_llm_provider: "anthropic".into(),
            messages: json!([{"role": "user", "content": "hi"}]),
            params: object(json!({
                "tools": [{"name": "original_tool", "input_schema": {"type": "object"}}],
                "max_tokens": 16,
            })),
            fields: &["tools", "stream", "tool_choice", "max_tokens"],
        }
    }

    /// A call past `begin` and its deployment hook, where the route projects it.
    fn prepared(py: Python<'_>, locals: &Bound<'_, PyDict>, asynchronous: bool) -> LegacyLogging {
        let mut logging = legacy_call(py, locals, asynchronous);
        let kwargs = local(locals, "kwargs")
            .cast_into::<PyDict>()
            .unwrap()
            .unbind();
        match logging.begin(py, kwargs, 0.0).unwrap() {
            LifecycleStep::Await(hook_result) => {
                logging.resume(py, Ok(hook_result)).unwrap();
            }
            LifecycleStep::Arguments(_) => {}
            _ => panic!("begin ends in the prepared arguments"),
        }
        logging
    }

    fn awaiting(step: LifecycleStep) -> Py<PyAny> {
        let LifecycleStep::Await(awaitable) = step else {
            panic!("expected the hooks' awaitable");
        };
        awaitable
    }

    fn params(step: LifecycleStep) -> Map<String, Value> {
        let LifecycleStep::Params(params) = step else {
            panic!("expected the answered params");
        };
        params
    }

    fn expose_view(py: Python<'_>, locals: &Bound<'_, PyDict>, logging: &LegacyLogging) {
        locals
            .set_item("view", logging.call.kwargs().clone_ref(py))
            .unwrap();
    }

    #[rstest]
    #[case::synchronous(false)]
    #[case::asynchronous(true)]
    fn pre_request_hooks_run_only_for_asynchronous_calls(#[case] asynchronous: bool) {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, CALL);
            let mut logging = prepared(py, &locals, asynchronous);
            let step = logging.pre_request(py, request()).unwrap();
            let names: Vec<String> = local(&locals, "logger")
                .call_method0("names")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(names.contains(&"pre_request".to_string()), asynchronous);
            match step {
                LifecycleStep::Await(_) => {
                    assert!(asynchronous, "a sync call has no loop to await on")
                }
                LifecycleStep::Params(answered) => {
                    assert!(!asynchronous, "an async call awaits its hooks");
                    assert_eq!(answered, request().params);
                }
                _ => panic!("pre_request awaits the hooks or answers unchanged"),
            }
        });
    }

    #[test]
    fn the_hooks_receive_the_callers_objects_and_the_resolved_provider() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, CALL);
            let mut logging = prepared(py, &locals, true);
            awaiting(logging.pre_request(py, request()).unwrap());
            run(
                py,
                &locals,
                c"
[(model, hooked_messages, hooked)] = [value for name, value in logger.calls if name == 'pre_request']
assert model == 'claude'
assert hooked_messages is messages
assert hooked['tools'] is tools
assert hooked['litellm_params'] == {'custom_llm_provider': 'anthropic'}
assert hooked['litellm_logging_obj'] is logger
assert hooked['max_tokens'] == 16
",
            );
        });
    }

    #[test]
    fn edits_the_hooks_return_reach_the_route_and_every_later_reader() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, CALL);
            run(
                py,
                &locals,
                c"
renamed = [{'name': 'renamed_tool', 'input_schema': {'type': 'object'}}]
logger.hooks['pre_request'] = lambda model, messages, kwargs: {
    **kwargs, 'tools': renamed, 'stream': False, 'max_agentic_loops': 3
}
",
            );
            let mut logging = prepared(py, &locals, true);
            let returned = awaiting(logging.pre_request(py, request()).unwrap());
            let answered = params(logging.resume(py, Ok(returned)).unwrap());
            assert_eq!(
                answered,
                object(json!({
                    "tools": [{"name": "renamed_tool", "input_schema": {"type": "object"}}],
                    "stream": false,
                    "max_tokens": 16,
                }))
            );
            expose_view(py, &locals, &logging);
            run(
                py,
                &locals,
                c"
assert view['tools'] is renamed
assert view['stream'] is False
assert view['max_agentic_loops'] == 3
assert 'litellm_params' not in view
assert view['litellm_logging_obj'] is logger
",
            );
        });
    }

    #[test]
    fn hooks_that_return_none_change_nothing() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, CALL);
            run(
                py,
                &locals,
                c"logger.hooks['pre_request'] = lambda model, messages, kwargs: None",
            );
            let mut logging = prepared(py, &locals, true);
            let returned = awaiting(logging.pre_request(py, request()).unwrap());
            let answered = params(logging.resume(py, Ok(returned)).unwrap());
            assert_eq!(answered, request().params);
            expose_view(py, &locals, &logging);
            run(
                py,
                &locals,
                c"
assert view['tools'] is tools
assert 'stream' not in view
assert 'litellm_params' not in view
",
            );
        });
    }

    #[test]
    fn a_raising_hook_fails_the_call_with_its_error() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, CALL);
            let mut logging = prepared(py, &locals, true);
            awaiting(logging.pre_request(py, request()).unwrap());
            let failure = PyRuntimeError::new_err("hook refused");
            let raised = failure.value(py).clone();
            let error = logging.resume(py, Err(failure)).err().unwrap();
            assert!(error.value(py).is(&raised));
        });
    }
}
