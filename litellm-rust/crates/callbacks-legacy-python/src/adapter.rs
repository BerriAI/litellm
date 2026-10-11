//! The legacy `Logging` contract as one adapter: every event and interception the driver
//! raises is answered with the same `Logging` calls, in the same order, as the Python
//! `@client` path makes them. Deployment hooks and the async failure handler are awaited on
//! the caller's task through the [`PythonTask`].

use crate::LoggingOperation;
use litellm_host_python::{PythonOwned, PythonTask};

use litellm_host::{
    interceptors::{RawResponse, RequestContext, WireRequest},
    lifecycle::{FailureOrigin, Timing, epoch_seconds},
};
use litellm_host_python::{from_py, missing_state, to_py};
use pyo3::{
    exceptions::{PyBaseException, PyException},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDateTime, PyDict, PyList},
};
use serde_json::Value;

use crate::{
    DeploymentHooks, LegacyCallbacks, PublicCall, PythonLogger,
    deferred::{PendingLogging, PendingSuccess},
    finalize, is_internal_call,
    python::Streaming,
    setup,
};

/// What a stream iterator keeps for its end-of-stream logging.
struct DeliveredStream {
    chunks: Py<PyList>,
    first_chunk: Option<Py<PyAny>>,
}

struct LoggedRequest {
    body: Py<PyDict>,
    headers: Py<PyDict>,
    context: RequestContext,
}

pub struct LegacyLogging {
    operation: LoggingOperation,
    call: PublicCall,
    logger: Option<PythonLogger>,
    start: Py<PyAny>,
    end: Option<Py<PyAny>>,
    response: Option<Py<PyAny>>,
    error: Option<Py<PyBaseException>>,
    request: Option<LoggedRequest>,
    stream: Option<DeliveredStream>,
    asynchronous: bool,
    internal: bool,
    cache_key: Option<String>,
}

enum FailurePath {
    Stream,
    DeploymentHook(Py<PyAny>),
    Dispatch,
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
        operation: LoggingOperation,
        call: PublicCall,
        asynchronous: bool,
    ) -> Self {
        Self {
            operation,
            call,
            logger: None,
            start: py.None(),
            end: None,
            response: None,
            error: None,
            request: None,
            stream: None,
            asynchronous,
            internal: false,
            cache_key: None,
        }
    }

    fn call_type(&self) -> &'static str {
        match (self.operation, self.asynchronous) {
            (LoggingOperation::Completion, false) => "completion",
            (LoggingOperation::Completion, true) => "acompletion",
            (LoggingOperation::Responses, false) => "responses",
            (LoggingOperation::Responses, true) => "aresponses",
            (LoggingOperation::Messages, _) => "anthropic_messages",
            (LoggingOperation::Ocr, false) => "ocr",
            (LoggingOperation::Ocr, true) => "aocr",
        }
    }

    fn input_description(&self) -> &'static str {
        match self.operation {
            LoggingOperation::Completion => "Chat completions",
            LoggingOperation::Responses => "Responses",
            LoggingOperation::Messages => "Messages",
            LoggingOperation::Ocr => "OCR document processing",
        }
    }

    /// Whether the operation's Python stream iterator logs the stream itself, which the
    /// shim's `stream_success` and `stream_failure` arms reproduce.
    fn logs_streams(&self) -> bool {
        match self.operation {
            LoggingOperation::Messages | LoggingOperation::Responses => true,
            LoggingOperation::Completion | LoggingOperation::Ocr => false,
        }
    }

    pub(crate) fn adopt_arguments(&mut self, py: Python<'_>, arguments: &Py<PyDict>) {
        self.call.set_kwargs(arguments.clone_ref(py));
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
    fn prepare(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let prepared = self.call.kwargs().bind(py).copy()?;
        prepared.set_item("litellm_logging_obj", self.logger()?.object(py))?;
        self.call.set_kwargs(prepared.unbind());
        Ok(self.call.kwargs().clone_ref(py))
    }

    fn finalize(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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
            .map(|response| response.clone_ref(py))
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
        if !self.logs_streams() {
            return Err(missing_state());
        }
        let billed = Streaming::Success.call(
            py,
            (
                <&str>::from(self.operation),
                logger.object(py),
                &self.request.as_ref().map_or_else(
                    || self.call.kwargs().clone_ref(py),
                    |request| request.body.clone_ref(py),
                ),
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

    /// Awaits a scheduled failure handler. Its own errors never replace the selected
    /// failure; a cancellation does end the call.
    async fn await_failure_handler(&self, task: &PythonTask, awaitable: Py<PyAny>) -> PyResult<()> {
        task.settle(Ok(awaitable)).await.map(|_| ())
    }

    /// A failure after the stream reached the caller bills the delivered chunks as
    /// partial usage. The sync path has no loop to schedule that on, so it falls back to
    /// the plain failure handler.
    async fn stream_failure(&mut self, task: &PythonTask) -> PyResult<()> {
        let delivered = self.logger.is_some()
            && self.error.is_some()
            && self.stream.is_some()
            && self.logs_streams();
        if !delivered {
            return Ok(());
        }
        if !self.asynchronous {
            return self.dispatch_failure(task).await;
        }
        let scheduled = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let (Some(logger), Some(error), Some(stream)) = (&self.logger, &self.error, &self.stream)
            else {
                return Err(missing_state());
            };
            Streaming::Failure
                .call(
                    py,
                    (
                        <&str>::from(self.operation),
                        logger.object(py),
                        &self.request.as_ref().map_or_else(
                            || self.call.kwargs().clone_ref(py),
                            |request| request.body.clone_ref(py),
                        ),
                        &stream.chunks,
                        error,
                    ),
                )
                .map(Bound::unbind)
        });
        match scheduled {
            Ok(awaitable) => self.await_failure_handler(task, awaitable).await,
            Err(failure) if Python::attach(|py| is_cancellation(py, &failure)) => Err(failure),
            Err(_) => Ok(()),
        }
    }

    /// The sync failure handler, then the async one for async calls. Ordinary handler
    /// errors never replace the selected failure or suppress the other family; a
    /// cancellation does end the call.
    async fn dispatch_failure(&mut self, task: &PythonTask) -> PyResult<()> {
        let scheduled = Python::attach(|py| -> PyResult<Option<Py<PyAny>>> {
            let (Some(logger), Some(error)) = (&self.logger, &self.error) else {
                return Ok(None);
            };
            if self.asynchronous && self.internal {
                return Ok(None);
            }
            if let Err(failure) = logger.failure(py, error, &self.start, &self.end, false)
                && is_cancellation(py, &failure)
            {
                return Err(failure);
            }
            if !self.asynchronous {
                return Ok(None);
            }
            match logger.failure(py, error, &self.start, &self.end, true) {
                Ok(awaitable) => Ok(awaitable),
                Err(failure) if is_cancellation(py, &failure) => Err(failure),
                Err(_) => Ok(None),
            }
        })?;
        match scheduled {
            Some(awaitable) => self.await_failure_handler(task, awaitable).await,
            None => Ok(()),
        }
    }
}

impl LegacyLogging {
    pub(crate) async fn prepare_call(
        &mut self,
        task: &PythonTask,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<Py<PyDict>> {
        let hook = Python::attach(|py| -> PyResult<Option<Py<PyAny>>> {
            self.call.set_kwargs(arguments);
            self.start = datetime(py, started_at)?;
            self.internal = is_internal_call(py)?;
            let result = setup(
                py,
                self.call_type(),
                self.call.args(),
                self.call.kwargs(),
                &self.start,
                self.asynchronous,
            )?;
            self.logger = Some(result.logger()?);
            self.call.set_kwargs(result.kwargs()?);
            if !self.runs_deployment_hooks() {
                return Ok(None);
            }
            DeploymentHooks::before_call(py, self.call.kwargs(), self.call_type()).map(Some)
        })?;
        if let Some(hook) = hook {
            let result = task.settle(Ok(hook)).await?;
            Python::attach(|py| -> PyResult<()> {
                self.call
                    .set_kwargs(result?.into_bound(py).cast_into::<PyDict>()?.unbind());
                Ok(())
            })?;
        }
        Python::attach(|py| self.prepare(py))
    }

    pub(crate) fn pre_call(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<Box<WireRequest>> {
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
        self.request = Some(LoggedRequest {
            body: body.clone().unbind(),
            headers: headers.clone().unbind(),
            context: context.clone(),
        });
        self.logger()?.pre_call(
            py,
            self.input_description(),
            context.api_key.as_ref().map(|api_key| api_key.expose()),
            &body,
            &headers,
            &wire.url,
        )?;
        let headers = headers
            .iter()
            .map(|(name, value)| Ok((name.extract::<String>()?, value.extract::<String>()?)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Box::new(WireRequest {
            body: from_py(&body)?,
            headers,
            ..*wire
        }))
    }

    pub(crate) async fn transform_public_response(
        &mut self,
        task: &PythonTask,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<Py<PyAny>> {
        let hook = Python::attach(|py| -> PyResult<Option<Py<PyAny>>> {
            self.end = Some(datetime(py, timing.end_time)?);
            self.response = Some(response);
            if !self.runs_deployment_hooks() {
                return Ok(None);
            }
            DeploymentHooks::after_success(py, self.call.kwargs(), &self.response, self.call_type())
                .map(Some)
        })?;
        if let Some(hook) = hook {
            let result = task.settle(Ok(hook)).await?;
            self.response = Some(result?);
        }
        Python::attach(|py| self.finalize(py))
    }

    pub(crate) fn result_ready(
        &mut self,
        py: Python<'_>,
        facts: &litellm_host::interceptors::ExecutionFacts,
    ) -> PyResult<()> {
        use litellm_host::interceptors::ResultSource;

        let logger = self.logger()?.object(py);
        let params = logger
            .getattr("litellm_params")?
            .cast_into::<PyDict>()?
            .copy()?;
        params.set_item("custom_llm_provider", &facts.provider.provider)?;
        crate::python::Logging::Update.call(
            py,
            (
                &logger,
                self.call.kwargs(),
                &facts.provider.model,
                logger.getattr("optional_params")?,
                params,
                &facts.provider.provider,
            ),
        )?;
        let details = logger.getattr("model_call_details")?;
        self.cache_key = match &facts.source {
            ResultSource::Provider => None,
            ResultSource::Cache { key } => Some(key.clone()),
        };
        details.set_item("cache_hit", self.cache_key.is_some())?;
        details.set_item("cache_key", self.cache_key.as_deref())?;
        Ok(())
    }

    pub(crate) fn post_call(&mut self, py: Python<'_>, raw: &RawResponse) -> PyResult<()> {
        let api_key = self
            .request
            .as_ref()
            .and_then(|request| request.context.api_key.as_ref())
            .map(|api_key| api_key.expose());
        self.logger()?.post_call(
            py,
            &raw.body,
            api_key,
            self.request.as_ref().map(|request| &request.body),
            self.request.as_ref().map(|request| &request.headers),
        )
    }

    pub(crate) fn succeeded(
        &mut self,
        py: Python<'_>,
        timing: Timing,
        response: &Py<PyAny>,
    ) -> PyResult<()> {
        self.end = Some(datetime(py, timing.end_time)?);
        self.response = Some(response.clone_ref(py));
        match &self.stream {
            Some(stream) => self.stream_success(py, stream),
            None => self.dispatch_success(py),
        }
    }

    /// The failure callbacks run after the deployment failure hook however that hook ends,
    /// and a cancellation raised there still ends the call once they have run.
    pub(crate) async fn failed(
        &mut self,
        task: &PythonTask,
        timing: Timing,
        origin: FailureOrigin,
        error: &PyErr,
    ) -> PyResult<()> {
        let path = Python::attach(|py| -> PyResult<FailurePath> {
            self.end = Some(datetime(py, timing.end_time)?);
            self.error = Some(error.clone_ref(py).into_value(py));
            if self.stream.is_some() {
                return Ok(FailurePath::Stream);
            }
            if origin == FailureOrigin::Call
                && self.logger.is_some()
                && self.runs_deployment_hooks()
            {
                let error = self.error.as_ref().ok_or_else(missing_state)?;
                return DeploymentHooks::after_failure(
                    py,
                    self.call.kwargs(),
                    error,
                    self.call_type(),
                )
                .map(FailurePath::DeploymentHook);
            }
            Ok(FailurePath::Dispatch)
        })?;
        match path {
            FailurePath::Stream => self.stream_failure(task).await,
            FailurePath::Dispatch => self.dispatch_failure(task).await,
            FailurePath::DeploymentHook(hook) => {
                let settled = task.settle(Ok(hook)).await;
                self.dispatch_failure(task).await?;
                settled.map(|_| ())
            }
        }
    }

    pub(crate) fn stream_opened(&mut self, py: Python<'_>, head: &Py<PyAny>) -> PyResult<()> {
        if !self.logs_streams() {
            return Err(missing_state());
        }
        if let Some(key) = &self.cache_key {
            head.bind(py).set_item("cache_key", key)?;
            head.bind(py).set_item("cache_hit", true)?;
        }
        Streaming::Opened.call(py, (self.logger()?.object(py), head.bind(py)))?;
        self.stream = Some(DeliveredStream {
            chunks: PyList::empty(py).unbind(),
            first_chunk: None,
        });
        Ok(())
    }

    pub(crate) fn stream_chunk(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()> {
        let stream = self.stream.as_mut().ok_or_else(missing_state)?;
        if stream.first_chunk.is_none() {
            stream.first_chunk = Some(datetime(py, epoch_seconds())?);
        }
        stream.chunks.bind(py).append(chunk)
    }
}

impl PythonOwned for LegacyLogging {
    fn close(&mut self, py: Python<'_>) {
        if let Some(logger) = self.logger.take()
            && let Err(error) = logger.restore_context(py)
        {
            error.write_unraisable(py, None);
        }
        self.request = None;
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
        if let Some(request) = &self.request {
            visit.call(&request.body)?;
            visit.call(&request.headers)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod deployment_hooks_tests {
    use std::ffi::CStr;

    use litellm_host::hooks::CallHooks;
    use litellm_host::lifecycle::{FailureOrigin, Timing};
    use litellm_host_python::{PythonCallEvent, PythonTask};
    use pyo3::exceptions::asyncio::CancelledError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use rstest::rstest;

    use super::LegacyLogging;
    use crate::test_support::{Driven, drive, legacy_call, local, namespace, run};

    const CALL: &CStr = c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
kwargs = {'logger': logger, 'document': document}
";

    const TIMING: Timing = Timing {
        start_time: 0.0,
        end_time: 1.0,
    };

    /// Prepares the namespace's `kwargs`, answering the deployment pre-call hook, when it
    /// runs, with the namespace's `prepared_by_hook` or the hook's own awaitable.
    fn begin<'py>(
        py: Python<'py>,
        locals: &Bound<'py, PyDict>,
        asynchronous: bool,
    ) -> (LegacyLogging, PythonTask, Driven<PyResult<Py<PyDict>>>) {
        let mut logging = legacy_call(py, locals, asynchronous);
        let task = PythonTask::new(asynchronous);
        let kwargs = local(locals, "kwargs")
            .cast_into::<PyDict>()
            .unwrap()
            .unbind();
        let driven = drive(
            py,
            &task,
            logging.prepare_arguments(&task, kwargs, 0.0),
            |py, _, awaitable| {
                Ok(locals
                    .get_item("prepared_by_hook")
                    .unwrap()
                    .map_or_else(|| awaitable.clone_ref(py), Bound::unbind))
            },
        );
        (logging, task, driven)
    }

    #[rstest]
    #[case::sync_completion(crate::LoggingOperation::Completion, false, "completion")]
    #[case::async_completion(crate::LoggingOperation::Completion, true, "acompletion")]
    #[case::sync_responses(crate::LoggingOperation::Responses, false, "responses")]
    #[case::async_responses(crate::LoggingOperation::Responses, true, "aresponses")]
    #[case::sync_messages(crate::LoggingOperation::Messages, false, "anthropic_messages")]
    #[case::async_messages(crate::LoggingOperation::Messages, true, "anthropic_messages")]
    #[case::sync_ocr(crate::LoggingOperation::Ocr, false, "ocr")]
    #[case::async_ocr(crate::LoggingOperation::Ocr, true, "aocr")]
    fn operation_selects_the_legacy_setup_and_deployment_hook_contract(
        #[case] operation: crate::LoggingOperation,
        #[case] asynchronous: bool,
        #[case] expected: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, CALL);
            run(py, &locals, c"prepared_by_hook = kwargs");
            let mut logging = LegacyLogging {
                operation,
                ..legacy_call(py, &locals, asynchronous)
            };
            let task = PythonTask::new(asynchronous);
            let kwargs = local(&locals, "kwargs")
                .cast_into::<PyDict>()
                .unwrap()
                .unbind();
            let driven = drive(
                py,
                &task,
                logging.prepare_arguments(&task, kwargs.clone_ref(py), 0.0),
                |_, _, _| Ok(kwargs.clone_ref(py).into_any()),
            );
            driven.outcome.unwrap();
            assert_eq!(driven.awaited.len(), usize::from(asynchronous));
            locals.set_item("expected", expected).unwrap();
            locals.set_item("asynchronous", asynchronous).unwrap();
            run(
                py,
                &locals,
                c"
assert logger.setup_call_type == expected
if asynchronous:
    assert logger.calls == [('pre_hook', expected)]
",
            );
        });
    }

    #[rstest]
    #[case::synchronous(false)]
    #[case::asynchronous(true)]
    fn deployment_pre_call_hook_runs_only_for_asynchronous_calls(#[case] asynchronous: bool) {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, CALL);
            run(py, &locals, c"prepared_by_hook = kwargs");
            let (_, _, driven) = begin(py, &locals, asynchronous);
            driven.outcome.unwrap();
            assert_eq!(driven.awaited.len(), usize::from(asynchronous));
            let names: Vec<String> = local(&locals, "logger")
                .call_method0("names")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(names.contains(&"pre_hook".to_string()), asynchronous);
        });
    }

    #[rstest::rstest]
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
prepared_by_hook = replaced_kwargs
",
            );
            let (_, _, driven) = begin(py, &locals, true);
            assert_eq!(driven.awaited.len(), 1);
            locals
                .set_item("prepared", driven.outcome.unwrap())
                .unwrap();
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

    /// The hook returns its own result as the awaitable here, so what it hands back is what
    /// the call prepares.
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
            let (_, _, driven) = begin(py, &locals, asynchronous);
            locals
                .set_item("prepared", driven.outcome.unwrap())
                .unwrap();
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

    #[rstest::rstest]
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
prepared_by_hook = kwargs
",
            );
            let (mut logging, task, driven) = begin(py, &locals, true);
            driven.outcome.unwrap();
            let replacement = local(&locals, "replacement").unbind();
            let driven = drive(
                py,
                &task,
                logging.transform_response(&task, local(&locals, "response").unbind(), TIMING),
                |py, _, _| Ok(replacement.clone_ref(py)),
            );
            assert_eq!(driven.awaited.len(), 1);
            assert!(driven.outcome.unwrap().bind(py).is(&replacement));
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
            let cancellation = CancelledError::new_err("cancelled");
            let cancelled = cancellation.value(py).clone().unbind();
            let cancel =
                move |py: Python<'_>, _: usize, _: &Py<PyAny>| Err(cancellation.clone_ref(py));
            let error = if post_call {
                run(py, &locals, c"prepared_by_hook = kwargs");
                let (mut logging, task, driven) = begin(py, &locals, true);
                driven.outcome.unwrap();
                drive(
                    py,
                    &task,
                    logging.transform_response(&task, local(&locals, "response").unbind(), TIMING),
                    cancel,
                )
                .outcome
                .unwrap_err()
            } else {
                let mut logging = legacy_call(py, &locals, true);
                let task = PythonTask::new(true);
                let kwargs = local(&locals, "kwargs")
                    .cast_into::<PyDict>()
                    .unwrap()
                    .unbind();
                drive(
                    py,
                    &task,
                    logging.prepare_arguments(&task, kwargs, 0.0),
                    cancel,
                )
                .outcome
                .unwrap_err()
            };
            assert!(error.value(py).is(cancelled.bind(py)));
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
                c"kwargs = {'logger': logger}\nfailure = ValueError('provider')\nprepared_by_hook = kwargs",
            );
            let (mut logging, task, driven) = begin(py, &locals, true);
            driven.outcome.unwrap();
            let failure = PyErr::from_value(local(&locals, "failure"));
            let failed = PythonCallEvent::Failed {
                timing: TIMING,
                origin: FailureOrigin::Call,
                error: &failure,
            };
            let driven = drive(
                py,
                &task,
                logging.on_event(&task, failed),
                |py, index, _| {
                    if index == 0 && cancelled {
                        Err(CancelledError::new_err("cancelled"))
                    } else {
                        Ok(py.None())
                    }
                },
            );
            assert_eq!(driven.awaited.len(), 2);
            match driven.outcome {
                Ok(()) => assert!(!cancelled),
                Err(error) => {
                    assert!(cancelled);
                    assert!(error.is_instance_of::<CancelledError>(py));
                }
            }
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
    use litellm_host::hooks::CallHooks;
    use litellm_host::interceptors::{RawResponse, RequestContext, WireRequest};
    use litellm_host::lifecycle::ExecutionEvent;
    use litellm_host_python::{PythonCallEvent, PythonOwned, PythonTask, to_py};
    use proptest::prelude::*;
    use pyo3::gc::{PyTraverseError, PyVisit};
    use pyo3::prelude::*;
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::LegacyLogging;
    use crate::PythonLogger;
    use crate::test_support::{finish, legacy_call, local, namespace, run};

    /// The payload phases of `Logging` on top of `StubLogger`, with `pre_call` handing the
    /// payload to the case's `on_pre_call`.
    const PAYLOAD_LOGGER: &CStr = c"
class PayloadLogger(StubLogger):
    def update_from_kwargs(self, **update):
        self.update = update

    def pre_call(self, input, api_key, additional_args):
        self.record('pre_call', None)
        self.pre = additional_args
        self.pre_input = input
        self.pre_api_key = api_key
        on_pre_call(additional_args)

    def post_call(self, original_response, api_key, additional_args):
        self.record('post_call', None)
        self.post = (original_response, api_key, additional_args)

bound = {}
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

    fn before_provider_request(script: &CStr, body: Value) -> WireRequest {
        before_send_with_secrets(script, json!({}), body, &[])
    }

    /// Runs `before_provider_request` over `body` for a route whose parameters are `optional_params`, with
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
            let wire = send_and_receive(py, &PythonTask::new(false), &mut logging, wire, &context);
            run(py, &locals, c"check()");
            *wire
        })
    }

    /// `before_provider_request` over `wire`, then the provider's raw response the way the driver
    /// delivers it, so `pre_call` and `post_call` have both seen the retained payload.
    fn send_and_receive(
        py: Python<'_>,
        task: &PythonTask,
        logging: &mut LegacyLogging,
        wire: WireRequest,
        context: &RequestContext,
    ) -> Box<WireRequest> {
        let wire = finish(
            py,
            task,
            logging.before_provider_request(task, Box::new(wire), context),
        )
        .unwrap();
        let raw = RawResponse {
            body: "raw response".into(),
        };
        finish(
            py,
            task,
            logging.on_event(
                task,
                PythonCallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw: &raw }),
            ),
        )
        .unwrap();
        wire
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

    #[rstest]
    #[case::completion(crate::LoggingOperation::Completion, "Chat completions")]
    #[case::responses(crate::LoggingOperation::Responses, "Responses")]
    #[case::messages(crate::LoggingOperation::Messages, "Messages")]
    #[case::ocr(crate::LoggingOperation::Ocr, "OCR document processing")]
    fn prepared_arguments_replace_the_legacy_view_without_losing_callback_aliases(
        #[case] operation: crate::LoggingOperation,
        #[case] description: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, PAYLOAD_LOGGER);
            run(
                py,
                &locals,
                c"
original = [0]
replacement = [1]
kwargs['pages'] = original
prepared = {'pages': replacement}
",
            );
            let mut logging = LegacyLogging {
                operation,
                logger: Some(PythonLogger::new(local(&locals, "logger").unbind())),
                ..legacy_call(py, &locals, false)
            };
            let prepared = local(&locals, "prepared")
                .cast_into::<pyo3::types::PyDict>()
                .unwrap()
                .unbind();
            let task = PythonTask::new(false);
            finish(py, &task, logging.arguments_prepared(&task, &prepared)).unwrap();
            let wire = WireRequest {
                body: json!({"pages": [1]}),
                ..route_wire()
            };
            send_and_receive(py, &task, &mut logging, wire, &route_context());
            locals.set_item("description", description).unwrap();
            run(
                py,
                &locals,
                c"
assert logger.pre['complete_input_dict']['pages'] is replacement
assert logger.pre_input == description
assert original == [0]
",
            );
        });
    }

    #[rstest::rstest]
    fn a_cycle_through_the_retained_headers_is_collected() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, PAYLOAD_LOGGER);
            let mut logging = LegacyLogging {
                logger: Some(PythonLogger::new(local(&locals, "logger").unbind())),
                ..legacy_call(py, &locals, false)
            };
            send_and_receive(
                py,
                &PythonTask::new(false),
                &mut logging,
                route_wire(),
                &route_context(),
            );
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

    #[rstest::rstest]
    fn close_releases_the_retained_headers() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, PAYLOAD_LOGGER);
            let mut logging = LegacyLogging {
                logger: Some(PythonLogger::new(local(&locals, "logger").unbind())),
                ..legacy_call(py, &locals, false)
            };
            send_and_receive(
                py,
                &PythonTask::new(false),
                &mut logging,
                route_wire(),
                &route_context(),
            );
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
    #[case::bound_value_behind_an_omitted_keyword(c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
pages = [0]
bound['document'] = document
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
        let wire = before_provider_request(script, body.clone());
        assert_eq!(wire.body, body);
    }

    #[rstest::rstest]
    fn pre_call_edit_of_a_passthrough_object_reaches_the_caller_and_the_wire() {
        let wire = before_provider_request(
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

    #[rstest::rstest]
    fn a_body_key_the_route_rewrote_is_not_the_callers_object() {
        let wire = before_provider_request(
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

    #[rstest::rstest]
    fn a_caller_value_with_no_json_form_is_left_out_of_realiasing() {
        let body = json!({"pages": [0]});
        let wire = before_provider_request(
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
        let wire = before_provider_request(script, body.clone());
        assert_eq!(wire.body, body);
        assert_eq!(wire.headers, [("x-route".to_string(), "route".to_string())]);
    }

    #[rstest::rstest]
    fn pre_call_header_edit_reaches_the_wire() {
        let wire = before_provider_request(
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

    #[rstest::rstest]
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
        let wire = before_provider_request(script, body);
        assert_eq!(wire.body, expected);
    }

    #[rstest::rstest]
    fn retained_headers_edited_after_rebinding_reach_the_wire() {
        let wire = before_provider_request(
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

    #[rstest::rstest]
    fn post_call_receives_the_raw_response_the_route_key_and_the_body_and_headers_pre_call_saw() {
        before_provider_request(
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

    #[rstest::rstest]
    fn every_request_runs_the_full_pre_call_and_post_call() {
        let wire = before_provider_request(
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
        #[rstest::rstest]
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

    use litellm_host::hooks::CallHooks;
    use litellm_host::lifecycle::{FailureOrigin, Timing};
    use litellm_host_python::{PythonCallEvent, PythonOwned, PythonTask};
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::exceptions::asyncio::CancelledError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use rstest::rstest;

    use super::LegacyLogging;
    use crate::PythonLogger;
    use crate::test_support::{Driven, drive, finish, legacy_call, local, namespace, run};

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

    fn succeed(py: Python<'_>, locals: &Bound<'_, PyDict>, logging: &mut LegacyLogging) {
        let task = PythonTask::new(logging.asynchronous);
        let response = local(locals, "response").unbind();
        finish(
            py,
            &task,
            logging.on_event(
                &task,
                PythonCallEvent::Succeeded {
                    timing: TIMING,
                    response: &response,
                },
            ),
        )
        .unwrap();
    }

    /// Fails the call, answering the async failure handler with `answer`.
    fn fail(
        py: Python<'_>,
        locals: &Bound<'_, PyDict>,
        logging: &mut LegacyLogging,
        answer: impl FnMut(Python<'_>, usize, &Py<PyAny>) -> PyResult<Py<PyAny>>,
    ) -> Driven<PyResult<()>> {
        let task = PythonTask::new(logging.asynchronous);
        let failure = PyErr::from_value(local(locals, "failure"));
        drive(
            py,
            &task,
            logging.on_event(
                &task,
                PythonCallEvent::Failed {
                    timing: TIMING,
                    origin: FailureOrigin::Host,
                    error: &failure,
                },
            ),
            answer,
        )
    }

    fn fail_completing(
        py: Python<'_>,
        locals: &Bound<'_, PyDict>,
        logging: &mut LegacyLogging,
    ) -> Driven<PyResult<()>> {
        fail(py, locals, logging, |py, _, _| Ok(py.None()))
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
            succeed(py, &locals, &mut logging);
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
    fn dropped_observations_preserve_deferred_success_and_response_identity() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(
                py,
                c"response = object()\nlogger._defer_async_logging = True",
            );
            let mut logging = logged(py, &locals, true);
            let task = PythonTask::new(true);
            let response = local(&locals, "response").unbind();
            let event = PythonCallEvent::Succeeded {
                timing: TIMING,
                response: &response,
            };
            let (sender, receiver) = litellm_host::observation::observation_channel(
                std::num::NonZeroUsize::new(1).unwrap(),
            );
            drop(receiver);
            sender.emit(event.snapshot());
            finish(py, &task, logging.on_event(&task, event)).unwrap();
            assert_eq!(sender.dropped_events(), 1);
            run(py, &locals, c"
assert logger.names() == ['sync_success_for_async_call'], logger.calls
logger._native_pending_logging.release(True)
logger._native_pending_logging.release(True)
assert logger.names() == ['sync_success_for_async_call', 'async_success_handler', 'enqueued'], logger.calls
assert logger.calls[0][1] is response
assert logger.calls[1][1] is response
");
        });
    }

    #[rstest]
    fn stream_bindings_deliver_collected_chunks_in_order_without_success_fan_out() {
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, c"first = b'first'\nlast = b'last'\nresponse = None\nhead = {'additional_headers': {'request-id': 'req_native'}}");
            let mut logging = LegacyLogging {
                operation: crate::LoggingOperation::Messages,
                ..logged(py, &locals, true)
            };
            let task = PythonTask::new(true);
            finish(
                py,
                &task,
                logging.on_stream_open(&task, &local(&locals, "head").unbind()),
            )
            .unwrap();
            finish(
                py,
                &task,
                logging.on_stream_chunk(&task, &local(&locals, "first").unbind()),
            )
            .unwrap();
            finish(
                py,
                &task,
                logging.on_stream_chunk(&task, &local(&locals, "last").unbind()),
            )
            .unwrap();
            succeed(py, &locals, &mut logging);
            run(
                py,
                &locals,
                c"
assert logger.names() == ['stream_opened', 'stream_success'], logger.calls
assert logger.calls[0][1] is head
chunks = logger.calls[1][1]
assert len(chunks) == 2
assert chunks[0] is first
assert chunks[1] is last
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
            let driven = fail_completing(py, &locals, &mut logging);
            driven.outcome.unwrap();
            assert!(driven.awaited.is_empty());
            let names: Vec<String> = local(&locals, "logger")
                .call_method0("names")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(names, expected);
        });
    }

    #[rstest::rstest]
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

    #[rstest::rstest]
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
            succeed(py, &locals, &mut logging);
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
            let driven = fail_completing(py, &locals, &mut logging);
            driven.outcome.unwrap();
            let awaits_async_handler = expected.contains(&"async_failure_handler");
            assert_eq!(driven.awaited.len(), usize::from(awaits_async_handler));
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

    #[rstest::rstest]
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
            let driven = fail_completing(py, &locals, &mut logging);
            driven.outcome.unwrap();
            assert_eq!(driven.awaited.len(), 1);
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
            let raised = match error {
                None => None,
                Some(false) => Some(PyRuntimeError::new_err("handler failed")),
                Some(true) => Some(CancelledError::new_err("cancelled")),
            };
            let expected = raised
                .as_ref()
                .map(|error| error.value(py).clone().unbind());
            let driven = fail(py, &locals, &mut logging, |py, _, _| match &raised {
                None => Ok(py.None()),
                Some(error) => Err(error.clone_ref(py)),
            });
            assert_eq!(driven.awaited.len(), 1);
            match driven.outcome {
                Ok(()) => assert!(done),
                Err(propagated) => {
                    assert!(!done);
                    assert!(propagated.value(py).is(expected.unwrap().bind(py)));
                }
            }
        });
    }

    #[rstest::rstest]
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
