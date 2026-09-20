//! The legacy `Logging` contract as one adapter: every event and interception the driver
//! raises is answered with the same `Logging` calls, in the same order, as the Python
//! `@client` path makes them.

use litellm_host::event::{
    FailureOrigin, MachineEvent, RequestContext, Timing, WireRequest, epoch_seconds,
};
use litellm_host_python::{
    LifecycleEvent, LifecycleStep, PythonLifecycle, from_py, missing_state, to_py,
};
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
    finalize, is_internal_call, prepare,
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
    CacheRead,
    CacheWrite,
    StreamCacheWrite,
    DeploymentPreCall,
    DeploymentPostCall,
    DeploymentFailure,
    AsyncFailure,
}

pub struct LegacyLogging {
    cache: Option<crate::caching::Caching>,
    cache_hit: bool,
    accounted: bool,
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
            cache: None,
            cache_hit: false,
            accounted: false,
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

    fn prepare(&mut self, py: Python<'_>) -> PyResult<LifecycleStep> {
        let prepared = prepare(py, self.call.kwargs().bind(py), self.logger()?)?.unbind();
        self.call.set_kwargs(prepared);
        if self.surface.call_type == "anthropic_messages" {
            self.cache = crate::caching::Caching::prepare(
                py,
                &self.call,
                self.surface.call_type,
                self.asynchronous,
            )?;
            if let Some(cache) = &self.cache
                && cache.controls.reads()
            {
                match cache.read(py, self.asynchronous) {
                    Ok(value) if self.asynchronous => {
                        self.pending = Some(Pending::CacheRead);
                        return Ok(LifecycleStep::Await(value));
                    }
                    result => return self.cache_read(py, result),
                }
            }
        }
        self.arguments(py)
    }

    fn arguments(&mut self, py: Python<'_>) -> PyResult<LifecycleStep> {
        if self.surface.call_type == "anthropic_messages" {
            if let Err(error) =
                crate::limits::adjust_max_tokens(py, &self.call, self.surface.call_type)
                && is_cancellation(py, &error)
            {
                return Err(error);
            }
        }
        Ok(LifecycleStep::Arguments(self.call.kwargs().clone_ref(py)))
    }

    fn cache_read(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<LifecycleStep> {
        let decoded = result.and_then(|value| {
            self.cache
                .as_ref()
                .ok_or_else(missing_state)?
                .decode(value.bind(py))
        });
        let response = match decoded {
            Ok(Some(response)) => response,
            Err(error) if is_cancellation(py, &error) => return Err(error),
            _ => return self.arguments(py),
        };
        self.cache_hit = true;
        self.logger()?
            .object(py)
            .getattr("model_call_details")?
            .set_item("cache_hit", true)?;
        let cache = self.cache.as_ref().ok_or_else(missing_state)?;
        self.logger()?
            .object(py)
            .getattr("litellm_params")?
            .set_item("preset_cache_key", &cache.key)?;
        if response
            .get("litellm_cached_anthropic_sse_events")
            .is_some()
        {
            self.call
                .kwargs()
                .bind(py)
                .set_item("_rust_messages_cached_response", to_py(py, &response)?)?;
            return Ok(LifecycleStep::Arguments(self.call.kwargs().clone_ref(py)));
        }
        self.end = Some(datetime(py, epoch_seconds())?);
        self.response = Some(crate::caching::cached_public(py, &response)?);
        self.finalize(py)?;
        Ok(LifecycleStep::Cached(
            self.response
                .as_ref()
                .ok_or_else(missing_state)?
                .clone_ref(py),
        ))
    }

    fn cache_write(&mut self, py: Python<'_>) -> PyResult<LifecycleStep> {
        if let (Some(cache), Some(response)) = (&self.cache, &self.response)
            && cache.controls.writes()
            && !self.cache_hit
        {
            match cache.write(py, response.bind(py), self.asynchronous) {
                Ok(awaitable) if self.asynchronous => {
                    self.pending = Some(Pending::CacheWrite);
                    return Ok(LifecycleStep::Await(awaitable));
                }
                Err(error) if is_cancellation(py, &error) => return Err(error),
                _ => {}
            }
        }
        self.finalize(py)
    }

    fn dispatch_cache_hit(&self, py: Python<'_>) -> PyResult<()> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("result", &self.response)?;
        kwargs.set_item("start_time", &self.start)?;
        kwargs.set_item("end_time", &self.end)?;
        kwargs.set_item("cache_hit", true)?;
        let logger = self.logger()?.object(py);
        if self.asynchronous {
            let coroutine = logger.call_method("async_success_handler", (), Some(&kwargs))?;
            crate::python::Logging::Enqueue.call(py, (coroutine,))?;
        }
        logger.call_method(
            "handle_sync_success_callbacks_for_async_calls",
            (),
            Some(&kwargs),
        )?;
        Ok(())
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

    fn record_usage(&mut self, py: Python<'_>, usage: &Value) -> PyResult<()> {
        if self.accounted || self.surface.call_type != "anthropic_messages" {
            return Ok(());
        }
        self.accounted = true;
        let model = self
            .call
            .lookup(py, "model")?
            .map(|value| value.extract::<String>())
            .transpose()?;
        let info = model
            .map(|model| crate::limits::model_info(py, &model))
            .transpose()?
            .flatten();
        let cost = if self.cache_hit {
            Some(0.0)
        } else {
            info.as_ref()
                .and_then(|info| litellm_core_utils::cost_calculator::messages_cost(info, usage))
        };
        if let Some(cost) = cost {
            let logger = self.logger()?.object(py);
            logger.setattr("_native_response_cost", cost)?;
            logger
                .getattr("model_call_details")?
                .set_item("response_cost", cost)?;
            let sdk = py.import("litellm")?;
            if !self.cache_hit && sdk.getattr("max_budget")?.is_truthy()? {
                let current: f64 = sdk.getattr("_current_cost")?.extract()?;
                sdk.setattr("_current_cost", current + cost)?;
            }
        }
        Ok(())
    }

    fn finish_stream(&mut self, py: Python<'_>) -> PyResult<LifecycleStep> {
        let stream = self.stream.as_ref().ok_or_else(missing_state)?;
        let mut bytes = Vec::new();
        for chunk in stream.chunks.bind(py).iter() {
            bytes.extend_from_slice(chunk.cast::<pyo3::types::PyBytes>()?.as_bytes());
        }
        if let Some(parsed) = litellm_core_utils::messages_stream::MessagesStream::parse(&bytes) {
            self.record_usage(py, &parsed.usage)?;
            if !self.cache_hit
                && let (Some(cache), Some(response)) = (&self.cache, parsed.cached_response)
                && cache.controls.writes()
            {
                let response = to_py(py, &response)?;
                match cache.write(py, response.bind(py), self.asynchronous) {
                    Ok(awaitable) if self.asynchronous => {
                        self.pending = Some(Pending::StreamCacheWrite);
                        return Ok(LifecycleStep::Await(awaitable));
                    }
                    Err(error) if is_cancellation(py, &error) => return Err(error),
                    _ => {}
                }
            }
        }
        self.stream_success(py, self.stream.as_ref().ok_or_else(missing_state)?)?;
        Ok(LifecycleStep::Done)
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
        if self.cache_hit {
            return self.dispatch_cache_hit(py);
        }
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
        if self.runs_deployment_hooks() {
            self.pending = Some(Pending::DeploymentPreCall);
            return Ok(LifecycleStep::Await(DeploymentHooks::before_call(
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
            return Ok(LifecycleStep::Await(DeploymentHooks::after_success(
                py,
                self.call.kwargs(),
                &self.response,
                self.surface.call_type,
            )?));
        }
        self.cache_write(py)
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
                if self.stream.is_some() {
                    return self.finish_stream(py);
                }
                if self.surface.call_type == "anthropic_messages" {
                    let value: Value = from_py(response.bind(py))?;
                    self.record_usage(py, value.get("usage").unwrap_or(&Value::Null))?;
                }
                self.dispatch_success(py)?;
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
                    return Ok(LifecycleStep::Await(DeploymentHooks::after_failure(
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
            Pending::DeploymentPreCall => {
                self.call
                    .set_kwargs(result?.into_bound(py).cast_into::<PyDict>()?.unbind());
                self.prepare(py)
            }
            Pending::DeploymentPostCall => {
                self.response = Some(result?);
                self.cache_write(py)
            }
            Pending::CacheRead => self.cache_read(py, result),
            Pending::CacheWrite => {
                if let Err(error) = result
                    && is_cancellation(py, &error)
                {
                    return Err(error);
                }
                self.finalize(py)
            }
            Pending::StreamCacheWrite => {
                if let Err(error) = result
                    && is_cancellation(py, &error)
                {
                    return Err(error);
                }
                if let Some(stream) = &self.stream {
                    self.stream_success(py, stream)?;
                }
                Ok(LifecycleStep::Done)
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
        self.context = None;
        self.stream = None;
        self.cache = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.call.traverse(visit)?;
        if let Some(cache) = &self.cache {
            cache.traverse(visit)?;
        }
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
        visit.call(&self.body)
    }
}

#[cfg(test)]
#[path = "../tests/deployment_hooks.rs"]
mod deployment_hooks_tests;
#[cfg(test)]
#[path = "../tests/payload.rs"]
mod payload_tests;
#[cfg(test)]
#[path = "../tests/terminal.rs"]
mod terminal_tests;
