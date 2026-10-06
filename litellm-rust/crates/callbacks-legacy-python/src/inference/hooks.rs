use litellm_host::{
    hooks::CallHooks,
    interceptors::{RawResponse, RequestContext, WireRequest},
    lifecycle::{ExecutionEvent, FailureOrigin, Timing},
};
use litellm_host_python::{HookStep, PythonCallEvent, PythonRuntime};
use pyo3::{prelude::*, types::PyDict};

use super::InferenceAdapter;
use crate::mapping::{CallbackMapping, Dispatch, Firing};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceBoundary {
    PrepareArguments,
    BeforeProviderRequest,
    AfterProviderResponse,
    TransformResponse,
    Succeeded,
    Failed,
    StreamOpened,
    StreamChunk,
}

struct Binding<H> {
    boundary: InferenceBoundary,
    invoke: H,
    callbacks: &'static [&'static str],
}

impl<H> Binding<H> {
    fn mappings(&self) -> impl Iterator<Item = CallbackMapping> {
        self.callbacks.iter().map(|callback| CallbackMapping {
            callback,
            dispatch: Dispatch::Inference(Firing::Native(self.boundary)),
        })
    }
}

type Step<T> = PyResult<HookStep<InferenceAdapter, T>>;
type Prepare = fn(&mut InferenceAdapter, Python<'_>, Py<PyDict>, f64) -> Step<Py<PyDict>>;
type Before = fn(
    &mut InferenceAdapter,
    Python<'_>,
    Box<WireRequest>,
    &RequestContext,
) -> Step<Box<WireRequest>>;
type After = fn(&mut InferenceAdapter, Python<'_>, &RawResponse) -> Step<()>;
type Transform = fn(&mut InferenceAdapter, Python<'_>, Py<PyAny>, Timing) -> Step<Py<PyAny>>;
type Success = fn(&mut InferenceAdapter, Python<'_>, Timing, &Py<PyAny>) -> Step<()>;
type Failure = fn(&mut InferenceAdapter, Python<'_>, Timing, FailureOrigin, &PyErr) -> Step<()>;
type Open = fn(&mut InferenceAdapter, Python<'_>, &Py<PyAny>) -> PyResult<()>;
type Chunk = fn(&mut InferenceAdapter, Python<'_>, &Py<PyAny>) -> PyResult<()>;

const PREPARE: Binding<Prepare> = Binding {
    boundary: InferenceBoundary::PrepareArguments,
    invoke: InferenceAdapter::prepare_call,
    callbacks: &["async_pre_call_deployment_hook"],
};

const BEFORE: Binding<Before> = Binding {
    boundary: InferenceBoundary::BeforeProviderRequest,
    invoke: InferenceAdapter::pre_call,
    callbacks: &["log_pre_api_call", "log_input_event"],
};

const AFTER: Binding<After> = Binding {
    boundary: InferenceBoundary::AfterProviderResponse,
    invoke: InferenceAdapter::post_call,
    callbacks: &["log_post_api_call"],
};

const TRANSFORM: Binding<Transform> = Binding {
    boundary: InferenceBoundary::TransformResponse,
    invoke: InferenceAdapter::transform_public_response,
    callbacks: &["async_post_call_success_deployment_hook"],
};

const SUCCESS: Binding<Success> = Binding {
    boundary: InferenceBoundary::Succeeded,
    invoke: InferenceAdapter::succeeded,
    callbacks: &[
        "log_success_event",
        "async_log_success_event",
        "logging_hook",
        "async_logging_hook",
        "redact_standard_logging_payload_from_model_call_details",
        "log_event",
        "async_log_event",
    ],
};

const FAILURE: Binding<Failure> = Binding {
    boundary: InferenceBoundary::Failed,
    invoke: InferenceAdapter::failed,
    callbacks: &[
        "async_post_call_failure_deployment_hook",
        "log_failure_event",
        "async_log_failure_event",
        "log_model_group_rate_limit_error",
        "log_event",
        "async_log_event",
    ],
};

const OPEN: Binding<Open> = Binding {
    boundary: InferenceBoundary::StreamOpened,
    invoke: InferenceAdapter::stream_opened,
    callbacks: &[],
};

const CHUNK: Binding<Chunk> = Binding {
    boundary: InferenceBoundary::StreamChunk,
    invoke: InferenceAdapter::stream_chunk,
    callbacks: &[],
};

pub(crate) fn mappings() -> impl Iterator<Item = CallbackMapping> {
    PREPARE
        .mappings()
        .chain(BEFORE.mappings())
        .chain(AFTER.mappings())
        .chain(TRANSFORM.mappings())
        .chain(SUCCESS.mappings())
        .chain(FAILURE.mappings())
        .chain(OPEN.mappings())
        .chain(CHUNK.mappings())
}

impl CallHooks<PythonRuntime> for InferenceAdapter {
    fn prepare_arguments(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> Step<Py<PyDict>> {
        (PREPARE.invoke)(self, py, arguments, started_at)
    }

    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
        self.adopt_arguments(py, arguments);
        Ok(())
    }

    fn before_provider_request(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> Step<Box<WireRequest>> {
        (BEFORE.invoke)(self, py, wire, context)
    }

    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> Step<Py<PyAny>> {
        (TRANSFORM.invoke)(self, py, response, timing)
    }

    fn on_event(&mut self, py: Python<'_>, event: PythonCallEvent<'_>) -> Step<()> {
        match event {
            PythonCallEvent::Started { .. } | PythonCallEvent::Cancelled { .. } => {
                Ok(HookStep::Ready(()))
            }
            PythonCallEvent::Execution(ExecutionEvent::ResultReady { facts }) => {
                self.result_ready(py, &facts)
            }
            PythonCallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }) => {
                (AFTER.invoke)(self, py, raw)
            }
            PythonCallEvent::Succeeded { timing, response } => {
                (SUCCESS.invoke)(self, py, timing, response)
            }
            PythonCallEvent::Failed {
                timing,
                origin,
                error,
            } => (FAILURE.invoke)(self, py, timing, origin, error),
        }
    }

    fn on_stream_open(&mut self, py: Python<'_>, head: &Py<PyAny>) -> PyResult<()> {
        (OPEN.invoke)(self, py, head)
    }

    fn on_stream_chunk(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()> {
        (CHUNK.invoke)(self, py, chunk)
    }
}
