use litellm_host::{
    hooks::CallHooks,
    interceptors::{RawResponse, RequestContext, WireRequest},
    lifecycle::{ExecutionEvent, FailureOrigin, Timing},
};
use litellm_host_python::{HookStep, PythonCallEvent, PythonRuntime};
use pyo3::{prelude::*, types::PyDict};

use crate::LegacyLogging;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallBoundary {
    PrepareArguments,
    BeforeProviderRequest,
    AfterProviderResponse,
    TransformResponse,
    Succeeded,
    Failed,
    StreamOpened,
    StreamChunk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispatch {
    Call(CallBoundary),
    Python(&'static str),
    DeclarationOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallbackMapping {
    pub callback: &'static str,
    pub dispatch: Dispatch,
}

struct Binding<H> {
    boundary: CallBoundary,
    invoke: H,
    callbacks: &'static [&'static str],
}

impl<H> Binding<H> {
    fn mappings(&self) -> impl Iterator<Item = CallbackMapping> {
        self.callbacks.iter().map(|callback| CallbackMapping {
            callback,
            dispatch: Dispatch::Call(self.boundary),
        })
    }
}

type Step<T> = PyResult<HookStep<LegacyLogging, T>>;
type Prepare = fn(&mut LegacyLogging, Python<'_>, Py<PyDict>, f64) -> Step<Py<PyDict>>;
type Before =
    fn(&mut LegacyLogging, Python<'_>, Box<WireRequest>, &RequestContext) -> Step<Box<WireRequest>>;
type After = fn(&mut LegacyLogging, Python<'_>, &RawResponse) -> Step<()>;
type Transform = fn(&mut LegacyLogging, Python<'_>, Py<PyAny>, Timing) -> Step<Py<PyAny>>;
type Success = fn(&mut LegacyLogging, Python<'_>, Timing, &Py<PyAny>) -> Step<()>;
type Failure = fn(&mut LegacyLogging, Python<'_>, Timing, FailureOrigin, &PyErr) -> Step<()>;
type Open = fn(&mut LegacyLogging, Python<'_>) -> PyResult<()>;
type Chunk = fn(&mut LegacyLogging, Python<'_>, &Py<PyAny>) -> PyResult<()>;

const PREPARE: Binding<Prepare> = Binding {
    boundary: CallBoundary::PrepareArguments,
    invoke: LegacyLogging::prepare_call,
    callbacks: &["async_pre_call_deployment_hook"],
};

const BEFORE: Binding<Before> = Binding {
    boundary: CallBoundary::BeforeProviderRequest,
    invoke: LegacyLogging::pre_call,
    callbacks: &["log_pre_api_call", "log_input_event"],
};

const AFTER: Binding<After> = Binding {
    boundary: CallBoundary::AfterProviderResponse,
    invoke: LegacyLogging::post_call,
    callbacks: &["log_post_api_call"],
};

const TRANSFORM: Binding<Transform> = Binding {
    boundary: CallBoundary::TransformResponse,
    invoke: LegacyLogging::transform_public_response,
    callbacks: &["async_post_call_success_deployment_hook"],
};

const SUCCESS: Binding<Success> = Binding {
    boundary: CallBoundary::Succeeded,
    invoke: LegacyLogging::succeeded,
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
    boundary: CallBoundary::Failed,
    invoke: LegacyLogging::failed,
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
    boundary: CallBoundary::StreamOpened,
    invoke: LegacyLogging::stream_opened,
    callbacks: &[],
};

const CHUNK: Binding<Chunk> = Binding {
    boundary: CallBoundary::StreamChunk,
    invoke: LegacyLogging::stream_chunk,
    callbacks: &[],
};

pub fn callback_mappings() -> impl Iterator<Item = CallbackMapping> {
    PREPARE
        .mappings()
        .chain(BEFORE.mappings())
        .chain(AFTER.mappings())
        .chain(TRANSFORM.mappings())
        .chain(SUCCESS.mappings())
        .chain(FAILURE.mappings())
        .chain(OPEN.mappings())
        .chain(CHUNK.mappings())
        .chain(PYTHON_CALLBACKS.iter().copied())
}

impl CallHooks<PythonRuntime> for LegacyLogging {
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

    fn on_stream_open(&mut self, py: Python<'_>) -> PyResult<()> {
        (OPEN.invoke)(self, py)
    }

    fn on_stream_chunk(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()> {
        (CHUNK.invoke)(self, py, chunk)
    }
}

macro_rules! python_callbacks {
    ($($dispatch:expr => [$($callback:literal),* $(,)?]),* $(,)?) => {
        const PYTHON_CALLBACKS: &[CallbackMapping] = &[
            $($(CallbackMapping { callback: $callback, dispatch: $dispatch },)*)*
        ];
    };
}

python_callbacks! {
    Dispatch::Python("litellm.router") => [
        "async_pre_routing_hook",
        "async_filter_deployments",
        "pre_call_check",
        "async_pre_call_check",
    ],
    Dispatch::Python("litellm.router_utils.fallback_event_handlers") => [
        "log_success_fallback_event",
        "log_failure_fallback_event",
    ],
    Dispatch::Python("litellm.proxy.utils") => [
        "async_pre_call_hook",
        "async_post_call_response_headers_hook",
        "async_post_call_failure_hook",
        "async_post_call_success_hook",
        "async_moderation_hook",
        "async_post_call_streaming_hook",
        "async_post_call_streaming_iterator_hook",
        "async_filter_listed_models",
    ],
    Dispatch::Python("litellm.litellm_core_utils.litellm_logging") => [
        "async_get_chat_completion_prompt",
        "get_chat_completion_prompt",
        "log_stream_event",
        "async_log_stream_event",
        "async_post_mcp_tool_call_hook",
    ],
    Dispatch::Python("litellm.llms.anthropic.pass_through.messages.handler") => [
        "async_pre_request_hook",
    ],
    Dispatch::Python("litellm.litellm_core_utils.streaming_handler") => [
        "async_post_call_streaming_deployment_hook",
    ],
    Dispatch::Python("litellm.responses.streaming_iterator") => [
        "async_post_call_streaming_deployment_hook",
    ],
    Dispatch::Python("litellm.main") => [
        "translate_completion_input_params",
        "translate_completion_output_params",
        "translate_completion_output_params_streaming",
    ],
    Dispatch::Python("litellm.integrations.argilla") => ["async_dataset_hook"],
    Dispatch::Python("litellm.proxy.management_helpers.audit_logs") => ["async_log_audit_log_event"],
    Dispatch::Python("litellm.llms.custom_httpx.llm_http_handler") => [
        "async_should_run_agentic_loop",
        "async_run_agentic_loop",
        "async_build_agentic_loop_plan",
        "async_post_agentic_loop_response_hook",
        "async_agentic_loop_cleanup_hook",
        "async_should_run_chat_completion_agentic_loop",
        "async_run_chat_completion_agentic_loop",
        "async_build_chat_completion_agentic_loop_plan",
    ],
    Dispatch::Python("litellm.litellm_core_utils.chat_completion_agentic_loop") => [
        "async_should_run_agentic_loop",
        "async_run_agentic_loop",
        "async_build_agentic_loop_plan",
        "async_post_agentic_loop_response_hook",
        "async_agentic_loop_cleanup_hook",
    ],
    Dispatch::Python("litellm.llms.openai.openai") => [
        "async_should_run_chat_completion_agentic_loop",
        "async_run_chat_completion_agentic_loop",
    ],
    Dispatch::Python("litellm.proxy.spend_tracking.cold_storage_handler") => [
        "get_proxy_server_request_from_cold_storage_with_object_key",
    ],
    Dispatch::Python("litellm.integrations.custom_logger") => [
        "truncate_standard_logging_payload_content",
        "redacts_messages_itself",
        "handle_callback_failure",
        "get_callback_env_vars",
    ],
    Dispatch::DeclarationOnly => [
        "async_log_pre_api_call",
        "async_log_input_event",
    ],
}
