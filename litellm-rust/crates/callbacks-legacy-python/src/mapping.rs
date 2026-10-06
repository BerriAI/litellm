use litellm_host::{
    hooks::CallHooks,
    interceptors::{RawResponse, RequestContext, WireRequest},
    lifecycle::{ExecutionEvent, FailureOrigin, Timing},
};
use litellm_host_python::{HookStep, PythonCallEvent, PythonRuntime};
use pyo3::{prelude::*, types::PyDict};

use crate::InferenceAdapter;

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
    SucceededOrFailed,
    NotFiredNativelyYet(&'static [&'static str]),
    DeclarationOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispatch {
    Inference(InferenceBoundary),
    Gateway(&'static str),
    Router(&'static str),
    Management(&'static str),
    HandlerTrait(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallbackMapping {
    pub callback: &'static str,
    pub dispatch: Dispatch,
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
            dispatch: Dispatch::Inference(self.boundary),
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
        .chain(OTHER_CALLBACKS.iter().copied())
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

macro_rules! callbacks {
    ($($dispatch:expr => [$($callback:literal),* $(,)?]),* $(,)?) => {
        const OTHER_CALLBACKS: &[CallbackMapping] = &[
            $($(CallbackMapping { callback: $callback, dispatch: $dispatch },)*)*
        ];
    };
}

callbacks! {
    Dispatch::Inference(InferenceBoundary::SucceededOrFailed) => ["log_event", "async_log_event"],
    Dispatch::Router("litellm.router") => [
        "async_pre_routing_hook",
        "async_filter_deployments",
        "pre_call_check",
        "async_pre_call_check",
    ],
    Dispatch::Router("litellm.router_utils.fallback_event_handlers") => [
        "log_success_fallback_event",
        "log_failure_fallback_event",
    ],
    Dispatch::Gateway("litellm.proxy.utils") => [
        "async_pre_call_hook",
        "async_post_call_response_headers_hook",
        "async_post_call_failure_hook",
        "async_post_call_success_hook",
        "async_moderation_hook",
        "async_post_call_streaming_hook",
        "async_post_call_streaming_iterator_hook",
        "async_filter_listed_models",
        "apply_guardrail",
    ],
    Dispatch::Gateway("litellm.litellm_core_utils.litellm_logging") => [
        "async_post_mcp_tool_call_hook",
    ],
    Dispatch::Inference(InferenceBoundary::NotFiredNativelyYet(&[
        "litellm.litellm_core_utils.litellm_logging",
    ])) => [
        "async_get_chat_completion_prompt",
        "get_chat_completion_prompt",
        "log_stream_event",
        "async_log_stream_event",
    ],
    Dispatch::Inference(InferenceBoundary::NotFiredNativelyYet(&[
        "litellm.llms.anthropic.pass_through.messages.handler",
    ])) => ["async_pre_request_hook"],
    Dispatch::Inference(InferenceBoundary::NotFiredNativelyYet(&[
        "litellm.litellm_core_utils.streaming_handler",
        "litellm.responses.streaming_iterator",
    ])) => ["async_post_call_streaming_deployment_hook"],
    Dispatch::Inference(InferenceBoundary::NotFiredNativelyYet(&["litellm.main"])) => [
        "translate_completion_input_params",
        "translate_completion_output_params",
        "translate_completion_output_params_streaming",
    ],
    Dispatch::Inference(InferenceBoundary::NotFiredNativelyYet(&["litellm.integrations.argilla"])) => [
        "async_dataset_hook",
    ],
    Dispatch::Management("litellm.proxy.management_helpers.audit_logs") => ["async_log_audit_log_event"],
    Dispatch::Inference(InferenceBoundary::NotFiredNativelyYet(&[
        "litellm.llms.custom_httpx.llm_http_handler",
        "litellm.litellm_core_utils.chat_completion_agentic_loop",
    ])) => [
        "async_should_run_agentic_loop",
        "async_run_agentic_loop",
        "async_build_agentic_loop_plan",
        "async_post_agentic_loop_response_hook",
        "async_agentic_loop_cleanup_hook",
    ],
    Dispatch::Inference(InferenceBoundary::NotFiredNativelyYet(&[
        "litellm.llms.custom_httpx.llm_http_handler",
        "litellm.llms.openai.openai",
    ])) => [
        "async_should_run_chat_completion_agentic_loop",
        "async_run_chat_completion_agentic_loop",
    ],
    Dispatch::Inference(InferenceBoundary::NotFiredNativelyYet(&[
        "litellm.llms.custom_httpx.llm_http_handler",
    ])) => ["async_build_chat_completion_agentic_loop_plan"],
    Dispatch::Management("litellm.proxy.spend_tracking.cold_storage_handler") => [
        "get_proxy_server_request_from_cold_storage_with_object_key",
    ],
    Dispatch::HandlerTrait("litellm.integrations.custom_logger") => [
        "truncate_standard_logging_payload_content",
        "redacts_messages_itself",
        "handle_callback_failure",
        "get_callback_env_vars",
    ],
    Dispatch::Inference(InferenceBoundary::DeclarationOnly) => [
        "async_log_pre_api_call",
        "async_log_input_event",
    ],
    Dispatch::HandlerTrait("litellm.integrations.custom_guardrail") => [
        "add_standard_logging_guardrail_information_to_request_data",
        "async_pre_call_hook_on_messages",
        "filter_new_texts_for_session",
        "get_config_model",
        "get_disable_global_guardrail",
        "get_guardrail_dynamic_request_body_params",
        "get_guardrail_from_metadata",
        "get_guardrails_messages_for_call_type",
        "get_opted_out_global_guardrails_from_metadata",
        "get_supported_event_hooks",
        "handle_sensitive_data_detection",
        "inject_advisory_message",
        "mark_pre_call_hook_ran",
        "mark_texts_scanned",
        "mask_content_in_string",
        "raise_passthrough_exception",
        "raise_sensitive_data_route_exception",
        "render_violation_message",
        "should_route_on_sensitive_data",
        "should_run_guardrail",
        "structured_messages_cover_full_request",
        "supports_scan_only_tool_results",
        "update_in_memory_litellm_params",
        "uses_apply_guardrail_interface",
    ],
}
