use std::future::{Future, ready};

use futures_util::future::BoxFuture;
use litellm_host::{
    hooks::CallHooks,
    interceptors::{RawResponse, RequestContext, WireRequest},
    lifecycle::{ExecutionEvent, FailureOrigin, Timing},
};
use litellm_host_python::{Awaiter, PythonCallEvent, PythonRuntime};
use pyo3::{exceptions::PyBaseException, prelude::*, types::PyDict};

use crate::LegacyHooks;

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

type Prepare = for<'a> fn(
    &'a LegacyHooks,
    &'a dyn Awaiter,
    Py<PyDict>,
    f64,
) -> BoxFuture<'a, PyResult<Py<PyDict>>>;
type Before = fn(&LegacyHooks, Box<WireRequest>, &RequestContext) -> PyResult<Box<WireRequest>>;
type After = fn(&LegacyHooks, &RawResponse) -> PyResult<()>;
type Transform = for<'a> fn(
    &'a LegacyHooks,
    &'a dyn Awaiter,
    Py<PyAny>,
    Timing,
) -> BoxFuture<'a, PyResult<Py<PyAny>>>;
type Success = fn(&LegacyHooks, Timing, &Py<PyAny>) -> PyResult<()>;
type Failure = for<'a> fn(
    &'a LegacyHooks,
    &'a dyn Awaiter,
    Timing,
    FailureOrigin,
    &'a Py<PyBaseException>,
) -> BoxFuture<'a, PyResult<()>>;
type Open = fn(&LegacyHooks, &Py<PyAny>) -> PyResult<()>;
type Chunk = fn(&LegacyHooks, &Py<PyAny>) -> PyResult<()>;

const PREPARE: Binding<Prepare> = Binding {
    boundary: CallBoundary::PrepareArguments,
    invoke: |hooks, task, arguments, started_at| {
        Box::pin(hooks.prepare_call(task, arguments, started_at))
    },
    callbacks: &["async_pre_call_deployment_hook"],
};

const BEFORE: Binding<Before> = Binding {
    boundary: CallBoundary::BeforeProviderRequest,
    invoke: LegacyHooks::pre_call,
    callbacks: &["log_pre_api_call", "log_input_event"],
};

const AFTER: Binding<After> = Binding {
    boundary: CallBoundary::AfterProviderResponse,
    invoke: LegacyHooks::post_call,
    callbacks: &["log_post_api_call"],
};

const TRANSFORM: Binding<Transform> = Binding {
    boundary: CallBoundary::TransformResponse,
    invoke: |hooks, task, response, timing| {
        Box::pin(hooks.transform_public_response(task, response, timing))
    },
    callbacks: &["async_post_call_success_deployment_hook"],
};

const SUCCESS: Binding<Success> = Binding {
    boundary: CallBoundary::Succeeded,
    invoke: LegacyHooks::succeeded,
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
    invoke: |hooks, task, timing, origin, error| {
        Box::pin(hooks.failed(task, timing, origin, error))
    },
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
    invoke: LegacyHooks::stream_opened,
    callbacks: &[],
};

const CHUNK: Binding<Chunk> = Binding {
    boundary: CallBoundary::StreamChunk,
    invoke: LegacyHooks::stream_chunk,
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

impl CallHooks<PythonRuntime> for LegacyHooks {
    fn prepare_arguments<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> impl Future<Output = PyResult<Py<PyDict>>> + Send {
        (PREPARE.invoke)(self, task, arguments, started_at)
    }

    fn arguments_prepared(&self, arguments: &Py<PyDict>) -> PyResult<()> {
        self.adopt_arguments(arguments);
        Ok(())
    }

    fn before_provider_request<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> impl Future<Output = PyResult<Box<WireRequest>>> + Send {
        ready((BEFORE.invoke)(self, wire, context))
    }

    fn transform_response<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        response: Py<PyAny>,
        timing: Timing,
    ) -> impl Future<Output = PyResult<Py<PyAny>>> + Send {
        (TRANSFORM.invoke)(self, task, response, timing)
    }

    async fn on_event<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        event: PythonCallEvent<'_>,
    ) -> PyResult<()> {
        match event {
            PythonCallEvent::Started { .. } | PythonCallEvent::Cancelled { .. } => Ok(()),
            PythonCallEvent::Execution(ExecutionEvent::ResultReady { facts }) => {
                self.result_ready(&facts)
            }
            PythonCallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }) => {
                (AFTER.invoke)(self, raw)
            }
            PythonCallEvent::Succeeded { timing, response } => {
                (SUCCESS.invoke)(self, timing, response)
            }
            PythonCallEvent::Failed {
                timing,
                origin,
                error,
            } => (FAILURE.invoke)(self, task, timing, origin, error).await,
        }
    }

    fn on_stream_open(&self, head: &Py<PyAny>) -> PyResult<()> {
        (OPEN.invoke)(self, head)
    }

    fn on_stream_chunk(&self, chunk: &Py<PyAny>) -> PyResult<()> {
        (CHUNK.invoke)(self, chunk)
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
