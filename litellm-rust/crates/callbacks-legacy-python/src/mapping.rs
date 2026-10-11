use litellm_host::{
    hooks::CallHooks,
    interceptors::{RequestContext, WireRequest},
    lifecycle::{ExecutionEvent, Timing},
};
use litellm_host_python::{PythonCallEvent, PythonRuntime, PythonTask};
use pyo3::{prelude::*, types::PyDict};
use strum::IntoStaticStr;

use crate::{LegacyLogging, LoggingOperation};

#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq)]
pub enum CallBoundary {
    #[strum(serialize = "PrepareArguments")]
    PrepareArguments,
    #[strum(serialize = "PrepareRequest")]
    PrepareRequest,
    #[strum(serialize = "BeforeProviderRequest")]
    BeforeProviderRequest,
    #[strum(serialize = "AfterProviderResponse")]
    AfterProviderResponse,
    #[strum(serialize = "TransformResponse")]
    TransformResponse,
    #[strum(serialize = "Succeeded")]
    Succeeded,
    #[strum(serialize = "Failed")]
    Failed,
    #[strum(serialize = "StreamOpened")]
    StreamOpened,
    #[strum(serialize = "StreamChunk")]
    StreamChunk,
}

/// The operations a boundary runs a callback on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operations {
    All,
    Only(&'static [LoggingOperation]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispatch {
    Call {
        boundary: CallBoundary,
        operations: Operations,
    },
    Python(&'static str),
    DeclarationOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallbackMapping {
    pub callback: &'static str,
    pub dispatch: Dispatch,
}

struct Binding {
    boundary: CallBoundary,
    operations: Operations,
    callbacks: &'static [&'static str],
}

impl Binding {
    fn mappings(&self) -> impl Iterator<Item = CallbackMapping> {
        self.callbacks.iter().map(|callback| CallbackMapping {
            callback,
            dispatch: Dispatch::Call {
                boundary: self.boundary,
                operations: self.operations,
            },
        })
    }
}

const MESSAGES: Operations = Operations::Only(&[LoggingOperation::Messages]);

/// Every `CustomLogger` callback a native call runs itself, by the boundary that runs it.
const BINDINGS: &[Binding] = &[
    Binding {
        boundary: CallBoundary::PrepareArguments,
        operations: Operations::All,
        callbacks: &["async_pre_call_deployment_hook"],
    },
    Binding {
        boundary: CallBoundary::PrepareRequest,
        operations: MESSAGES,
        callbacks: &["async_pre_request_hook"],
    },
    Binding {
        boundary: CallBoundary::BeforeProviderRequest,
        operations: Operations::All,
        callbacks: &["log_pre_api_call", "log_input_event"],
    },
    Binding {
        boundary: CallBoundary::AfterProviderResponse,
        operations: Operations::All,
        callbacks: &["log_post_api_call"],
    },
    Binding {
        boundary: CallBoundary::TransformResponse,
        operations: MESSAGES,
        callbacks: &[
            "async_should_run_agentic_loop",
            "async_run_agentic_loop",
            "async_build_agentic_loop_plan",
            "async_post_agentic_loop_response_hook",
        ],
    },
    Binding {
        boundary: CallBoundary::TransformResponse,
        operations: Operations::All,
        callbacks: &["async_post_call_success_deployment_hook"],
    },
    Binding {
        boundary: CallBoundary::Succeeded,
        operations: Operations::All,
        callbacks: &[
            "log_success_event",
            "async_log_success_event",
            "logging_hook",
            "async_logging_hook",
            "redact_standard_logging_payload_from_model_call_details",
            "log_event",
            "async_log_event",
        ],
    },
    Binding {
        boundary: CallBoundary::Failed,
        operations: Operations::All,
        callbacks: &[
            "async_post_call_failure_deployment_hook",
            "log_failure_event",
            "async_log_failure_event",
            "log_model_group_rate_limit_error",
            "log_event",
            "async_log_event",
        ],
    },
    Binding {
        boundary: CallBoundary::StreamOpened,
        operations: Operations::All,
        callbacks: &[],
    },
    Binding {
        boundary: CallBoundary::StreamChunk,
        operations: Operations::All,
        callbacks: &[],
    },
];

pub fn callback_mappings() -> impl Iterator<Item = CallbackMapping> {
    BINDINGS
        .iter()
        .flat_map(Binding::mappings)
        .chain(PYTHON_CALLBACKS.iter().copied())
}

impl CallHooks<PythonRuntime> for LegacyLogging {
    async fn prepare_arguments(
        &mut self,
        task: &PythonTask,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<Py<PyDict>> {
        self.prepare_call(task, arguments, started_at).await
    }

    async fn arguments_prepared(&mut self, _: &PythonTask, arguments: &Py<PyDict>) -> PyResult<()> {
        Python::attach(|py| self.adopt_arguments(py, arguments));
        Ok(())
    }

    async fn before_provider_request(
        &mut self,
        _: &PythonTask,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<Box<WireRequest>> {
        Python::attach(|py| self.pre_call(py, wire, context))
    }

    async fn transform_response(
        &mut self,
        task: &PythonTask,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<Py<PyAny>> {
        self.transform_public_response(task, response, timing).await
    }

    async fn on_event(&mut self, task: &PythonTask, event: PythonCallEvent<'_>) -> PyResult<()> {
        match event {
            PythonCallEvent::Started { .. } | PythonCallEvent::Cancelled { .. } => Ok(()),
            PythonCallEvent::Execution(ExecutionEvent::ResultReady { facts }) => {
                Python::attach(|py| self.result_ready(py, &facts))
            }
            PythonCallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }) => {
                Python::attach(|py| self.post_call(py, raw))
            }
            PythonCallEvent::Succeeded { timing, response } => {
                Python::attach(|py| self.succeeded(py, timing, response))
            }
            PythonCallEvent::Failed {
                timing,
                origin,
                error,
            } => self.failed(task, timing, origin, error).await,
        }
    }

    async fn on_stream_open(&mut self, _: &PythonTask, head: &Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| self.stream_opened(py, head))
    }

    async fn on_stream_chunk(&mut self, _: &PythonTask, chunk: &Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| self.stream_chunk(py, chunk))
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
        "async_agentic_loop_cleanup_hook",
        "async_should_run_chat_completion_agentic_loop",
        "async_run_chat_completion_agentic_loop",
        "async_build_chat_completion_agentic_loop_plan",
    ],
    Dispatch::Python("litellm.litellm_core_utils.chat_completion_agentic_loop") => [
        "async_post_agentic_loop_response_hook",
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

#[cfg(test)]
mod tests {
    use serde_json::{Map, Value, json};

    use super::{Dispatch, Operations, callback_mappings};

    fn render(dispatch: Dispatch) -> Value {
        match dispatch {
            Dispatch::Call {
                boundary,
                operations: Operations::All,
            } => json!({"boundary": <&str>::from(boundary)}),
            Dispatch::Call {
                boundary,
                operations: Operations::Only(operations),
            } => json!({
                "boundary": <&str>::from(boundary),
                "operations": operations.iter().map(|&operation| <&str>::from(operation)).collect::<Vec<_>>(),
            }),
            Dispatch::Python(module) => json!({"python": module}),
            Dispatch::DeclarationOnly => json!({"declaration_only": true}),
        }
    }

    fn table() -> Value {
        let mut table = Map::new();
        for mapping in callback_mappings() {
            let rows = table
                .entry(mapping.callback)
                .or_insert_with(|| Value::Array(Vec::new()));
            if let Value::Array(rows) = rows {
                rows.push(render(mapping.dispatch));
            }
        }
        Value::Object(table)
    }

    /// `custom_logger_contract.json` is the table as Python reads it:
    /// `tests/unit/rust_bridge/test_callbacks_legacy_python.py` checks every public
    /// `CustomLogger` method has a row, so a hook added to Python fails there until this
    /// table says where the native call runs it.
    #[test]
    fn the_callback_table_is_published() {
        let published: Value =
            serde_json::from_str(include_str!("../custom_logger_contract.json")).unwrap();
        assert_eq!(
            serde_json::to_string_pretty(&published).unwrap(),
            serde_json::to_string_pretty(&table()).unwrap()
        );
    }
}
