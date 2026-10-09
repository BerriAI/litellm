use litellm_host::{call::Operation, hooks::CallBoundary};

/// The operations a boundary runs a callback on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operations {
    All,
    Only(&'static [Operation]),
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

const MESSAGES: Operations = Operations::Only(&[Operation::Messages]);

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
        operations: Operations::All,
        callbacks: &["async_post_call_success_deployment_hook"],
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

#[cfg(test)]
impl Dispatch {
    fn contract_entry(self) -> String {
        match self {
            Self::Call {
                boundary,
                operations: Operations::All,
            } => format!("call:{boundary:?}"),
            Self::Call {
                boundary,
                operations: Operations::Only(operations),
            } => {
                let names: Vec<String> = operations
                    .iter()
                    .map(|operation| format!("{operation:?}"))
                    .collect();
                format!("call:{boundary:?}[{}]", names.join(","))
            }
            Self::Python(module) => format!("python:{module}"),
            Self::DeclarationOnly => "declaration-only".to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::callback_mappings;
    use crate::test_support::CUSTOM_LOGGER_CONTRACT;

    /// `custom_logger_contract.json` is the table as Python reads it: the unit test in
    /// `tests/unit/rust_bridge/test_callbacks_legacy_python.py` keeps its keys equal to the
    /// public methods of `CustomLogger`, so a new callback fails there until a row names
    /// where it runs.
    #[test]
    fn the_table_is_the_custom_logger_contract() {
        let mut table: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
        for mapping in callback_mappings() {
            table
                .entry(mapping.callback)
                .or_default()
                .insert(mapping.dispatch.contract_entry());
        }
        let contract: BTreeMap<&str, BTreeSet<String>> =
            serde_json::from_str(CUSTOM_LOGGER_CONTRACT).unwrap();
        assert_eq!(
            contract,
            table,
            "custom_logger_contract.json must be:\n{}",
            serde_json::to_string_pretty(&table).unwrap()
        );
    }
}
