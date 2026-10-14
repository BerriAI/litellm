//! Which Python callbacks each neutral call stage reaches, and which callbacks Python
//! still invokes itself.

use crate::facets::{
    ARGUMENT_CALLBACKS, OUTCOME_CALLBACKS, PROVIDER_CALLBACKS, RESPONSE_CALLBACKS,
};

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

fn call_mappings(
    stages: &'static [(CallBoundary, &'static [&'static str])],
) -> impl Iterator<Item = CallbackMapping> {
    stages.iter().flat_map(|(boundary, callbacks)| {
        callbacks.iter().map(|callback| CallbackMapping {
            callback,
            dispatch: Dispatch::Call(*boundary),
        })
    })
}

pub fn callback_mappings() -> impl Iterator<Item = CallbackMapping> {
    call_mappings(ARGUMENT_CALLBACKS)
        .chain(call_mappings(PROVIDER_CALLBACKS))
        .chain(call_mappings(RESPONSE_CALLBACKS))
        .chain(call_mappings(OUTCOME_CALLBACKS))
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
