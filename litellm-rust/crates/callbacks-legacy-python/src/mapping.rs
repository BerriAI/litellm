use crate::inference::{self, InferenceBoundary};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Firing {
    Native(InferenceBoundary),
    Python(&'static [&'static str]),
    Never,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispatch {
    Gateway(&'static [&'static str]),
    Router(&'static [&'static str]),
    Inference(Firing),
    Management(&'static [&'static str]),
    HandlerTrait,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallbackMapping {
    pub callback: &'static str,
    pub dispatch: Dispatch,
}

pub fn callback_mappings() -> impl Iterator<Item = CallbackMapping> {
    inference::mappings().chain(PYTHON_OWNED.iter().copied())
}

macro_rules! python_owned {
    ($($dispatch:expr => [$($callback:literal),* $(,)?]),* $(,)?) => {
        const PYTHON_OWNED: &[CallbackMapping] = &[
            $($(CallbackMapping { callback: $callback, dispatch: $dispatch },)*)*
        ];
    };
}

python_owned! {
    Dispatch::Gateway(&["litellm.proxy.utils"]) => [
        "async_pre_call_hook",
        "async_moderation_hook",
        "async_post_call_success_hook",
        "async_post_call_streaming_hook",
        "async_post_call_streaming_iterator_hook",
        "async_post_call_failure_hook",
        "async_post_call_response_headers_hook",
        "async_filter_listed_models",
    ],
    Dispatch::Gateway(&["litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail"]) => [
        "apply_guardrail",
    ],
    Dispatch::Gateway(&["litellm.litellm_core_utils.litellm_logging"]) => [
        "async_post_mcp_tool_call_hook",
    ],
    Dispatch::Router(&["litellm.router"]) => [
        "async_pre_routing_hook",
        "async_filter_deployments",
        "pre_call_check",
        "async_pre_call_check",
    ],
    Dispatch::Router(&["litellm.router_utils.fallback_event_handlers"]) => [
        "log_success_fallback_event",
        "log_failure_fallback_event",
    ],
    Dispatch::Inference(Firing::Python(&["litellm.litellm_core_utils.litellm_logging"])) => [
        "async_get_chat_completion_prompt",
        "get_chat_completion_prompt",
        "log_stream_event",
        "async_log_stream_event",
    ],
    Dispatch::Inference(Firing::Python(&["litellm.llms.anthropic.pass_through.messages.handler"])) => [
        "async_pre_request_hook",
    ],
    Dispatch::Inference(Firing::Python(&[
        "litellm.litellm_core_utils.streaming_handler",
        "litellm.responses.streaming_iterator",
    ])) => ["async_post_call_streaming_deployment_hook"],
    Dispatch::Inference(Firing::Python(&["litellm.main"])) => [
        "translate_completion_input_params",
        "translate_completion_output_params",
        "translate_completion_output_params_streaming",
    ],
    Dispatch::Inference(Firing::Python(&["litellm.integrations.argilla"])) => ["async_dataset_hook"],
    Dispatch::Inference(Firing::Python(&[
        "litellm.llms.custom_httpx.llm_http_handler",
        "litellm.litellm_core_utils.chat_completion_agentic_loop",
    ])) => [
        "async_should_run_agentic_loop",
        "async_run_agentic_loop",
        "async_build_agentic_loop_plan",
        "async_post_agentic_loop_response_hook",
        "async_agentic_loop_cleanup_hook",
    ],
    Dispatch::Inference(Firing::Python(&[
        "litellm.llms.custom_httpx.llm_http_handler",
        "litellm.llms.openai.openai",
    ])) => [
        "async_should_run_chat_completion_agentic_loop",
        "async_run_chat_completion_agentic_loop",
    ],
    Dispatch::Inference(Firing::Python(&["litellm.llms.custom_httpx.llm_http_handler"])) => [
        "async_build_chat_completion_agentic_loop_plan",
    ],
    Dispatch::Inference(Firing::Never) => [
        "async_log_pre_api_call",
        "async_log_input_event",
    ],
    Dispatch::Management(&["litellm.proxy.management_helpers.audit_logs"]) => [
        "async_log_audit_log_event",
    ],
    Dispatch::Management(&["litellm.proxy.spend_tracking.cold_storage_handler"]) => [
        "get_proxy_server_request_from_cold_storage_with_object_key",
    ],
    Dispatch::HandlerTrait => [
        "truncate_standard_logging_payload_content",
        "redacts_messages_itself",
        "handle_callback_failure",
        "get_callback_env_vars",
    ],
}
