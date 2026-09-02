from typing import Final

PROTOCOL_MAJOR: Final = 1
PROTOCOL_MINOR: Final = 0
TOKEN_METADATA_KEY: Final = "x-litellm-extension-token"

CALLBACK_HOOKS: Final = frozenset(
    {
        "async_pre_call_hook",
        "async_moderation_hook",
        "async_post_call_success_hook",
        "async_log_success_event",
        "async_log_failure_event",
        "async_log_stream_event",
        "async_post_call_streaming_hook",
        "async_post_call_streaming_iterator_hook",
    }
)
GUARDRAIL_HOOKS: Final = frozenset(
    {
        "async_pre_call_hook",
        "async_moderation_hook",
        "async_post_call_success_hook",
        "async_post_call_streaming_hook",
        "async_post_call_streaming_iterator_hook",
        "async_log_success_event",
        "async_log_failure_event",
        "async_log_stream_event",
    }
)
SUPPORTED_HOOKS: Final = CALLBACK_HOOKS | GUARDRAIL_HOOKS
