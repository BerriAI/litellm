from typing import Final

from integration.translation.case import TranslationTestCase

GROK_4_7_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/global.xai.grok-4.7",
        "max_tokens": 1024,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/openai/v1/chat/completions",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "model": "global.xai.grok-4.7",
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "max_completion_tokens": 1024,
    },
    mock_provider_response={
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"annotations": [], "content": "Hello.", "refusal": None, "role": "assistant"},
            }
        ],
        "created": 1791505413,
        "id": "chatcmpl-3xfhgmhrhhwplbepas3pjzbgjcccj4hx5tzplx7fw35giw3efznq",
        "model": "global.xai.grok-4.7",
        "object": "chat.completion",
        "service_tier": "default",
        "usage": {
            "completion_tokens": 37,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 27,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens": 30,
            "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
            "total_tokens": 67,
        },
        "system_fingerprint": None,
    },
    expected_litellm_response={
        "id": "chatcmpl-3xfhgmhrhhwplbepas3pjzbgjcccj4hx5tzplx7fw35giw3efznq",
        "type": "message",
        "role": "assistant",
        "model": "bedrock/global.xai.grok-4.7",
        "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 37},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)
