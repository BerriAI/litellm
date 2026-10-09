from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 771},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 17,
            "outputTokens": 5,
            "serverToolUsage": {},
            "totalTokens": 22,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "stop_sequence": None,
        "usage": {"input_tokens": 17, "output_tokens": 5},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-sonnet-4-6",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-4-6/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 997},
        "output": {"message": {"content": [{"text": "Hello!"}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 18,
            "outputTokens": 5,
            "serverToolUsage": {},
            "totalTokens": 23,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/converse/us.anthropic.claude-sonnet-4-6",
        "stop_sequence": None,
        "usage": {"input_tokens": 18, "output_tokens": 5},
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 1201},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 21,
            "outputTokens": 6,
            "serverToolUsage": {},
            "totalTokens": 27,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5",
        "stop_sequence": None,
        "usage": {"input_tokens": 21, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-opus-4-8",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-opus-4-8/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 827},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 21,
            "outputTokens": 6,
            "serverToolUsage": {},
            "totalTokens": 27,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/converse/us.anthropic.claude-opus-4-8",
        "stop_sequence": None,
        "usage": {"input_tokens": 21, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 892},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 23,
            "outputTokens": 6,
            "serverToolUsage": {},
            "totalTokens": 29,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5-5",
        "stop_sequence": None,
        "usage": {"input_tokens": 23, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-opus-5-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-opus-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 10460},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 23,
            "outputTokens": 6,
            "serverToolUsage": {},
            "totalTokens": 29,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/converse/us.anthropic.claude-opus-5-5",
        "stop_sequence": None,
        "usage": {"input_tokens": 23, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

GROK_4_7_REDACTED_THINKING: Final = "cnNuX2pRenlYWklGSkxscEt5ZmxZa0NmdndJRmVKZHp5K1NpZzFWZGdlRlpydTUvUTFNRmVtNG91SEY3MzRKTXZ2VTR6KzhYQUY4QUFRQVZZWGR6TFdOeWVYQjBieTF3ZFdKc2FXTXRhMlY1QUVSQk9XWndORmxZTURSRlpraGpXRnBKWkZnMGVVMTNkVFJHYlZCNGNuYzVhREp3YWpFeFFuQkhPSHBJYTJsQmRHWmhlVlJJYlcxUFFUWTFTWHBVUWxvd1dGRTlQUUFCQUJGaGQzTXRhMjF6TFdocFpYSmhjbU5vZVFBS1ltVnliUzF6ZEc5eVpRQmNWZndMMExrcTAyOUJzMSsweThwTmNQbTB0Sm1ITWZaRTRucXppdFYwWXJ0MlowNkxpMXpVK1ZFZFFSVE5ZSDZrcC9KbjlVbEdqT0Vud3lOemp0S281RUJjZm5TSVJnSGxNelExaXV1bzNqeUtWNVdteGVvY2dBUnNIeXNDQUFBUUFDajhBTEZsbjVNN1hUZmRWMnJGMkJFeWF2V2d1RDg4Mll1Q0t6emNuN1dUNVRBZG90TDAvallkV1FWeVM4NmhJdi8vLy84QUFBQUJBQUFBQUFBQUFBQUFBQUFCQUFBQlVPcmY0NEIzRDVUbW5hNS8rZlUyUzVSalV0NDFYVzY5TFRyc0F2ZE53ZTA1WDRVL0cyT2llWGt3YXB2ZkwvZXoxb0JsY0QwcmNtS3dsSVNPdTJidTE3blNwRGdOdXQ2VFZYMkgycTRTTkxZOU84TWV0cDNwWmdSclcrTm9hV1gzY090Y1QrcXQwSTAwaEJKeW11YXRjbUxRWllpOFJHTXhIVHpsMXVHTXgvaGxuZmUvSnIwTUZaMFQvc3l0RjhuTWswUmJFUTBicGpyUTB4ZnZsbm00WEY1MnRLN05YYmM4QzRWN3g3V3dHL1o2MkdJK0pISHRMbWVrNWVCOEFCblNDbGE2TklnYVk5SU9ieXlGRG5mU25IdnQxMklrL2xlZUl1YnQ4VkFmR1dpYlNReE1NRFZ3RHVtekxsVU5EQjFzUEd1dHJpd0Rqa2c4ZzFGK3oyVUxCUEVZQ0hOWjA0NzRseXB5SFQxcDUwbEpBN05pSXZ6ckdhY2RRVDRHSFVmNHpQQ1h2RHpPU1djL3ZUVmJEWXp6a2hRZHRGUnF5WERRdFU0aWRCS0hzQ05JeXlBUmRCQi92cVBEWkRWVTNFcGpVcER5QURRTE1LM0krNXd2VEIzeVA5WUFaekJsQWpCZS9TakVscHJhbTc5U0dVT0hPOEtDZXdSQUxheUNpZGFGbjZPQThlbFVzRm5HRlhxeVJyL0VNam5UN0lCdDlHQUNNUUR3NTFEbkhGN3h4bm1ZMEZxeXlzSUlWaTVXbGpzNmRsWVdMUXV6UkxSd3UrYms2c0JmY05vZm12ODBTMFBhc2NvPQ=="

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
    expected_provider_endpoint="/model/global.xai.grok-4.7/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 1024},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 1445},
        "output": {
            "message": {
                "content": [{"reasoningContent": {"redactedContent": GROK_4_7_REDACTED_THINKING}}, {"text": "Hello."}],
                "role": "assistant",
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 30,
            "outputTokens": 53,
            "serverToolUsage": {},
            "totalTokens": 83,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/global.xai.grok-4.7",
        "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 53},
        "content": [
            {"type": "redacted_thinking", "data": GROK_4_7_REDACTED_THINKING},
            {"type": "text", "text": "Hello."},
        ],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)
