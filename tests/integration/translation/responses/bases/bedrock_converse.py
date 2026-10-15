import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase
from integration.translation.responses.templates import RESPONSES_REQUEST, RESPONSES_RESPONSE
from integration.translation.templates.bedrock_converse import (
    BEDROCK_CONVERSE_REQUEST,
    BEDROCK_CONVERSE_RESPONSE,
)
from pydantic import JsonValue

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
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
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 17,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 17,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 5,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 5},
            "total_tokens": 22,
            "cost": None,
        },
        "user": None,
        "store": None,
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-sonnet-4-6",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
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
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "bedrock/converse/us.anthropic.claude-sonnet-4-6",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 18,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 18,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 5,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 5},
            "total_tokens": 23,
            "cost": None,
        },
        "user": None,
        "store": None,
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
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
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 21,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 21,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 6},
            "total_tokens": 27,
            "cost": None,
        },
        "user": None,
        "store": None,
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-opus-4-8",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
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
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "bedrock/converse/us.anthropic.claude-opus-4-8",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 21,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 21,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 6},
            "total_tokens": 27,
            "cost": None,
        },
        "user": None,
        "store": None,
    },
)

BEDROCK_CONVERSE_RESPONSES_RESPONSE: Final[Mapping[str, JsonValue]] = {
    key: value for key, value in RESPONSES_RESPONSE.items() if key not in ("provider_specific_fields",)
}

BEDROCK_CONVERSE_RESPONSES_SYSTEM: Final[JsonValue] = [
    {
        "text": "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
        "Explain which file you need next, then call a read-only tool. Never change files. Use the "
        "responses client for this project review. Consider zero, negative and non-integer quantities "
        "before choosing the next file to inspect.",
    },
]

CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQS3AYKEggSEAEYAjgBQgh0aGlua2luZxIMvtg77qfREr4ZWBn7GgzxzrtNwc5N8DqpBHgiMMJUcH5Vw6hj0kYyAwbI34q5FDQHs+VeRDCU"
    "N9EDE5JENmjhQSbLvD87AP+BtUfZPSr3BegcgjSctkyUvfkBNE3E+FvfkZAAwgvBMZ904om1Ua3usYegKOyyc4dHlwRSvDFvJFDSeIrtTek2"
    "ugzdsjeSeJLr8FoiYxqVIDSfe40IraTHZs5omadfA/F2I9Yj6nGHUCkGQK/TKZWA4G+3+H4FiPK84gjyFL4686p/8ljiWsz11SO862UHRR7J"
    "QAl9e6tprj4qpJRawR58Ky+JyOAUzUD/DpZyFYjYFwsgAiv5IY7kCfW/osGKrwnEr9dYX8JQu6h5kZBxFBxETdASJQBHrIWC/zrYADuzIO7Q"
    "YzRwnQqc0v/hCnuf01rBX+q41iRVHlJIDsEmdkxiNB4tGE/Azo+RUbTd4fBBGH2t9iY4zi+5ykBZ38BzjTIQAjBc5Ip3LSoNHqNzFAj65wBj"
    "nI1L21+jANpMxjZ1DvXCkMc2ZMh/hkPNGqnEKPaIZjYTp6ZX4MJ1G7orq1VqFUYKJ39whqkQR2i64iaBRJ7f+n4eHYNR0ZAGF6E0/pJzpB27"
    "2lsWrz2Ky6OsK0UU5HsGgFmkOLI1IGEvXqeE4gmRz8AMseNY1RobNPmNmv4nj7DjLrg9wnkDLlQP0Vg8N6LXqGPlIZeICtcNEM/7KhiqIwzf"
    "KZnPc5erpBrCS3qcM0jzP/vGwOdpuZWbiDfn8iwJatVEYgwzsZOi1kIJDGHSCLyznM8CiGGpTBY5BIh8yj2k6D8JIOdKOLfVVJtrCE+LowpB"
    "BTGLnC9mEQkJ4ZD/wS4rHQOeWiN2tvbm6casvCvzDDRPISoLI0mQa7WXr+ovbScBIgmZhoJFAwaU5b0ii5hoizTDx6Uc4Ak5zr/DETMNvVde"
    "dTvD0MWzZ0Olo1W3bUoHfu+I7yiPGE9jmp3Yugj293UtIUm5GDJpuXwDeDTrcu+dYrFCWm68qn7dRAk2/47a9BqknHboVsD+Z0Zxi3DItUBa"
    "/iFxdh7kgJSVZ+oHy2EcEu3Tlg1vEDDPXoivGCzV/sSQ1L2kimTI2jtYhHkWZU7kn7VUmCXdsRgB"
)

CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "bedrock/converse/us.anthropic.claude-haiku-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-haiku-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_CONVERSE_REQUEST,
        "system": BEDROCK_CONVERSE_RESPONSES_SYSTEM,
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 116}}},
        "metrics": {"latencyMs": 3114},
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                                "text": "I need to check shop/validation.py to answer this, so I'll read the first "
                                "chunk of lines to see its contents before responding.\n\n",
                            },
                        },
                    },
                    {
                        "text": "I'll read `shop/validation.py` first, since that's where `validate_quantity` is "
                        "defined and where the zero-quantity rejection should come from.",
                    },
                    {
                        "toolUse": {
                            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                            "name": "read_file",
                            "toolUseId": "tooluse_l8Pc3cjAkezdGOeQoxs9XG",
                            "type": "tool_use",
                        },
                    },
                ],
                "role": "assistant",
            },
        },
        "usage": {
            "cacheDetails": [{"inputTokens": 853, "ttl": "5m"}],
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 853,
            "cacheWriteInputTokens": 853,
            "inputTokens": 4,
            "outputTokens": 262,
            "serverToolUsage": {},
            "totalTokens": 1119,
        },
    },
    expected_litellm_response={
        **BEDROCK_CONVERSE_RESPONSES_RESPONSE,
        "model": "bedrock/converse/us.anthropic.claude-haiku-5-5",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I need to check shop/validation.py to answer this, so I'll read the first chunk of "
                        "lines to see its contents before responding.\n\n",
                        "annotations": [],
                    },
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "I need to check shop/validation.py to answer this, so I'll read the first "
                            "chunk of lines to see its contents before responding.\n\n",
                            "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'll read `shop/validation.py` first, since that's where `validate_quantity` is "
                        "defined and where the zero-quantity rejection should come from.",
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "tooluse_l8Pc3cjAkezdGOeQoxs9XG",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_tooluse_l8Pc3cjAkezdGOeQoxs9XG",
                "namespace": None,
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 857,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
                "cache_write_tokens": 853,
            },
            "output_tokens": 262,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 116, "text_tokens": 146},
            "total_tokens": 1119,
            "cost": None,
        },
    },
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQS6gkKEggSEAEYAjgBQgh0aGlua2luZxIM5Rk71l/Z9rkMtA6ZGgwuf1QIpmmol0EDZgwiMAXhxIVSflqszyyicoACM0MZMWNy03vjepAm"
    "b4tmo7kCof9sNzr0fhQHwTSdDGzUlyqFCdgG/z4e8K7FKJQXQvtfpj0ubUZ0e3Cipo/DjMOZdQlMuAy+mywzO73iQG/afjuWcEMc0NKMgPki"
    "0EFN23Fy6adXfSQXvRej3PxBT16bNSth1ytGBnoby/VgtMa1JVW3E9h9kCkbEnA8+AT1XSgqQs1pigN5LbTPDN13nYutXxd3BoP/cnPsR5vN"
    "znslFbfjd19eup0W0iKTk9xpP1HEvcuYnYlIIVDHYGDP7rzUXtWIqI9ZG1KdzVHFBa9qhEcmKMeWks4iXcKVAiNRcdip3kdowlGZWyG4MEmJ"
    "94BWm9JJbn6f+DpxsDk6bFyqq1RJW5l+aApuKX/yN915Ccd6gARvz+nuxpjuDZOln/DzHiBqlnDBPRgn82q+csvQ3iXAxIeNEcTscEu+zsIU"
    "zwtZciFKXs1TlMIiBAtoQ3NE7HM9vJNkWWiQRHtE5n4jhtN2Yqs5hnvg05vTGvXnfqs5sGjbP2yj18TOYmu2HlZd6NFADuLTulA/zovgGm5t"
    "gNzJE26Blh/4QoiT3TMUMjvOJtRhPz2VmdkKbbiBn1R3Nf1CTStND00zdOQ6NQululcU5Y74fbnzADHl1dOAw7AkoHlp953vkEOJvK1N9kPw"
    "ZOwBz5Nc0XRRoZ4zXSM50wg6OdTWqS8vT2nlJslvKT9AghPV3LHyY5HBUpZPR0Up2OZQKZr/G5/BAKpM5Jr5P0tIAf9gHLBiHNKcpk+Upkzv"
    "HfT8QV3dl0x80yaUyQrU1Glv8K56pI3OUe/NkmJw7CMhVSUfPt4U609XzzKNEEwFKUjoUtJt/ZOsBxESqGfs3hSayF6i9kFzIzWdHKE6RnAj"
    "7JtjVQlff0aIpoAH89oq5ZhFaA8xkqbxu8eDY0L4HtNRVTEyNiGrnJBlp0O7eEoL/4ZRW+j6HcK9bB7O/8diJ/JXk3MEpyYt8MLBHbhVbzzX"
    "PJuIZM5NObOchFbR53EjBHMvyRVvF4SnZaZAW3DxOeA49UpuiBDbDPiPe0XOmtM3BwK+vRnmeI61nWfn4mAOVQeNZLjf+spO+3bDxXtHt1bz"
    "yK1Z4pLO3dkBEhZKEk6Ha6fRPSPBi2/xcK80R1WyWbVhGvgYK0hP9Uy36ZK6WvTMiEVL9b5/+t+yNNJsflstuMYLNuASU0PE7ne35zYao/gc"
    "qKzM8t9gjJpqGUtPoR9ndaBlilgHOkgXV+h29jA4OIPtJnFfI49sztRrA0thymK+lm8LY0F1nnhsPkxc1dohz8EykdV18vO8y2cyeZHNwVrp"
    "OEW3nWGz+/GFyRCKsfexlGk6npiFvk5ZJQFZY/Kdh4VuTDKEhmm9QhZTwEYFrh3B5OlO8UjaiIliJ+4Xx642I2MvRbkeqDbJGxSCqxNIS1yJ"
    "Yq9UVZzkFXhPvy9/9XFfXH3tedoDceTGG8YD4IummDngMs204wz7VHve814/eTD3qS70oX5O36aY/Aswnz9lS4PhYrAtITp0YmNyBingiqpi"
    "Csxq8ePR2IQXawltEYg3cS+weEVJBUaxusjRf/3SdwhyoDuXqaqtrr/r4L7UMf5HGAE="
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1: Final = (
    "CAQS9wUKEwgSEAEYAjgBQgluYXJyYXRpb24SDNP7b6gtJ4xvIEG9ohoMqC7J3vxdFt/qo/sOIjC6trR5FvvYWW/MCffxtMIugtRws7utg3y+"
    "18ZIygEbDw4yGfS2DkcGZ9q6obx1Zs4qkQV27sYDMo7sDReqPo0XpOQHMt7NOsJmrJQ9IdAfTOXPqKPwnC6LU3dnS6lvSWw5gDrdXgXd9iHu"
    "Yb1+pVCrkpBnYwrb5N7tOqMQvRI60qxGBOarSLTpvhysv5G3ZIXfmliYZbmITa3I8tlvKoqIIAGXc7zqgI7VfMU8yuK/dtDbCO6/Q+/qPnIL"
    "lrY+F4NIaByk9WcjJMXGarMfuohw2acciumMpDu1rqxUpLaPMjhaTFv17lu07uOOu3cuiySN06Fzwp7TbWiKS2pi1aORKXYDowrF59gyTppc"
    "G6Y3M5983lXB2oLtz7Gnyr8uBbcjayWDqZCMitoa8m5twh+VEOnHdzay+a4glnSiPr5A1D3nH1ENocWj6+eKS+LXkdvOUse5XkDaOUPAfAm1"
    "4f3tn9G1+0S6Zo6Lmwjw/+79OfUPFEHgkZ6Ih6xhMR0yM/3FIAjSXMm+t7VNH23Ei7EkCWdv4OrSPA5PweCX8lnMwQ5avMuna6ZeiAV3A/tu"
    "SGy2tYkbjpUJ3JPGfuRmwfCV499fkJJY6oZthK1ot0746frbEgHAnYFyts064x+/yp9D0KariKPuCpfyFqDYn4EhoA+ROvBTxkrQu1KveTAC"
    "55/v1tOBlLAZsVzXT7XqbCItMt+AXQSA3jaOmtrU425oOyhc9AUKSxvSR+jTBCcGmJCWKrnEWUjqTW5AbG5byposBH3tYor6L769fli2cTRy"
    "IerlH8AHj2CgjuXNtGbxP2Acjf6TjGyY2WeIhT/dQXWKQNzfsvdVtfLRyEESr7QKyiBC/vOFeTTMfQoFB7+Q6mTh6mnYmw52ytfociSKPi3x"
    "yAYXpJQGxmCINv++wCIocN3xp703kU+Yt+9I4wKL2+JuN7IYAQ=="
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "bedrock/converse/us.anthropic.claude-sonnet-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_CONVERSE_REQUEST,
        "system": BEDROCK_CONVERSE_RESPONSES_SYSTEM,
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 235}}},
        "metrics": {"latencyMs": 5698},
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                                "text": "The user asked to read validation first, though both files could "
                                "technically be read independently. I'll follow their sequencing and read "
                                "the validation file first.\n\n",
                            },
                        },
                    },
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1,
                                "text": "Now checking shop/validation.py to see how validate_quantity handles zero, "
                                "negative, and non-integer inputs, starting with imports and helper "
                                "functions near the top.\n\n",
                            },
                        },
                    },
                    {
                        "toolUse": {
                            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                            "name": "read_file",
                            "toolUseId": "tooluse_HZpviLzbdTE4T0GMawHsmi",
                            "type": "tool_use",
                        },
                    },
                ],
                "role": "assistant",
            },
        },
        "usage": {
            "cacheDetails": [{"inputTokens": 853, "ttl": "5m"}],
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 853,
            "cacheWriteInputTokens": 853,
            "inputTokens": 4,
            "outputTokens": 404,
            "serverToolUsage": {},
            "totalTokens": 1261,
        },
    },
    expected_litellm_response={
        **BEDROCK_CONVERSE_RESPONSES_RESPONSE,
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5-5",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The user asked to read validation first, though both files could technically be "
                        "read independently. I'll follow their sequencing and read the validation file "
                        "first.\n\nNow checking shop/validation.py to see how validate_quantity handles "
                        "zero, negative, and non-integer inputs, starting with imports and helper functions "
                        "near the top.\n\n",
                        "annotations": [],
                    },
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "The user asked to read validation first, though both files could technically "
                            "be read independently. I'll follow their sequencing and read the validation "
                            "file first.\n\n",
                            "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": "Now checking shop/validation.py to see how validate_quantity handles zero, "
                            "negative, and non-integer inputs, starting with imports and helper functions "
                            "near the top.\n\n",
                            "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "tooluse_HZpviLzbdTE4T0GMawHsmi",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_tooluse_HZpviLzbdTE4T0GMawHsmi",
                "namespace": None,
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 857,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
                "cache_write_tokens": 853,
            },
            "output_tokens": 404,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 235, "text_tokens": 169},
            "total_tokens": 1261,
            "cost": None,
        },
    },
)

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSzQUKEggSEAEYAjgBQgh0aGlua2luZxIMBQgAH/utL+SlHBltGgxDyOhF0rniv4pJNTQiMNmF1xF0QjswZwV2rMEyccqYlGdGvMcM6/MX"
    "lTjznngJK6CmT4WJ0ueKjnCuwChARCroBMQXhDXHvXCbvjESc/c/WnSjWeZtFvycp287jestqbK0BDZ8v9ErwC6iSx2yBNgVFHzpJuOmO06e"
    "8Y49ALNuhk9VMl4CGjfbGVdgeGGDrjkk4m9oo0JlYvBykNJiSf6Tu82PuYLgirixNBOf5ZMRX19v6oMZA7RHxJz72XlNAde6VBEjMy6LV/Jy"
    "0tcrYI12e9sxjJaDZ4wF2Si6/+VRMRM0QkQHFhVALNFnxvb4kawAPEkx/I1gdBvEB76ACTOjmKYnYdPZPdqu9eScnTamuOGxpet6+82cSPnb"
    "IsUXD1XXT0eJUFi40ot60YFcFWTX3vpJYkzMjndouRx998TaLS7TxqUiYFPu2kquJBOY/pDtkh0oysaXBQ0XnZyHtXIRuzFkfWgjkmvPL+UJ"
    "ZxDUA0RMuUUM3b6cqjvOgGBg7/1qBFfgc5cr8nPOeec+0E1zk0szyTQ+14fwMMkRte0emSjhPiIIKk2HspOP2IMahYKXx8Ni9xp4WSXwylp6"
    "gXyikRTmJRlnknU9NK/VlqJEAcOlZMSPC55ug53ZppDw71ssCiFeCcEGQCMajAbG5DEfLxxY4h4fqGxCqj2qJI8ATA4caGytPY0lBS3+DsrJ"
    "sUuzBdVtHiMceMJtEsPVpvAc9ITzl2+rwXiC2+MMeJVXgGpEyCCcksI2Bs2wIoly1/1J9rG9kz71neIhItwY9B3r5LgSEliWYpuSI/LbiZHB"
    "9ZwZJj87gKYH/C3U5w8oc+OLZO7zOQ/8bbLIQTyuk+hepc8KZgq5eouSMvhWGbmE+eA+0On+mB+85EmSmFu1JTXt2+D5/BVVbnQYAQ=="
)

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_1: Final = (
    "CAQS6gUKEwgSEAEYAjgBQgluYXJyYXRpb24SDJFe0B/jn7Vc9l86UxoM+3QEi7NVYrrEAJXWIjDH/Ye9cV6IW0LpbnOf4ZukC5Dm+ug53eFV"
    "0Aad2kQp1qLUJ3qu6oXbevVE6AGblSYqhAUYC6yg3pfX1HlZfxFG2YmiNPSo4jxv/nXTib4cG6c64wVIKb7lcnUKVhqsbii2eBZ3+dcyeqah"
    "LqgqhS5gz8qM0JWf8erzMW27BwS4z+VflWHyDNcYxZ0N5xg8hxKObWmDNJ1hv46wCUL3BVB2q4IcKMsdxWtIIz6PAllot/4G6ton3kESWcVj"
    "k6Q1Moj+RzvX1pDESbIaWry84vTfuH8vq2x5p3yTICKl6+lHFrxt9HWPRW6SebPOJ/7y2Y6qOb6h4bEb8M8sRdm5p/LbhfwBc3DVQ7plPic2"
    "IkS7mMwCInmJYbjRdefnUX5fneCwccH56O9IarBZ/s1VM0D/Xv1tgalZb9bNaaw7BvJX/5IkBP6yjYhbtPeh9I/J9XPYeOcNRn1QgbskbEJ1"
    "Ff3lvL4jEqydqRyuQcyasCXGxAsFkNqA6grC5sFNBBlk5lcR1T8poXh7d/36BMNTcU5AzEBjzA/i0QQvKwVVw4IkjphHd3r+6ZgRSHGc0m6R"
    "WY6NlKfTCQEhF0fZeL77TURzs921CXiynXmSyqvuuEOU6aEy6n56AkjlxdpZ4UgEzGMiij4i21qOpBsS18Fb5lIBC12DqhEXWu/NFthVDlG/"
    "7Gwu4o/v1RQIXHaM8Kvxobt75gyaIWp1NIeN9GN/SAiTOjc/zos+HDFzGQYSmV5tPcly/xItwTBDgtJChfujxwd09b6j2WH38Mds89Cdz7CT"
    "aplQBwhdVmVTRXhAv9nRDxPWzVnYAtI9hHeLoxU8jJAAXq5f2DErZxF7TdYwR4UGVrj21w+eQ97yPMyPAYP9XbKmo6xp56fMz+TmpBfiNJM9"
    "fU+/b+ImXYstTNT/cNu7z7gwZv6fthgB"
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "bedrock/converse/us.anthropic.claude-opus-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-opus-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_CONVERSE_REQUEST,
        "system": BEDROCK_CONVERSE_RESPONSES_SYSTEM,
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 71}}},
        "metrics": {"latencyMs": 12388},
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                                "text": "Since the user asked for the validation file first, I'll read "
                                "shop/validation.py around line 18 before looking at the test file.\n\n",
                            },
                        },
                    },
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_1,
                                "text": "I'll check `shop/validation.py`, reading lines 1–50 to see "
                                "`validate_quantity` along with any related imports or constants.\n\n",
                            },
                        },
                    },
                    {
                        "toolUse": {
                            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 50},
                            "name": "read_file",
                            "toolUseId": "tooluse_ZbXhsiKlmJB473d9A8N5ZI",
                            "type": "tool_use",
                        },
                    },
                ],
                "role": "assistant",
            },
        },
        "usage": {
            "cacheDetails": [{"inputTokens": 853, "ttl": "5m"}],
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 853,
            "cacheWriteInputTokens": 853,
            "inputTokens": 4,
            "outputTokens": 245,
            "serverToolUsage": {},
            "totalTokens": 1102,
        },
    },
    expected_litellm_response={
        **BEDROCK_CONVERSE_RESPONSES_RESPONSE,
        "model": "bedrock/converse/us.anthropic.claude-opus-5-5",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Since the user asked for the validation file first, I'll read shop/validation.py "
                        "around line 18 before looking at the test file.\n\nI'll check "
                        "`shop/validation.py`, reading lines 1–50 to see `validate_quantity` along "
                        "with any related imports or constants.\n\n",
                        "annotations": [],
                    },
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "Since the user asked for the validation file first, I'll read "
                            "shop/validation.py around line 18 before looking at the test file.\n\n",
                            "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": "I'll check `shop/validation.py`, reading lines 1–50 to see "
                            "`validate_quantity` along with any related imports or constants.\n\n",
                            "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_1,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 50}',
                "call_id": "tooluse_ZbXhsiKlmJB473d9A8N5ZI",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_tooluse_ZbXhsiKlmJB473d9A8N5ZI",
                "namespace": None,
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 857,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
                "cache_write_tokens": 853,
            },
            "output_tokens": 245,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 71, "text_tokens": 174},
            "total_tokens": 1102,
            "cost": None,
        },
    },
)

CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0: Final = (
    "CAQS4gYKEggSEAEYAjgBQgh0aGlua2luZxIMwBPNPxtNdRTj9z+3GgwSLCcSbOT5eOdg4a0iMMMEuXnI1wytKqyHTo9ejWTxkMM26SCWUqTI"
    "H8eIgHNrJHcYTESawe9nmU18j4tufir9Bdw7vFeVYpvoISrtD0A+reIZiKHpRSyvhuL7aiOVQmX8Nm+o9LWnDjMzO8VeHw/JhmEY53nv/kb/"
    "aU2CxklgPf+LYWHzlvVUNt+z4vWca/66spf3iduVP1rofkqnKNNSe1q6ECXv9x5/esewnGGjtuxLOB/lTYU3j67KUSGbTmoV9008oLpTnAkC"
    "9XOfWprSoh12L0XSqip8rorUlcyD3XLbiVu6MZt+mGbaVv7Qip7tOzOcAwPl/vQKPAFlV7Fwk7Jg/l4vwErw7s+xqANYxZRNPOs/quWe2/tc"
    "stL5T7Cm2sGNJx0LI022V9Xfu+MihJvl54c3IWh0duPIU4DToEkNHeRJY++DjrynqXdeWH8eydrtdzbuNhWCGW936UlUKfYoziUhkSS+7MpZ"
    "R1hhK7TErXI+5CWbghy7c4KdH8darDfyLCb0KLwYM9A4PqzLL2ZpIZS4oL7EvuXFTC/Wa40Omtmtn4klBwGHmTUHL6JXXaZGXTtz5HqGSaK4"
    "8Ia/W+YG+q0IaApw1NSs8MMdnOlt5eF7qVla73EjF8JE3TLYn/NKPqBAdoPKoe+PxxtFY9fkHIHNrnJupT7nt9YR7nf8nzPm6Y7+KeGHkJ6x"
    "LHTiv58t2iV2E5jbDP1awcQg4azhpTr/cBz7cyUnTiGJGjwt/NS55rNnfnH24E3F3yUMLGVySc035LaInTJOVTzWfmT9BKmc3T18WQERUEs2"
    "SFO6X7BHWV0Dcx4XqedsR//Fjp9v3lspa2fVqd5/+g4xq8z2kBZlBnXkLQMFNpFwd4ZRiVzRx94u3AzSza2fHFByDhLSHkaQpblg79lo1JDG"
    "Op2fbO0TZnwwUsth4Qq/9QvvtpYtAEMox9ISsqMGCELZeT8StjFfAmro7iwp2nGmVfGfjvoTpOZ8oaQChOdi7V/cy7An1v/58GMapRqEKWCj"
    "JtuI5qNyka4ECp7FdT++oQJeU7U85PQurp55wlEf4drBvjYl/aICnnxH6ueg8izsmC/hya5IRdRfD7dQoxgB"
)

CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "bedrock/converse/us.anthropic.claude-fable-5-1"},
    expected_provider_endpoint="/model/us.anthropic.claude-fable-5-1/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_CONVERSE_REQUEST,
        "system": BEDROCK_CONVERSE_RESPONSES_SYSTEM,
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "paths": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["query", "paths"],
                                "additionalProperties": False,
                            },
                        },
                        "name": "search_files",
                        "description": "Search source and test files for a literal term.",
                    },
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "start_line": {"type": "integer", "minimum": 1},
                                    "end_line": {"type": "integer", "minimum": 1},
                                },
                                "required": ["path", "start_line", "end_line"],
                                "additionalProperties": False,
                            },
                        },
                        "name": "read_file",
                        "description": "Read a range of lines from a project file.",
                    },
                },
            ],
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 102}}},
        "metrics": {"latencyMs": 9400},
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0,
                                "text": "The user wants validation read first even though the files are "
                                "independent, so I'll respect that order: read validation.py now, then the "
                                "test file next.\n\n",
                            },
                        },
                    },
                    {
                        "text": "I'll read `shop/validation.py` around `validate_quantity` to see how zero, "
                        "negative, and non-integer values are treated; the test file comes next.",
                    },
                    {
                        "toolUse": {
                            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
                            "name": "read_file",
                            "toolUseId": "tooluse_IZcZuTirqaML2t3hxtMt9K",
                            "type": "tool_use",
                        },
                    },
                ],
                "role": "assistant",
            },
        },
        "usage": {
            "cacheDetails": [{"inputTokens": 873, "ttl": "5m"}],
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 873,
            "cacheWriteInputTokens": 873,
            "inputTokens": 4,
            "outputTokens": 250,
            "serverToolUsage": {},
            "totalTokens": 1127,
        },
    },
    expected_litellm_response={
        **BEDROCK_CONVERSE_RESPONSES_RESPONSE,
        "model": "bedrock/converse/us.anthropic.claude-fable-5-1",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The user wants validation read first even though the files are independent, so "
                        "I'll respect that order: read validation.py now, then the test file next.\n\n",
                        "annotations": [],
                    },
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "The user wants validation read first even though the files are independent, so "
                            "I'll respect that order: read validation.py now, then the test file next.\n\n",
                            "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'll read `shop/validation.py` around `validate_quantity` to see how zero, "
                        "negative, and non-integer values are treated; the test file comes next.",
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 40}',
                "call_id": "tooluse_IZcZuTirqaML2t3hxtMt9K",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_tooluse_IZcZuTirqaML2t3hxtMt9K",
                "namespace": None,
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 877,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
                "cache_write_tokens": 873,
            },
            "output_tokens": 250,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 102, "text_tokens": 148},
            "total_tokens": 1127,
            "cost": None,
        },
    },
)
