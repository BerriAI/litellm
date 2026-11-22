from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.templates import CHAT_COMPLETIONS_REQUEST, CHAT_COMPLETIONS_RESPONSE
from integration.translation.templates.bedrock_invoke import (
    BEDROCK_INVOKE_REQUEST,
    BEDROCK_INVOKE_RESPONSE,
    CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
)

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
    },
    mock_provider_response={
        "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "id": "msg_bdrk_uaxykqyuejauacvh6ordtb5bbst4xlruet5vju5zjinbcjhycmnq",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 17,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 5,
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 17,
            "total_tokens": 22,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 5},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 17,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "service_tier": "standard",
        },
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-4-6/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
    },
    mock_provider_response={
        "model": "claude-sonnet-4-6",
        "id": "msg_bdrk_4ky3vxhsv4ko5oyipjlcejlaxydydvgsfe4tnjdng5wsfzlzalca",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 18,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 5,
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-4-6",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello!",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 18,
            "total_tokens": 23,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 5},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 18,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "service_tier": "standard",
        },
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_bdrk_vrjdwnaebmlfxkxvd2a74ovr6xl6a3ovufg3gwh3hktoiockg56q",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 21,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-5",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 21,
            "total_tokens": 27,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 6},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 21,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "service_tier": "standard",
        },
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock/invoke/us.anthropic.claude-opus-4-8",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-opus-4-8/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
    },
    mock_provider_response={
        "model": "claude-opus-4-8",
        "id": "msg_bdrk_j2vze3iapymvotnltudlq6kyexytgav6tdiwmintqfjsdga266bq",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 21,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "bedrock/invoke/us.anthropic.claude-opus-4-8",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 21,
            "total_tokens": 27,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 6},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 21,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "service_tier": "standard",
        },
    },
)

CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock/invoke/us.anthropic.claude-haiku-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-haiku-5-5/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={**BEDROCK_INVOKE_REQUEST},
    mock_provider_response={**BEDROCK_INVOKE_RESPONSE},
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "bedrock/invoke/us.anthropic.claude-haiku-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I'll read `shop/validation.py` around the `validate_quantity` definition at line 18. "
                    "I'm checking how it treats zero, negative, and non-integer values.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                                "name": "read_file",
                            },
                            "id": "toolu_bdrk_01Hfy4YfUU7hZUxEmpp4oDYV",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "I should look at shop/validation.py, focusing on how it handles zero, negative, and "
                    "non-integer inputs before deciding what to read next. I'll check the first 60 lines or "
                    "so since I don't know the file's full length yet.\n\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I should look at shop/validation.py, focusing on how it handles zero, "
                            "negative, and non-integer inputs before deciding what to read next. I'll check "
                            "the first 60 lines or so since I don't know the file's full length yet.\n\n",
                            "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "I should look at shop/validation.py, focusing on how it handles zero, "
                                "negative, and non-integer inputs before deciding what to read next. I'll "
                                "check the first 60 lines or so since I don't know the file's full length "
                                "yet.\n\n",
                                "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                            },
                        ],
                    },
                },
            },
        ],
        "usage": {
            "completion_tokens": 295,
            "prompt_tokens": 880,
            "total_tokens": 1175,
            "completion_tokens_details": {"reasoning_tokens": 143, "text_tokens": 152},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 876,
                "cache_creation_tokens": 876,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "service_tier": "standard",
        },
    },
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSwgUKEggSEAEYAjgBQgh0aGlua2luZxIMwv5o3sIXxVdkUO0MGgxObtbvqBQXfANsnrwiMMS0BVrFyZB1bCUZfx4cUYR/+/9/NfiPwTEU"
    "jsGSOJFjxW95CdHDmohqchGNWlIBUirdBEZuOEcmqJBmrwvvX5mho5M6DJskEztgdUjCFIzTM5CXU24my+0yxBnIrFUrgzqaihEmKWmwxSEN"
    "rGOtF41225PHpYTpfWHKxn6tay7STRpm8K6x6AXYtYJSWMAa0pvCBrsc8U4jvbuAHhLUqxliQG0WwkbX4UK2/6uGMSwYDMHsh+kvDLa17WhU"
    "vjZsKg0ocntDgcMBG4WVxhaCMurK2lhDP14G5q3CdViTQd8TiM/ooU2cx7NrlqHWtB0tr4QuZaXZlTZjLz+wpIPzvLSQvyybecc/LrD+4EGV"
    "CbVHTucTj8qEBWNoT7JPnwHIH3nWyspyauyD37m3nE2GcTtyIaNFnyXwFMgPZb8CNphtgc0QIYu6/egzk0msGdhyxWo48D/PMs5uL9FjFIrK"
    "xiFgG+nE3ck4ipexy/W69W/G2fzL756UZc3dUpYMj21MbAgMyxqiIemKKaw8w8UTaEDJUw7IxXFXklqjOehMaGJT+YDR6G88bPgfG5FTjegK"
    "UBVhR472cBMPIauHwaGz5I6ZuetVBuaNuS1DhFnMkRvj+Z/GdtJANK4EySK28w/n9TTqrkFqwO0LBV9lGUgoubOOUfYM/AE55WD6OEX7HbKi"
    "t22c/5TTmH8UuSXG5YbMZ3LP5rnB52zvi/wi53E0jVEQraX4Yf2ueoOrunlngmR/xW9Sq7+CK/9mcO6RuMn9yD8LJxf7/bg+yNij8kFGTOvu"
    "B06RdfOrbD2MNAWkvtpdfvKcSzVElSQkltI7zoqnKJ23x8CF66vwj3ThszmGexzvdt4CZtrjw4ZcP6YwGUnOGAE="
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock/invoke/us.anthropic.claude-sonnet-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5-5/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={**BEDROCK_INVOKE_REQUEST},
    mock_provider_response={
        **BEDROCK_INVOKE_RESPONSE,
        "model": "claude-sonnet-5-5",
        "id": "msg_bdrk_a7sjc6m6nfuwupssre5yi22kzpdwizomnscfdrysperwjop7fe5a",
        "content": [
            {
                "type": "thinking",
                "thinking": "Since the user explicitly asked for the validation file first, I'll prioritize reading "
                "that before the tests, focusing on the area around line 18.\n\n",
                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I need `shop/validation.py` next. It defines `validate_quantity`, which should show how "
                "zero, negative and non-integer values are handled. I'll read the area around line 18.",
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01LNktDaewoxFywoyzT1eMrE",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 224,
            "output_tokens_details": {"thinking_tokens": 69},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I need `shop/validation.py` next. It defines `validate_quantity`, which should show "
                    "how zero, negative and non-integer values are handled. I'll read the area around line "
                    "18.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                                "name": "read_file",
                            },
                            "id": "toolu_bdrk_01LNktDaewoxFywoyzT1eMrE",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "Since the user explicitly asked for the validation file first, I'll prioritize reading "
                    "that before the tests, focusing on the area around line 18.\n\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "Since the user explicitly asked for the validation file first, I'll prioritize "
                            "reading that before the tests, focusing on the area around line 18.\n\n",
                            "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "Since the user explicitly asked for the validation file first, I'll "
                                "prioritize reading that before the tests, focusing on the area around line "
                                "18.\n\n",
                                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                            },
                        ],
                    },
                },
            },
        ],
        "usage": {
            "completion_tokens": 224,
            "prompt_tokens": 880,
            "total_tokens": 1104,
            "completion_tokens_details": {"reasoning_tokens": 69, "text_tokens": 155},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 876,
                "cache_creation_tokens": 876,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "service_tier": "standard",
        },
    },
)

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSigQKEggSEAEYAjgBQgh0aGlua2luZxIMq3frSfibHHldP8/aGgxVeuAxm7EIGP7vV10iMCQSYTMAj3EpHNaNy9ic8GJApnD9KmWXPGJ6"
    "BqKYlRYlNcvpHFdOGC3d8HHYlYWetCqlA/lyfcAC8HgdVosWlvsFkLvgcAmmV8VUSSJCD7VHV1sC8YjChiQ0bO/DWbfCeK3FUNbDwhAJEstb"
    "q3Pj4THSbsDtJHlIcukGM/xOqXvfjcisMMPjpF8xuK5ai1cnFehLGWHpdj99zC1nK1E/k4OqHYOhzWfHLMQdbeTgR5FP9cVexDyKI/5mlZw9"
    "YxIbOxEjSYAwsR5Q6cIDWjl2ao1M4OhorS/UR1FYp99jwQE5a/zRGOPsFpfPLXR4KCqDKCezAH31LLUkc9iHHgblh5N118kZtGCPRZA8EMmj"
    "HsKZEJ8NdZZfok3Jlrlx0F2773aBBdOsX9MyzopsW4Ap/izR7swjR3hfZy6j/7CnvyghiT7N/4rpIDXiz/2XjVdumk3Rddta3PijbpUpDsMJ"
    "63IJrMcFYLbDypGIdbdHjNyriHlOiwDUy3t4Yh/CwOgw9gkPLKoAVHStFt7YiNnoNCzCGCcbCav7zTCCBZDLdhvbV3QNDHc+b6KkSLsvN+md"
    "u1+4Y6IkqLrhjV7TMledvs22t7vrI5w9M39WGwLztkxmCZbPzfyCWEMYAQ=="
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock/invoke/us.anthropic.claude-opus-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-opus-5-5/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={**BEDROCK_INVOKE_REQUEST},
    mock_provider_response={
        **BEDROCK_INVOKE_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_bdrk_qyqgstng2m3crzmaqzbueka3ep22c7lsslqm5vqoatdm7pi37crq",
        "content": [
            {
                "type": "thinking",
                "thinking": "I need to check the validation file, particularly around line 18 where the user wants me "
                "to look.\n\n",
                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I'll read `shop/validation.py` around `validate_quantity` (line 18) to see how zero, "
                "negative and non-integer values are checked.",
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_016r8yEaJbeHsNj69b8tCYY5",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 45},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 163,
            "output_tokens_details": {"thinking_tokens": 18},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "bedrock/invoke/us.anthropic.claude-opus-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I'll read `shop/validation.py` around `validate_quantity` (line 18) to see how zero, "
                    "negative and non-integer values are checked.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 45}',
                                "name": "read_file",
                            },
                            "id": "toolu_bdrk_016r8yEaJbeHsNj69b8tCYY5",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "I need to check the validation file, particularly around line 18 where the user wants "
                    "me to look.\n\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I need to check the validation file, particularly around line 18 where the "
                            "user wants me to look.\n\n",
                            "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "I need to check the validation file, particularly around line 18 where the "
                                "user wants me to look.\n\n",
                                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                            },
                        ],
                    },
                },
            },
        ],
        "usage": {
            "completion_tokens": 163,
            "prompt_tokens": 880,
            "total_tokens": 1043,
            "completion_tokens_details": {"reasoning_tokens": 18, "text_tokens": 145},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 876,
                "cache_creation_tokens": 876,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "service_tier": "standard",
        },
    },
)

CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0: Final = (
    "CAQS5AcKEggSEAEYAjgBQgh0aGlua2luZxIMZ3CC5ydK6won2yd9GgxE4gMZ+E/WBfdMZvAiMEVQQ4qw23oAzYzAw0ZKeKq2slCfdKWoUJRd"
    "3yCnR0UZgiVPAHOYvnJkBTUpMEzpOSr/BuPzDT9Eso9jzyJX/huQF3Pd7Hwlj+PQFM9uKARSgoG2Dsr1igyT1mEfbp/ePlaLZdxWFpHnHjDk"
    "IiVcVqpUTsJ9XSCQyYAOENmalP+NP+DJ2JkQyaZNf3DIIdDNpOWT1hhq2CdYoHu0yuyUTbeihla45+rF68EY3TplHFoDAatrii0itWJ+cA5b"
    "OHdolQY1UUyVFkhWyxmiOlh0Zt2Tc/ITerLKB0fySyHWmBzwdU27u5Q1kCeOFB/Qx0FMz8X/tWz034Lh17vJsrxud7PNvREUuYfpDRFrr9+l"
    "HEXP2H/BqVEsN55egf1Nc8vu5E5EmBAxU3B7gkJes+PiD/yj/LkwUPDlUZGJnKJB7Vt1ylYcxHaMAfavUHrBLsTbvx6lwyor66phw6v6lQNR"
    "TDsMY6qibpE+H5bY5bpGtdH772Z9LWyHRXG94qNHg6YjH46m/NlocjSk5C4U9i6YqYAPYZrHFO4ELzwuqaNmWXUuf4FoL70DAUeUNPn4dsJl"
    "lKWyrqs/LmrGolJsde8GpwpTlNNRMkm1hhBMpCnjSrzCS9GElJ0FjF4zrzKpcFmVqGYJ7jf1mWA0lnGmUJQK8FPTAIhtQDgSWmh6sDos4A1T"
    "ox12fHGTxflSqi3CIF57+NFmLefxTJLgOEmY8i8yDssZrnDsJC2/1RoKxgEfUJ7a1S6Tx9LWLVwDJJU81FJkCk3R8ZIc3boT7wzlm5Nmxb02"
    "r/NubRSPOdsL5tL4UltVQJv2KMGK2/gIgF5vFB5LM5LtBNtdfLolEd6Nar0Loz8aI11mZ0toCIlo8m9nLdut9wx1cjB4ZqmQQLgaZIiM8ZuD"
    "I07fPCnfgsnSWAdJ2TrPioGSv1Z0wz+vMZMYzB1ebNpl66WlpWJPTVwbjc7YZH7gNKxduopuh6ZGikhtXuCsv7Xm3K4jb8mRW0esr43IjGO7"
    "u4U6bcDAOUwqBa4vR3czuHPSHn2EI69WlPEP1G4Im+DS1G25SJ6VVj4WkYmmid+OEuamPWNYgQ/RUL2nOBSzLFhfBl/R8Dk3ozTOIcooqAB6"
    "9e1vCyVh27iTrRahadf0Zx9EqYjc6anG/lIerWRiFvMhZBjvdSbU6H7Ft54ACQUq4ZVGJt747CgmtS2gmokM8klKqkeXZPxlIcPd6yJvxGa4"
    "MCh929kyGTs1EkoAt8Qu4OR924AtF3KkBQinwisYAQ=="
)

CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock/invoke/us.anthropic.claude-fable-5-1"},
    expected_provider_endpoint="/model/us.anthropic.claude-fable-5-1/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={**BEDROCK_INVOKE_REQUEST},
    mock_provider_response={
        **BEDROCK_INVOKE_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_bdrk_ygn75gjwiywsmw52hhhtugcac2xa3b6qjaoygnm3ihch23zsv4ba",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user explicitly said to read the validation file first, so I'll respect that "
                "sequencing rather than reading both files in parallel. I'll read the validation file now.\n\n",
                "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I'll read `shop/validation.py` around `validate_quantity` to see how zero and negative "
                "values are checked.",
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01MGYCujwiQim9cJXKY33qzc",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 276,
            "output_tokens_details": {"thinking_tokens": 141},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "bedrock/invoke/us.anthropic.claude-fable-5-1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I'll read `shop/validation.py` around `validate_quantity` to see how zero and negative "
                    "values are checked.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 40}',
                                "name": "read_file",
                            },
                            "id": "toolu_bdrk_01MGYCujwiQim9cJXKY33qzc",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "The user explicitly said to read the validation file first, so I'll respect that "
                    "sequencing rather than reading both files in parallel. I'll read the validation file "
                    "now.\n\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "The user explicitly said to read the validation file first, so I'll respect "
                            "that sequencing rather than reading both files in parallel. I'll read the "
                            "validation file now.\n\n",
                            "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0,
                        },
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "The user explicitly said to read the validation file first, so I'll "
                                "respect that sequencing rather than reading both files in parallel. I'll "
                                "read the validation file now.\n\n",
                                "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0,
                            },
                        ],
                    },
                },
            },
        ],
        "usage": {
            "completion_tokens": 276,
            "prompt_tokens": 880,
            "total_tokens": 1156,
            "completion_tokens_details": {"reasoning_tokens": 141, "text_tokens": 135},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 876,
                "cache_creation_tokens": 876,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "service_tier": "standard",
        },
    },
)
