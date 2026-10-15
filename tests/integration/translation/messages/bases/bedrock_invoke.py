from collections.abc import Mapping
from typing import Final

from integration.translation.case import TranslationTestCase
from integration.translation.messages.templates import MESSAGES_REQUEST, MESSAGES_RESPONSE
from integration.translation.templates.bedrock_invoke import (
    BEDROCK_INVOKE_REQUEST,
    BEDROCK_INVOKE_RESPONSE,
)
from pydantic import JsonValue

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
    },
    mock_provider_response={
        "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "id": "msg_bdrk_tl6mgwwa5lgxbhipnkc6qb7landewz65mwxm553sydrxicheqe4a",
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
        "model": "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "id": "msg_bdrk_tl6mgwwa5lgxbhipnkc6qb7landewz65mwxm553sydrxicheqe4a",
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
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-4-6",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-4-6/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
    },
    mock_provider_response={
        "model": "claude-sonnet-4-6",
        "id": "msg_bdrk_011CfjukiGyrRwzy4SYisy1x",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help you?"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 18,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 11,
        },
    },
    expected_litellm_response={
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-4-6",
        "id": "msg_bdrk_011CfjukiGyrRwzy4SYisy1x",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help you?"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 18,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 11,
        },
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_bdrk_ccbiw5fsz7qitzhh3dkgpdlk6l6cebry5wwwncia55axys522tfq",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
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
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-5",
        "id": "msg_bdrk_ccbiw5fsz7qitzhh3dkgpdlk6l6cebry5wwwncia55axys522tfq",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
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
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/invoke/us.anthropic.claude-opus-4-8",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-opus-4-8/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
    },
    mock_provider_response={
        "model": "claude-opus-4-8",
        "id": "msg_bdrk_hii3yom4fpd564flttcopesy4ghq7mdw42s4xqmegk2fqmgc2suq",
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
        "model": "bedrock/invoke/us.anthropic.claude-opus-4-8",
        "id": "msg_bdrk_hii3yom4fpd564flttcopesy4ghq7mdw42s4xqmegk2fqmgc2suq",
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
)

BEDROCK_INVOKE_MESSAGES_RESPONSE: Final[Mapping[str, JsonValue]] = {
    key: value for key, value in MESSAGES_RESPONSE.items() if key not in ("diagnostics",)
}

BEDROCK_INVOKE_MESSAGES_SYSTEM: Final[JsonValue] = [
    {
        "type": "text",
        "text": "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
        "Explain which file you need next, then call a read-only tool. Never change files. Use the messages "
        "client for this project review. Consider zero, negative and non-integer quantities before choosing "
        "the next file to inspect.",
    },
]

CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQS0AUKEggSEAEYAjgBQgh0aGlua2luZxIM9tsaCdqj5UGVQgHOGgylOg8RDlmh3f1L79YiMFFshwFAzn8fQTRWSqaR7dpBkcadG+cFDccq"
    "6Cu6MZxJNVXFtqpmgsWFQR5WGlEgKirrBMjA2IESdj0ySte+0ETFtNXPDkfiovCk9en3WHVYW+jvrAIJzOA7Y9oF52rkQUYY/oIeksjLdriJ"
    "3dTogQyx1PR+LkdBZp63sAr+Zn92Q/7/ZBj4k8f6/wlICa/KiRtklkGz/KfLpCJlhzCoxDZ9nqaMKeWLHBX5/UlJ7Lf8j4VrW4Tr9oQ3VRia"
    "iue189vHibmvLtb0lSOoiMq4hEvE2dA0x7RZnLXQ+mULWd7TVnEeQ1NnGt8SNV56ru58gCU6t6Z/KylY3K5H0EEkFImpOigKlNrdDrk7wRxj"
    "hCaFKsc88HUJ15V+3iA0ys3IcZarEFeuDj3fjiKYGtDzaWWDGpmQs7CsKVjAs6YfVh3Eks0KGuDppfnM8QyvmUz8y7aIkeuuES1u9F82g3NQ"
    "3FoReR4wwdy76xypUHY415wLODfAiSoJjqsI9DArNQc//8UeQX++ECLuvwNm72CgbcN7L13BznbeSQdV1Un65uXS2wT+1eGvDBgGuwuXgvY5"
    "ri2U5YgvnowPboiziBMaUxKNEhNouvD+l+p1bxalC03JcCUCA1MJzqAafufTscOdR1mTSl5V2+PPHz964pvxDCMRR0n+nBGsc3obeoS+9Vqp"
    "yRG2lrPQj8t9GYzL93C2gbB3o0mdBB2dZTlKPxTnkCanL+1PwLjuKiy7lklHNo/upTP84kISsX+pOSrSNRW//BDbC5wkqgXHx/GsVL9OfrBe"
    "P7f4SmUh6VWykR44kEO7vfR2LF1N3LRsHrI4OuNXlh6AbxShwGMvdimckzK5IjCTPX6DKQBXWmlmNJyeNzw2U37e44S9+vhdz5TBhfcYAQ=="
)

CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "bedrock/invoke/us.anthropic.claude-haiku-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-haiku-5-5/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_INVOKE_REQUEST,
        "messages": MESSAGES_REQUEST["messages"],
        "system": BEDROCK_INVOKE_MESSAGES_SYSTEM,
        "tools": MESSAGES_REQUEST["tools"],
    },
    mock_provider_response={
        **BEDROCK_INVOKE_RESPONSE,
        "id": "msg_bdrk_wzz4hwc4dbdg5xyblodt4ucbo6racirmvwxrmm7knfhbqy4m2tfq",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll take a look at shop/validation.py to check how it handles edge cases like zero, "
                "negative, or non-integer quantities before deciding what to review next.\n\n",
                "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01SAJFKxZ5AbvBAPai5cpFSs",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 185,
            "output_tokens_details": {"thinking_tokens": 85},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        **BEDROCK_INVOKE_MESSAGES_RESPONSE,
        "model": "bedrock/invoke/us.anthropic.claude-haiku-5-5",
        "id": "msg_bdrk_wzz4hwc4dbdg5xyblodt4ucbo6racirmvwxrmm7knfhbqy4m2tfq",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll take a look at shop/validation.py to check how it handles edge cases like zero, "
                "negative, or non-integer quantities before deciding what to review next.\n\n",
                "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01SAJFKxZ5AbvBAPai5cpFSs",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 185,
            "output_tokens_details": {"thinking_tokens": 85},
            "service_tier": "standard",
        },
    },
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSnwYKEggSEAEYAjgBQgh0aGlua2luZxIM5Rdi0bxwZPzY53K/Ggz5v0FGebtekQAdYi0iML9pU9O153URlvQmE61tNy33MyzsPW3gebf0"
    "uYHHYrfZvthRnZNKNSSJpftkUOnBHiq6BQpuaRBOmr5jlQmx/nra8YP41PjKMwpGTJHkjh2COy2g0Ex2ZJk2Zi5A4jdms8plNf0vG3bXg01s"
    "woW+wMC8KnoLBoDYQ5rrR+kSVwQUUir1/UMP0Si3Ep96qLJ9923Tf8ypQe/lTP5Ta/iWJ8jEc5SgMp/7HquIvNxyz3efFSyhUF/82y1WwPdQ"
    "YcITad5XfHA/gz/fdCOGCmYJndI6aauKBHXsq4dk+B3ZDbSr1URb29NolM2lWfHb3FngvRknZoHuMX7bVz6hrt2gsSRUoEaME/JNvf+L430t"
    "h/lhfk5dHljt8YVf6wI4ZMAwty5ViM43M4In1AEaTJ4VEd9VfYgZ48zpxRqjRFXQvBORAqoOEempYRDphZZcPuxjbrcqW2fzwMDH+rfMZlGu"
    "ssDEySgfxj3/ODN0W9US6zdQqJ1YgaOBTH+TEEmcsw0Y5Mq1qioYPeLJjVj9FVLjpwf+9xGBUGoTi5+20T6/o0R7KOMl7iUx7I7HanKoIHHQ"
    "uAl/eFlBNSB7nlv7L7i1VC8VFTQ83mk9vVyg74guzcidr8QNn6lCFh8b2sGEUj5uEYLoRFhSAa7uw0ja6cyLOw/62OwXKlJaoV8Yy2awvYYY"
    "/ojYlskRB31Ux+wVcSX4J4RqKUwz/067iQecmpXYKMSB6mxc6ig9oyZYbqNd0w4x7L129svgFAY0k3cDS2jUePia9Y+SXJgCNXgjU8BLMj82"
    "NUy6WNtjCBXb89yxHWEhXdyNQEtOruc/zWDM6pdHnMsbbgK8q+anyttH6wFsorXwLfJnPTIR4dYujPxRph+ac52uHbhMQKtNB9N0Hd8kT8E+"
    "kq9RXXwYjGoMj3CGc5KN0PELEpujCkx6omfN+6yMCJntks/78q4Zv1coVeb4SP22doPtMyjPR1wbBVMCoM+6cPF766uvFCg+J2oDGAE="
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "bedrock/invoke/us.anthropic.claude-sonnet-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5-5/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_INVOKE_REQUEST,
        "messages": MESSAGES_REQUEST["messages"],
        "system": BEDROCK_INVOKE_MESSAGES_SYSTEM,
        "tools": MESSAGES_REQUEST["tools"],
    },
    mock_provider_response={
        **BEDROCK_INVOKE_RESPONSE,
        "model": "claude-sonnet-5-5",
        "id": "msg_bdrk_rcnbkubesv2zarg2r6uit3hne2znqytllke3i5jo6sm6eh7ml7za",
        "content": [
            {
                "type": "thinking",
                "thinking": "Since the user asked me to look at the validation file first, I'll start there and read "
                "through roughly the first 60 lines to get the context I need before considering the tests.\n\n",
                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I need `shop/validation.py` first. It defines `validate_quantity` at line 18, so it will "
                "show how zero, negative and non-integer values are handled. I'll read the tests after "
                "that.",
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_012x6zRfbMwPWH1TvwvtJcTd",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 257,
            "output_tokens_details": {"thinking_tokens": 98},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        **BEDROCK_INVOKE_MESSAGES_RESPONSE,
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-5-5",
        "id": "msg_bdrk_rcnbkubesv2zarg2r6uit3hne2znqytllke3i5jo6sm6eh7ml7za",
        "content": [
            {
                "type": "thinking",
                "thinking": "Since the user asked me to look at the validation file first, I'll start there and read "
                "through roughly the first 60 lines to get the context I need before considering the tests.\n\n",
                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I need `shop/validation.py` first. It defines `validate_quantity` at line 18, so it will "
                "show how zero, negative and non-integer values are handled. I'll read the tests after "
                "that.",
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_012x6zRfbMwPWH1TvwvtJcTd",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 257,
            "output_tokens_details": {"thinking_tokens": 98},
            "service_tier": "standard",
        },
    },
)

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSigUKEggSEAEYAjgBQgh0aGlua2luZxIMo+bg7J1athsROSzTGgz8CIhzKVCiV3U3zR8iMFgRZ0GxT3RyB90ZBPfccjWj9SmeP5Su7x5K"
    "v4XcUof6Yw7d+Q31UhW8CTZdeivdDCqlBOkneYlYuN0wm6pkFQG8Cg+ThPVN8HGn/QiZPBihOr5ZJK1joYur01TDHIYWzKuBOHJ/fmYfdgoG"
    "I0d6WAsw3FdL4zsbhWSEHbbD8+RFz/wk/LqOdaQNYS4I42fytuO9+i5bVthVB3UVl0QEDxPw9s/4c/RaFMQpIyuCigqyJpZC6UKFCtAPHgPW"
    "ASOJtP7s0Yrv+S6MBqJOP3oic95i1IPy873PVLJu+mmyKp8yWRqiK8EN4oNE1HuXKqpvPhm+aCKQK7rUJgtoTxNfpiGeZCDaA8am0ysM1M5y"
    "2Zi1V8N4sKDz2+SbEDF8EaHp6eicIZ9f6Z5xII7tR/xS2gP7ZNvSt+HNC7pYoQ5rA9xEWO541RvtcloVj3D7OWeaCLgzP3rsp+XIrNMw5017"
    "+98q2G3FAHiCa25VLM3nUyxcG9yi6t6AQFl4ISByj6oh22xrcQJ7b+fiLabG48PtdeVZip/LUAeWm1Dw6e3Vl9kmRZi07/gT+bCp3FhnOHSW"
    "bkZKuWB0LWEWa1dBWWNllTDgc2t+jnOL9hVu3j92quUcauOQXA1de746z0l2A8M0bl+Vqf4dWrBk/rvdu5uumIUqgEk3YM8ClKy7m5ylOmct"
    "QI8Z9Rsuvt+ianHftrp7VckHdX4BFxmvvtjC+7NljF+aOUJbZpasecjeLT5hf72Xh/OTxKCoylsxrPYbFvT06ZAM12sLHiQ4II+FtF9nXvQl"
    "xBY7c4C7qxgB"
)

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_1: Final = (
    "CAQS7wUKEwgSEAEYAjgBQgluYXJyYXRpb24SDIDG/3Oi4Lc/e5FR9BoM/dTQTa3LvgbUByp9IjBRgDtuGyPNi3LkPeVXyLT98JrhBRqJktyj"
    "WMovEVaNb0VsiF+Q3xFd1i/Mtp/aHX0qiQVkW5ID1q6VrZOWTLrM7vI2Rz4WYUV8fOSa7n8kamwNbwrEKY+uQ7vsgjge+f0Yr9bllDitTOsg"
    "P19+CxdUmgLXTva7uu5w44S+KtSj+sSmYVW8Wl/S1DWBKVxL/8pXeJf6t6vQlHMu58EYfW2pY43v+7xUPNBRU+l4+UfBz4DRZHAJpQq82e5z"
    "QHjdyU6nSqcxlE1zDrAmmiSiCskbg6CwH0VAredEgdVwRFrdCK+A9NiqzDRSASmZu2VYoPZwgWdd/ufIJ2oqH56p+6JszgX92U0d9yJ8o6uw"
    "fvFrP08YYayBhJn7Ba2lpDTdlCtYLT9qsLyXwQ4NIwPr36nzBbDWhRhF5Z5sGdA8vRnyue/CYpbLdI8Qg3JiXaXXTsGhDvbAmX6x55PhXiag"
    "WaQSsQ6UzsJIdSUdjeD5bnrJ841oSnsG5qKaUfz+yr3cRn4ZnIHqDbZ4o9o/0lQC4++WHe2VddIIET3vVITKF+jKUsHo62vr5nx6tyUj5dGj"
    "ZMyFsEMYmf4b9hgij84/mgCOGLNQ5Nu1PM1GKnMpm2dEaosjmSB14pTkuJCzUWGrLANWcgKJRRfv1Wy6n1xaORcm8w7mIWaYbqcxBPddTVOi"
    "sXQ2Gr5pZRsbmcMnYcsDF/LejdLT7m1w4WEeOG7B7kXue2FkVaBq0k5K/mgGuIpdmhuwEYuzA5nf+E91Ik0+ktTT5tX/LfhlTc1RqIT07G86"
    "ufm8yKpp6Uc7PvWXGm7tR+/70sN5O4ltSiamehskjtAHj1ev3zc/dQJ3oQP6ljf1AY7/ZB9v6YgLY1c0tZ+cHUQ/0+ND+7sraZu/Xiihggfc"
    "s8SMejElcBxTWV86zG2ijctI5UlotIclfNMiGAE="
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "bedrock/invoke/us.anthropic.claude-opus-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-opus-5-5/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_INVOKE_REQUEST,
        "messages": MESSAGES_REQUEST["messages"],
        "system": BEDROCK_INVOKE_MESSAGES_SYSTEM,
        "tools": MESSAGES_REQUEST["tools"],
    },
    mock_provider_response={
        **BEDROCK_INVOKE_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_bdrk_twd7a5k6nmlhwudxctflvx44se2jpvhjrwimhwul2gwzgnni3zsa",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll start by reading the validation file around line 18, as the user requested first, "
                "before checking the tests.\n\n",
                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "Now I'll check `validate_quantity` in `shop/validation.py` around line 18 to see how it "
                "handles zero, negative, and non-integer inputs before reviewing the test.\n\n",
                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01Kf2DAxymu3xTnb2AwxZZgB",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 221,
            "output_tokens_details": {"thinking_tokens": 50},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        **BEDROCK_INVOKE_MESSAGES_RESPONSE,
        "model": "bedrock/invoke/us.anthropic.claude-opus-5-5",
        "id": "msg_bdrk_twd7a5k6nmlhwudxctflvx44se2jpvhjrwimhwul2gwzgnni3zsa",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll start by reading the validation file around line 18, as the user requested first, "
                "before checking the tests.\n\n",
                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "Now I'll check `validate_quantity` in `shop/validation.py` around line 18 to see how it "
                "handles zero, negative, and non-integer inputs before reviewing the test.\n\n",
                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01Kf2DAxymu3xTnb2AwxZZgB",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 221,
            "output_tokens_details": {"thinking_tokens": 50},
            "service_tier": "standard",
        },
    },
)

CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "bedrock/invoke/us.anthropic.claude-fable-5-1"},
    expected_provider_endpoint="/model/us.anthropic.claude-fable-5-1/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_INVOKE_REQUEST,
        "messages": MESSAGES_REQUEST["messages"],
        "system": BEDROCK_INVOKE_MESSAGES_SYSTEM,
        "tools": MESSAGES_REQUEST["tools"],
    },
    mock_provider_response={
        **BEDROCK_INVOKE_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_bdrk_gh2vug2tpleguniqdb6ry6otz5y4x2jxg3lneudi2hx4wrv6xp5q",
        "content": [
            {
                "type": "text",
                "text": "I'll read the validation function around line 18 in shop/validation.py.",
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01Wr8tPMkYit6TSkCn7oBKcr",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 122,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        **BEDROCK_INVOKE_MESSAGES_RESPONSE,
        "model": "bedrock/invoke/us.anthropic.claude-fable-5-1",
        "id": "msg_bdrk_gh2vug2tpleguniqdb6ry6otz5y4x2jxg3lneudi2hx4wrv6xp5q",
        "content": [
            {
                "type": "text",
                "text": "I'll read the validation function around line 18 in shop/validation.py.",
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01Wr8tPMkYit6TSkCn7oBKcr",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 122,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
        },
    },
)
