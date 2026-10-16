from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

AZURE_RESPONSES_REQUEST: Final[Mapping[str, JsonValue]] = {
    "input": [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": ("The checkout rejects quantities of zero. Find the validation code and its tests."),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I will locate the quantity validation and its tests."}],
        },
        {
            "type": "function_call",
            "call_id": "toolu_synthetic_search_01",
            "name": "search_files",
            "arguments": '{"query": "quantity", "paths": ["shop/", "tests/"]}',
        },
        {
            "type": "function_call_output",
            "call_id": "toolu_synthetic_search_01",
            "output": (
                '{"matches": [{"path": "shop/validation.py", "line": 18, "text": "def '
                'validate_quantity(quantity):"}, {"path": "tests/test_validation.py", "line": 7, '
                '"text": "def test_zero_quantity():"}]}'
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Read the validation file first so we can understand how zero and negative "
                        "quantities are handled."
                    ),
                }
            ],
        },
    ],
    "instructions": (
        "You help maintain a synthetic Python project. Read the relevant files before "
        "suggesting a fix. Explain which file you need next, then call a read-only tool. "
        "Never change files. Use the responses client for this project review. Consider "
        "zero, negative and non-integer quantities before choosing the next file to "
        "inspect."
    ),
    "max_output_tokens": 1024,
    "reasoning": {"effort": "high"},
    "tools": [
        {
            "type": "function",
            "name": "search_files",
            "description": "Search source and test files for a literal term.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}},
                "required": ["query", "paths"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a range of lines from a project file.",
            "parameters": {
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
    ],
}

AZURE_RESPONSES_BRIDGE_REQUEST: Final[Mapping[str, JsonValue]] = {
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": ("The checkout rejects quantities of zero. Find the validation code and its tests."),
                }
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I will locate the quantity validation and its tests."}],
        },
        {
            "type": "function_call",
            "call_id": "toolu_synthetic_search_01",
            "name": "search_files",
            "arguments": '{"query": "quantity", "paths": ["shop/", "tests/"]}',
        },
        {
            "type": "function_call_output",
            "call_id": "toolu_synthetic_search_01",
            "output": [
                {
                    "type": "input_text",
                    "text": (
                        '{"matches": [{"path": "shop/validation.py", "line": 18, "text": "def '
                        'validate_quantity(quantity):"}, {"path": "tests/test_validation.py", "line": 7, '
                        '"text": "def test_zero_quantity():"}]}'
                    ),
                }
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Read the validation file first so we can understand how zero and negative "
                        "quantities are handled."
                    ),
                }
            ],
        },
    ],
    "instructions": (
        "You help maintain a synthetic Python project. Read the relevant files before "
        "suggesting a fix. Explain which file you need next, then call a read-only tool. "
        "Never change files. Use the chat_completions client for this project review. "
        "Consider zero, negative and non-integer quantities before choosing the next file "
        "to inspect."
    ),
    "max_output_tokens": 1024,
    "reasoning": {"effort": "high"},
    "tools": [
        {
            "name": "search_files",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}},
                "required": ["query", "paths"],
                "additionalProperties": False,
            },
            "strict": None,
            "type": "function",
            "description": "Search source and test files for a literal term.",
        },
        {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path", "start_line", "end_line"],
                "additionalProperties": False,
            },
            "strict": None,
            "type": "function",
            "description": "Read a range of lines from a project file.",
        },
    ],
}

AZURE_RESPONSES_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "id": "resp_03952e888f4f1827006ad023aa00108190b24e51d376e9a373",
    "object": "response",
    "created_at": 1792025514,
    "status": "completed",
    "background": False,
    "completed_at": 1792025519,
    "content_filters": [
        {
            "blocked": False,
            "source_type": "prompt",
            "content_filter_raw": [],
            "content_filter_results": {
                "hate": {"filtered": False, "severity": "safe"},
                "sexual": {"filtered": False, "severity": "safe"},
                "violence": {"filtered": False, "severity": "safe"},
                "self_harm": {"filtered": False, "severity": "safe"},
                "jailbreak": {"detected": False, "filtered": False},
            },
            "content_filter_offsets": {"start_offset": 0, "end_offset": 575, "check_offset": 0},
        },
        {
            "blocked": False,
            "source_type": "completion",
            "content_filter_raw": [],
            "content_filter_results": {
                "protected_material_code": {"detected": False, "filtered": False},
                "protected_material_text": {"detected": False, "filtered": False},
                "hate": {"filtered": False, "severity": "safe"},
                "sexual": {"filtered": False, "severity": "safe"},
                "violence": {"filtered": False, "severity": "safe"},
                "self_harm": {"filtered": False, "severity": "safe"},
            },
            "content_filter_offsets": {"start_offset": 0, "end_offset": 578, "check_offset": 0},
        },
    ],
    "error": None,
    "frequency_penalty": 0.0,
    "incomplete_details": None,
    "instructions": (
        "You help maintain a synthetic Python project. Read the relevant files before "
        "suggesting a fix. Explain which file you need next, then call a read-only tool. "
        "Never change files. Use the chat_completions client for this project review. "
        "Consider zero, negative and non-integer quantities before choosing the next file "
        "to inspect."
    ),
    "max_output_tokens": 1024,
    "max_tool_calls": None,
    "model": "gpt-6.1-sol",
    "moderation": None,
    "output": [
        {
            "id": "rs_03952e888f4f1827006ad023ae0c048190acdee29b5d20c5b3",
            "type": "reasoning",
            "content": [],
            "encrypted_content": (
                "gAAAAABq0COvgomuqwo0Z3KFAStLOih2JmYOubPtUjtpD1zCRSCXdDnHGkj8asyVemaCsQr8szo_esDZ"
                "PIT2Y9ghPTx1d5moHnS_lyX4nn3Almf4SYZ1pm8maTgEdntGFzcNtPRwW70-CgpeKe5z8zTwxsxyyKDb"
                "OYTjt4hjUbAa4pM0K9JR9u5Q_P-LKgfcv3SNCeNSMs5lSV1aBMpuWEGJrlYF020C4wtmru3lH9VjQlOn"
                "nW8uD1SCcPm7Sp1_Mm16PJBX4j2grbcIOJZDqj7OcjzRMAb2Gw96QfIg1_efR4BDVHzACkDClzPaiWuI"
                "rAmGMA8bJDtQBPijZN3Hahu8eMoMd-XZDVQPx_7XM3F5OWUiuos_kEZNAhIIQHcLrXvyyf7slWvbT4Af"
                "c8bvO45t1wOu5E_ALLhWcUCvbfPKBmHrlsRtBEVd_5n0q8jRwR1e5nyjPhMo3YsHvPmwHUJ5Gi1DZycD"
                "-FDj6NpwlLcV3eeKaD8c-wNnHuXwfzi_KzIwU1phF_QxJ3ofegtM-3HwCqgaxxVFEhtIh3yezUKeqC7R"
                "fSSHpmI-Q59TRf6-R7h6NFJKMG4Hg6C5twvVloHV2YMUHfBrgWr44X9FTzEwxJ89niDRtaB9Xvo2Qzme"
                "9LUF-CWuGYTKj5QyRUQFOtclmBL335YTvTDWT8V-eN6KTE33XQNDEPSKuecES6fGHVCuIOeyQkWkvrW8"
                "O2OrPJAmoW5g9AFWSWbeLWmDTTav6vbTyu0Ef9cvaRHRTOMq6q8VlEQoTQkJNUM8OKguY7Ad-1VSoIFW"
                "D4_LvhuRsDsQiBu1diEq3hRNhXssBUd5wkFLOXw7UITRHONHhk7TDmVh0yjOwBr_e323vmL2e4BouKoX"
                "x_0Umeb1jkjFlGgmpW59LmqXd91aXxjioYLxVMOMC7ib7O01_dD-OvWavSxAy9GLnex06mgThS2T5_u4"
                "MRmpXIVVqdniqvoxx6YwX5F3TwRoIVc511oXyOG0QkKVLD1k0pI9OdFjiWNVH6XuhWPpVnAhqR6KYejn"
                "hQQwuHA4KV4Nd6wOMFbhPjKPCuRcaN1vbgTbRJlwD8O2OV537hjDq6TQaezFJMk3HPK5bB_-J3kmUCul"
                "YOXw0jgw-Ze8EmKLDZoCz3jq4NjNA9s0sLGd_Xg0kR6GeOmpLMWtT024yQ7gowGendkcHjIw3S-fIfiI"
                "zJtZ-Hk9_iMBk8FE5Syb-j0kmzGF2gFj7ywTTdGoMsM3aTOvusVL14D1N_IfMryUdeoNJ41G8SvqGjtH"
                "gyqCER9grGItYgKueDjt6IC5I086__HogkmrpzLZxSIpFCXX6rArVDsIYdS9XFgLW1oBKsVUs5DLnIYe"
                "Ncv_cC-oTvEDBaYdhDPYbn_5N6q0q7dtbvohO-XJxOfLh0jC3TelEn7m5EXMPpAygQP2sPtI4_SmhA1C"
                "Tji-B-6J60kBUiKmp65ZVxx8BDMI9dv3PLxsw4R0zh6fP_Fsok0UKJOQlfC6k47yhDuTcdN_vq7sAc3q"
                "6P9b7Ud7Bz3JtY8Lg516vLK2vti4iwu7dcHlNiIoNZTGMANKYccO-XSRuJbX09bypgJB9sSR56_-Cb1i"
                "b1C5RtkTv3ueRvq58Hw1RIINmiTTJn2MiYi_P-yXAirR39YTxcyRNUk4UDgK1lK73lrnpiBELJe-CPQg"
                "wgLlalzjk8j0CnIMUA=="
            ),
            "summary": [],
        },
        {
            "id": "msg_03952e888f4f1827006ad023ae6e608190afabb95cbcebf3c9",
            "type": "message",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": (
                        "I’ll read `shop/validation.py` first to check how it handles zero, negative, and "
                        "non-integer quantities.\n"
                    ),
                }
            ],
            "phase": "commentary",
            "role": "assistant",
        },
        {
            "id": "fc_03952e888f4f1827006ad023aeb68881909fd9d88d37ef120a",
            "type": "function_call",
            "status": "completed",
            "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
            "call_id": "call_mWfqifVeCXVWLiiNM1Ac2MjL",
            "name": "read_file",
        },
    ],
    "parallel_tool_calls": True,
    "presence_penalty": 0.0,
    "previous_response_id": None,
    "prompt_cache_key": None,
    "prompt_cache_retention": "24h",
    "reasoning": {"context": "all_turns", "effort": "high", "mode": "standard", "summary": None},
    "safety_identifier": None,
    "service_tier": "default",
    "store": True,
    "temperature": 1.0,
    "text": {"format": {"type": "text"}, "verbosity": "medium"},
    "tool_choice": "auto",
    "tool_usage": {
        "image_gen": {
            "input_tokens": 0,
            "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"image_tokens": 0, "text_tokens": 0},
            "total_tokens": 0,
        },
        "web_search": {"num_requests": 0},
    },
    "tools": [
        {
            "type": "function",
            "description": "Search source and test files for a literal term.",
            "name": "search_files",
            "output_schema": None,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}},
                "required": ["query", "paths"],
                "additionalProperties": False,
            },
            "strict": False,
        },
        {
            "type": "function",
            "description": "Read a range of lines from a project file.",
            "name": "read_file",
            "output_schema": None,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path", "start_line", "end_line"],
                "additionalProperties": False,
            },
            "strict": False,
        },
    ],
    "top_logprobs": 0,
    "top_p": 0.98,
    "truncation": "disabled",
    "usage": {
        "input_tokens": 312,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens": 154,
        "output_tokens_details": {"reasoning_tokens": 89},
        "total_tokens": 466,
    },
    "user": None,
    "metadata": {},
}
