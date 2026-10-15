from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

OPENAI_RESPONSES_REQUEST_FROM_CHAT_COMPLETIONS: Final[Mapping[str, JsonValue]] = {
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "The checkout rejects quantities of zero. Find the validation code and its tests.",
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
                    "text": '{"matches": [{"path": "shop/validation.py", "line": 18, "text": '
                    '"def validate_quantity(quantity):"}, {"path": '
                    '"tests/test_validation.py", "line": 7, "text": "def '
                    'test_zero_quantity():"}]}',
                }
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Read the validation file first so we can understand how zero and "
                    "negative quantities are handled.",
                }
            ],
        },
    ],
    "instructions": "You help maintain a synthetic Python project. Read the relevant files before "
    "suggesting a fix. Explain which file you need next, then call a read-only tool. "
    "Never change files. Use the chat_completions client for this project review. "
    "Consider zero, negative and non-integer quantities before choosing the next file "
    "to inspect.",
    "max_output_tokens": 1024,
    "reasoning": {"effort": "high"},
    "stream": False,
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

OPENAI_RESPONSES_REQUEST_FROM_MESSAGES: Final[Mapping[str, JsonValue]] = {
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "The checkout rejects quantities of zero. Find the validation code and its tests.",
                }
            ],
        },
        {
            "type": "function_call",
            "call_id": "toolu_synthetic_search_01",
            "name": "search_files",
            "arguments": '{"query": "quantity", "paths": ["shop/", "tests/"]}',
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I will locate the quantity validation and its tests."}],
        },
        {
            "type": "function_call_output",
            "call_id": "toolu_synthetic_search_01",
            "output": '{"matches": [{"path": "shop/validation.py", "line": 18, "text": "def '
            'validate_quantity(quantity):"}, {"path": "tests/test_validation.py", '
            '"line": 7, "text": "def test_zero_quantity():"}]}',
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Read the validation file first so we can understand how zero and "
                    "negative quantities are handled.",
                }
            ],
        },
    ],
    "include": ["reasoning.encrypted_content"],
    "instructions": "You help maintain a synthetic Python project. Read the relevant files before "
    "suggesting a fix. Explain which file you need next, then call a read-only tool. "
    "Never change files. Use the messages client for this project review. Consider "
    "zero, negative and non-integer quantities before choosing the next file to "
    "inspect.",
    "max_output_tokens": 1024,
    "reasoning": {"effort": "high"},
    "tools": [
        {
            "type": "function",
            "name": "search_files",
            "strict": False,
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
            "strict": False,
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

OPENAI_RESPONSES_REQUEST: Final[Mapping[str, JsonValue]] = {
    "input": [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "The checkout rejects quantities of zero. Find the validation code and its tests.",
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
            "output": '{"matches": [{"path": "shop/validation.py", "line": 18, "text": "def '
            'validate_quantity(quantity):"}, {"path": "tests/test_validation.py", '
            '"line": 7, "text": "def test_zero_quantity():"}]}',
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Read the validation file first so we can understand how zero and "
                    "negative quantities are handled.",
                }
            ],
        },
    ],
    "instructions": "You help maintain a synthetic Python project. Read the relevant files before "
    "suggesting a fix. Explain which file you need next, then call a read-only tool. "
    "Never change files. Use the responses client for this project review. Consider "
    "zero, negative and non-integer quantities before choosing the next file to "
    "inspect.",
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

OPENAI_RESPONSES_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "id": "resp_0207c20ae125feee006ad023aada5c87d082ce8ba652474312",
    "object": "response",
    "created_at": 1792025514,
    "status": "completed",
    "access_programs": None,
    "background": False,
    "billing": {"payer": "developer"},
    "completed_at": 1792025519,
    "error": None,
    "frequency_penalty": 0.0,
    "incomplete_details": None,
    "instructions": "You help maintain a synthetic Python project. Read the relevant files before "
    "suggesting a fix. Explain which file you need next, then call a read-only tool. "
    "Never change files. Use the responses client for this project review. Consider "
    "zero, negative and non-integer quantities before choosing the next file to "
    "inspect.",
    "max_output_tokens": 1024,
    "max_tool_calls": None,
    "moderation": None,
    "output": [
        {
            "id": "rs_0207c20ae125feee006ad023abed4087d09d32a32ad2d3182c",
            "type": "reasoning",
            "content": [],
            "encrypted_content": "gAAAAABq0COvx3R_lOOaYJijiIokUKBq9tt_p3mhoPHnm4KR0Pj_t0wf1sP7WEDHfYSzylhHilYoKjU4guIz-TAqBV6HNUpIx5ggp3aukKEW6zks2ERT3a40sCFFdUdKHk6py5trFuwYn6lHLwifwDkd7rI7MKIP7C7pZSHnXh8cmfk0t96p4nBplQ93C4m917Kyx6PqvWSxPP78h5s_RMEdhkWRU20xMhminpIBNArOkPYT4xsmBEuONlndM5-DHG7ZeLk2yMZw8b_hxk2LZ4mC66b5RHfJKVKfw4egn4K8tkr8mQSR4bnjpY56DsIRj4fjSH5GpoDZyolFFBh1BNkGMoVsicye_4gZn0eELuOWxCw7lVVsMpIzTLw8CHJkCo_8vKtNCFSZcKjOPKDDfX1bsFJ7LMonmqijHPV_kZi_YXjVNNJBhJLiGjYrVUFZzjS0OKQx-RnZf5luO3UM2Ff3Ilvx1eQtuWlsvJjPBRwObzk5zlPEdYwoh5OjUREbgGb4H71Q7A9YYjUIms2h5e13Z5s8X0onRVr9SUcSbbMA0FXWn6QGQ5qgiQV56xec7xk62SrxKBXeILgBttBf68YfLAbdINCPL89G5Z1X2toQLlFM7e0jKYUOvSmpWyPLZ3q1040oDf2Q92v6NjOwk4WlLtZF_4xrZ7bEIWmea7zcDEnsvNBRdKaUbeUOm0SfRuIH4gQfbbueSqM3i12eK4TQhTbJ9VivUUovUv7pCKvizdQuunf2MMXWhq79wLYQYXIleQ06E9hbx6QT-Ue5ELQVC3NSLhXiwucOSqd-NZFDQA5wgW2Uaq_-bFYzNOPwtArfaqnHOILmr5mDvWCrlnLs-DUPHU-DtzYH7Eu1VSlcEZLaZX4vC9ui3x2aOpTtmVCILZpw7MkouVxSR4rFE48doVoCamJHhey-TZ7OMl0A5JdacTt6oUE8Z2HEcuVh9yGAZe_Q-YB3p4egLbxO8ICWUkwSSNgrPq3WRbT9sbT-Ko7N7KDAfgyvzzdEYMKDmci6GBrVsAiscc97-gxbhVhlrMyIXmgTluW07o591YjlOkvERXeJobEfby-VUJCu0cHd0nMudzrhCpHIhU3aUMkk9GMlSNDPFnKeV-gylycLhYRb0xo1kXMRom82fKmvUgO5Wh_BhsCdeGaVQ2bm5wyLT4x9VhEe-uOwaYG02XxoSiIFqqSrf0J8auFHIRpUTKzNWMK_GcIaU6o3ON0wOLKaga3wcyrGkG6wtvGkNLCrvmDIXg0bmvFGZrmybQvys_u636NXVS-Y9irqUv2ghu5tyKI_F8uwibi9JkKa5XcotQGbAToTE8qaHfowZOdTyjIKrU3BjL_Hs5M41KXwrNkk_X5ttJxEPW1qCjyusFb9yeGa9wzHMsp_VO0ysdO8L2KSRAY4LHpGI175VxUmnSg66PtCPln0OYxWsX2FWE1A7B_la662tO4VoS3bcslVZreRWFDJ_7X8qKx2fStxd77B7JkG1ZElxlZYlBRN5_Lo_w1yM9SvqHzzewQYEprSA1gltjxAbIzy7xXReaT86PNI8ojSLfXOkXa6Cybu3hIxsIomkZwTtAEx85BoIDn2B8oTp744Lwkwn4ualAfV9TebiqOWCUYp47fX9rimte71X6Hnvf7tWEBZMMwSb0pTqY0boMOS0w3XKKyEVRNyCoGWMgyvhAxk9E5iao5N6i5W3aSel27QwsUaE4YDGKKMKSryH3BqP6X7d0kdUcAx-nd5yudVbePQyc-UC7EbISj5QXY8g6cTOy-IkwpBe68DvfcNGIhuaH6UscqrewXQjKZFJC_BJSl-ICOfMw-FCTumzCDXPqTeLBPHf1g5kd5lBd6HLPUldHtJfD48V12KCRtaoksZ4g2NibuClnmZRpnDMPDUr7r_D8-RfoLEiJeh8SbOSnLvMPVAJIB95F73B-UX5u0e0aWeyfG5QjBq-kB7vVGsRiMyHEVQW810xo2qULVIyVO03s7rAWwTxFY-EHmKkaAIbKU2b-qAasytni7f1Aq61OSIYqSCRdFhpUX6HpnQeCFC_s6_Q9P3CFx120ZaTB_HPeRJO6dhVSJ3nL3PiNi8sCnhRiNIXkJfm9l1TxRS8ok9s1NY8oi4Y98J4Ay52EkX6f9F9FtWHNu4GX_Z50jDNnVTVqQo3L3slpwxECsTAEU-0MsmIa-1RqJ00VqBLKtIDVbbnQ==",
            "summary": [],
        },
        {
            "id": "msg_0207c20ae125feee006ad023ade23487d093e272f84cb35723",
            "type": "message",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": "I’ll read `shop/validation.py` first to check how it handles "
                    "zero, negative, and non-integer quantities.",
                }
            ],
            "phase": "commentary",
            "role": "assistant",
        },
        {
            "id": "fc_0207c20ae125feee006ad023ae893c87d08068561e36e665ad",
            "type": "function_call",
            "status": "completed",
            "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
            "call_id": "call_S4WtfrQwa6ZLEUZf1BVtA8ON",
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
            "strict": True,
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
            "strict": True,
        },
    ],
    "top_logprobs": 0,
    "top_p": 0.98,
    "truncation": "disabled",
    "usage": {
        "input_tokens": 309,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens": 118,
        "output_tokens_details": {"reasoning_tokens": 53},
        "total_tokens": 427,
    },
    "user": None,
    "metadata": {},
}
