"""Request-corpus builder: OpenAI-format chat cases covering the feature grid.

Each case is written to ``cases/<case_id>.json`` (stable id, committed) and is
pushed through the REAL v1 request transform for every provider in
``_seams.PROVIDERS``. ``skip`` maps a provider key to the reason that provider
cannot run the case hermetically (e.g. v1 downloads URL media for bedrock).

Regenerate the case files with ``--snapshot-update`` (test_corpus_files_in_sync)
or ``python -m tests.translation_characterization._corpus``.
"""

import base64
from typing import Any, Dict

PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAC"
    "hwGA60e6kgAAAABJRU5ErkJggg=="
)
PDF_MIN = base64.b64encode(
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF\n"
).decode()

_USER = {"role": "user", "content": "What is the capital of France?"}
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get current time for a timezone.",
        "parameters": {
            "type": "object",
            "properties": {"tz": {"type": "string"}},
            "required": ["tz"],
        },
    },
}
_PARALLEL_TOOL_CALLS = [
    {
        "id": "call_char_001",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    },
    {
        "id": "call_char_002",
        "type": "function",
        "function": {"name": "get_time", "arguments": '{"tz": "Europe/Paris"}'},
    },
]
_URL_MEDIA_SKIP = "v1 downloads URL media for bedrock transforms (network)"


def build_request_cases() -> Dict[str, Dict[str, Any]]:
    cases: Dict[str, Dict[str, Any]] = {}

    def add(case_id: str, messages: list, params: dict, skip: dict = {}) -> None:
        cases[case_id] = {
            "id": case_id,
            "messages": messages,
            "params": params,
            "skip": skip,
        }

    add("plain_text", [_USER], {"max_tokens": 256})
    add(
        "multi_turn",
        [
            _USER,
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "And of Italy?"},
        ],
        {"max_tokens": 256},
    )
    add(
        "system_prompt",
        [{"role": "system", "content": "You are a terse geographer."}, _USER],
        {"max_tokens": 256},
    )
    add(
        "params_sampling",
        [_USER],
        {
            "max_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["\n\n", "END"],
            "user": "char-user-1234",
        },
    )
    add(
        "tools_basic",
        [_USER],
        {"max_tokens": 256, "tools": [_WEATHER_TOOL], "tool_choice": "auto"},
    )
    add(
        "tools_parallel",
        [_USER],
        {
            "max_tokens": 256,
            "tools": [_WEATHER_TOOL, _TIME_TOOL],
            "parallel_tool_calls": True,
        },
    )
    add(
        "tools_forced_choice",
        [_USER],
        {
            "max_tokens": 256,
            "tools": [_WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        },
    )
    add(
        "tools_streamed_args_roundtrip",
        [
            {"role": "user", "content": "Weather and time in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": _PARALLEL_TOOL_CALLS},
            {"role": "tool", "tool_call_id": "call_char_001", "content": "18C, clear"},
            {"role": "tool", "tool_call_id": "call_char_002", "content": "14:05"},
        ],
        {"max_tokens": 256, "tools": [_WEATHER_TOOL, _TIME_TOOL]},
    )
    add(
        "thinking_enabled",
        [_USER],
        {"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 1024}},
    )
    add(
        "thinking_history_blocks",
        [
            _USER,
            {
                "role": "assistant",
                "content": "Paris.",
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "The user asks about France; the capital is Paris.",
                        "signature": "sig-char-deadbeef",
                    }
                ],
            },
            {"role": "user", "content": "And of Italy?"},
        ],
        {"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 1024}},
    )
    add("reasoning_effort_low", [_USER], {"max_tokens": 2048, "reasoning_effort": "low"})
    add(
        "cache_control_messages",
        [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a terse geographer.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the capital of France?",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ],
        {"max_tokens": 256},
    )
    add(
        "cache_control_tools",
        [_USER],
        {
            "max_tokens": 256,
            "tools": [
                _WEATHER_TOOL,
                {**_TIME_TOOL, "cache_control": {"type": "ephemeral"}},
            ],
        },
    )
    add(
        "response_format_json_schema",
        [_USER],
        {
            "max_tokens": 256,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "capital",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"capital": {"type": "string"}},
                        "required": ["capital"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )
    add(
        "response_format_json_object",
        [{"role": "user", "content": "Reply in JSON: capital of France?"}],
        {"max_tokens": 256, "response_format": {"type": "json_object"}},
    )
    add(
        "image_base64",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{PNG_1PX}"},
                    },
                ],
            }
        ],
        {"max_tokens": 256},
    )
    add(
        "image_url",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
                ],
            }
        ],
        {"max_tokens": 256},
        skip={"bedrock_converse": _URL_MEDIA_SKIP, "bedrock_invoke": _URL_MEDIA_SKIP},
    )
    add(
        "pdf_base64",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this document."},
                    {
                        "type": "file",
                        "file": {
                            "file_data": f"data:application/pdf;base64,{PDF_MIN}",
                            "filename": "char.pdf",
                        },
                    },
                ],
            }
        ],
        {"max_tokens": 256},
    )
    add(
        "full_combo",
        [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a terse geographer.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [_PARALLEL_TOOL_CALLS[0]]},
            {"role": "tool", "tool_call_id": "call_char_001", "content": "18C, clear"},
            {"role": "user", "content": "Summarize, then check the weather again."},
        ],
        {
            "max_tokens": 2048,
            "temperature": 1,
            "tools": [_WEATHER_TOOL],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    )
    return cases


def write_case_files() -> None:
    from ._helpers import CASES_DIR, canonical_json

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    for case_id, case in build_request_cases().items():
        (CASES_DIR / f"{case_id}.json").write_text(canonical_json(case))


if __name__ == "__main__":
    write_case_files()
