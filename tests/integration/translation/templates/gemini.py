from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

GEMINI_GENERATE_CONTENT_REQUEST: Final[Mapping[str, JsonValue]] = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {
                    "text": "The checkout rejects quantities of zero. Find the validation code and its tests.",
                },
            ],
        },
        {
            "role": "model",
            "parts": [
                {
                    "text": "I will locate the quantity validation and its tests.",
                },
                {
                    "function_call": {
                        "name": "search_files",
                        "args": {
                            "query": "quantity",
                            "paths": [
                                "shop/",
                                "tests/",
                            ],
                        },
                        "id": "toolu_synthetic_search_01",
                    },
                    "thoughtSignature": "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=",
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "search_files",
                        "response": {
                            "matches": [
                                {
                                    "path": "shop/validation.py",
                                    "line": 18,
                                    "text": "def validate_quantity(quantity):",
                                },
                                {
                                    "path": "tests/test_validation.py",
                                    "line": 7,
                                    "text": "def test_zero_quantity():",
                                },
                            ],
                        },
                        "id": "toolu_synthetic_search_01",
                    },
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "Read the validation file first so we can understand how zero and negative quantities are handled"
                        "."
                    ),
                },
            ],
        },
    ],
    "system_instruction": {
        "parts": [
            {
                "text": (
                    "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. E"
                    "xplain which file you need next, then call a read-only tool. Never change files. Use the chat_co"
                    "mpletions client for this project review. Consider zero, negative and non-integer quantities bef"
                    "ore choosing the next file to inspect."
                ),
            },
        ],
    },
    "tools": [
        {
            "function_declarations": [
                {
                    "name": "search_files",
                    "description": "Search source and test files for a literal term.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                            },
                            "paths": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                },
                            },
                        },
                        "required": [
                            "query",
                            "paths",
                        ],
                    },
                },
                {
                    "name": "read_file",
                    "description": "Read a range of lines from a project file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                            },
                            "start_line": {
                                "type": "integer",
                                "minimum": 1,
                            },
                            "end_line": {
                                "type": "integer",
                                "minimum": 1,
                            },
                        },
                        "required": [
                            "path",
                            "start_line",
                            "end_line",
                        ],
                    },
                },
            ],
        },
    ],
    "generationConfig": {
        "max_output_tokens": 1024,
        "thinkingConfig": {
            "thinkingLevel": "high",
            "includeThoughts": True,
        },
    },
}

GEMINI_3_8_FLASH_THOUGHT: Final = (
    "**My Thought Process: Initializing Validation File Inspection**\n\n"
    "Okay, let's break this down. The user's goal is to help maintain a Python project, but important"
    "ly, to *read* relevant files and *never* change them. The instructions explicitly direct me to u"
    "nderstand how zero and negative quantities are handled, and to *consider* these before choosing "
    "the *next* file. The user's explicit request to read the validation file first immediately trigg"
    "ers the appropriate response.\n\n"
    "Based on the instructions and the user's initial prompt, I need to focus on `shop/validation.py`"
    " which I know is at line 18 (`def validate_quantity(quantity):`). I need to check how the code h"
    "andles zero, negative, and potentially non-integer quantities. \n\n"
    "Before proceeding, it's essential to confirm the instructions: read the file, explain the choice"
    ", and use a read-only tool. I need to be precise. To avoid any ambiguity, I'll start by reading "
    "lines 1 to 50 of `shop/validation.py`. This range should allow me to see the context of the `val"
    "idate_quantity` function.\n\nNow, for the critical step: I'm going to follow the instructions and "
    "explain my reasoning, then I'll use the read-only tool.\n\nTherefore, my next step is:\n\n"
    "We want to see how `validate_quantity` verifies quantity inputs, specifically whether it properl"
    "y handles zero, negative values, and non-integer types (e.g. floats, strings, or None), before w"
    "e look at the test suite. Therefore, the next file we need to inspect is `shop/validation.py`.\n\n"
    "\n"
)

GEMINI_3_8_FLASH_THOUGHT_SIGNATURE: Final = (
    "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"
)

GEMINI_GENERATE_CONTENT_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": GEMINI_3_8_FLASH_THOUGHT,
                        "thought": True,
                    },
                    {
                        "text": (
                            "To check how zero, negative, and non-integer quantities (such as floats, non-numeric values, or "
                            "values less than or equal to zero) are validated, I need to inspect `shop/validation.py` next."
                        ),
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "path": "shop/validation.py",
                                "end_line": 50,
                                "start_line": 1,
                            },
                            "id": "call_7668",
                        },
                        "thoughtSignature": GEMINI_3_8_FLASH_THOUGHT_SIGNATURE,
                    },
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
            "finishMessage": "Model generated function call(s).",
        },
    ],
    "usageMetadata": {
        "promptTokenCount": 358,
        "candidatesTokenCount": 80,
        "totalTokenCount": 1013,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 358,
            },
        ],
        "thoughtsTokenCount": 575,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.8-flash",
    "responseId": "1CbQat_JEK-jqtsPtf-QoQU",
}
