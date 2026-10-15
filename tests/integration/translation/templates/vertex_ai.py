from collections.abc import Mapping
from typing import Final

from integration.translation.templates.anthropic import ANTHROPIC_MESSAGES_REQUEST
from pydantic import JsonValue

VERTEX_AI_ANTHROPIC_THINKING_SIGNATURE_0: Final = (
    "CAQSsAcKEggSEAIYAjgBQgh0aGlua2luZxIM5AdGqr4T05oHxpggGgz1Xe5nFTdJd/ZREp8iMMOP/ne3HrVXil/Qby25EvgVLzla"
    "PlAqV9EuGZhZueHwbwVDEAZISWNgb6af+m03KCrLBoxdew1rYrrcHnuY8IsIpAvjDHAWs/wAS1J8xCNJO8pAWdDUkXz9je0gdBkP"
    "5EGORxwGsDr2ATCotY/rWi51A5rSI07lMVVEYjko0F1xbLOEW0Q/H76nWrp30p5v6HfxDmxBg3WvzYmu45onzgN/Q0R13KPzAy5A"
    "4RWV3ANP+SRwJFQyTSFLqn0O1dNnd7eENp2TvBaV6ruF+qbrEM3Bdfc3XkyCSQMn6RSSNM+LkK6VrMdjOOtQZybFHgmOC+ksFqFJ"
    "+nBeLOS0wI2QbaIVVR7k5yFGoGW3DLlIFDSKTzOPnhLJMdSMvLBpf156LuUkjk9cHzgriw8kyZJbVtuw/Myvwc5u2ZQsuAfuqwd8"
    "f5Ewf6OPt609o+C0G9dzKlt7jV+ZwVfMIm+MbSO0nxVaFMgZwBFrqcLH/+ie5m2QCBjVJXMUcc4dy2u7NfYxN1UzLJjdHzp/EOox"
    "2mieSdI0Llx9sqiWmJRlt73evYTY34Z0V29dFOaJGH6ZecuFj7JnaR4EhLTZZJixesoYKy6b8wJB0ZsLKsQ9N93Kbxc4ODzBZnad"
    "Nbxx5JH/weFk5MDKBtIiCoBXusFAvDO5xO7G/iuO3Rq1MB9P1GFf7e+CB6Y5Ej/5yL1nCivCrEd4tYJUnl3DCjSW2QuV+JqwmclM"
    "08RgVQKJ50OXYQNgev8aO1JSrs8NpmtodrUrgpwLDSqetMTsm8LkpqZadc+73gGjAscm4o9Q4+m02gpqLklXxgS+QxN8PBTNDC42"
    "VJzXowlJJHRXoczxrt54cj/iY8VEcP3o0Y99zNIyvbvCyZypnrkTfYlEQo+1+c5tjcUiMGaXb26UZpr91VO6vYqS06lgLIWYjlf1"
    "qF8Ym+EqPRVZsz/YZuy+7laG/9nF4qAIMWqASxL2BFpdPrhePcFrlyefEU3zt5Vl8xvR8K4zv9Glmtd+3mVIwDc6fSAwdm9rWtzg"
    "92Mq8LzeyxrNcLijq8iWK1CnFK3ayPJgSicuEhXgD0JpgltQh+lWZ6a8M71+/SGyJJb8hQp5x5wpEKFgGKTCBvl/oV0Op8/Ie44I"
    "IYHkm/aOvu+WSnHwU4IrcHwPBSUbk2pYKaWQb1cRlDfXWmuaHluOfiRPo0ORvzOdehgB"
)

VERTEX_AI_ANTHROPIC_THINKING_SIGNATURE_1: Final = (
    "CAQS/QUKEwgSEAIYAjgBQgluYXJyYXRpb24SDCLtxRBlqq3WVMEpRxoM5UkHHwV72f1gxZ0HIjB4+DqdmR5Sfg3wso78vOXyUJMZ"
    "H7w/XBD3JSH6I78plRQCkTuRWhd+umfXGpb9if4qlwXOtx5CQ1E+AabA3vqpLJarGaCFHO8HclGAk37uPKxmyKk7hSHvMyn3ZIG7"
    "aVPOT7f42kI2Kif4/xhMrZ0jANrgbxsT2I+UfbFVXeOaKFjkG3Enn46IXL11N+4vP8RbdGESWDxcWDyLmp0O74W825FHNsrf4tNA"
    "mfH19Mq1jXYC//8fsz01HC7mlsTGFQShBdbblbKcV43Mkdsq1OV1EOkUZf5JX1AKr1D2Ywsp11aoU//VfXb2Bq+/o8E/2ZRupoTL"
    "d1Sak7H1aNdfJ+5mfDHGLOrlPjxunSxxksfrZopjVFs8fN/OsSX2FjCDeHskShGoAF/75pfDHvQl+vsZgWgprTZ7nZ2Sirx3Asui"
    "670zquqc02ke2ZBxFhVvbfXikAm37QK3sSyzMp0xtXx7+B5TqguHMBPcZ2FwMkkejyM/2c8p1UnoimAjEnD7VxWuEgmg72cL62Qb"
    "TN1W6qRf50CQnC7yrTh+23804bQ94kea4mobxRPcveyZVCtlcwgKcvFzWSqZ+QYVORHbnBkKIQlUZB81VT7NxM54UN2Lu2clNu3E"
    "/6r+0pYrbmJazAtcaT1LLFS5wrfl6ngHGTDdPGIwQPZRclrL+tV3YGApuwauC1QQRD3NG8XAuHhEbtJcxOBmGx1ctOsHiRrhoFjM"
    "MQ+/qNprXLGy/JWeSOdJdwpWOIuqKf8g1LH2NDxbMHOnkIN+UbQBBCbMDdV8oLbpu7n8+moK5ggx4sQ3sWcn9/aBl9VTUC9Qjk3/"
    "GTNtZNGzlWBVn7w/JnYZINjy40DtReIHdqcZja5XKnYpzfh7G6EcYA7tW4SmtWb2sPB1Msz+S8FW944iQ+AyfZQqic9b8BEw2JYG"
    "it0butjvG2EWi4LnL966v5WCZQAYAQ=="
)

VERTEX_AI_GEMINI_THOUGHT_SIGNATURE_0: Final = (
    "AY89a1/K8QT9La7l3gDc5jImwnXSF0xUuyg8rAUnAyYS4c2/Yjd/fV3Ve1tHSww5bSooGMdZ0oqljadXXxUqYi+7fkKOWD0Ij80K"
    "LFu1O//r13u8cIpIqUxEi3lNiEn8rK6WVW5DPTMTBddZBL+/DcpCEi7U3m5vGYfoJUm9AJwQB8XDRdqLdA1hfiRnCqGpLJsFAQ2g"
    "Ln3FpQUPR3+FRoWRlJnNRjk8vWaq3mBKHSHttLMRd3rpvo6VAAZbs2OGoBJXFXA9S+RyN532kDK3fZIGGW2Fk+FSD9EJ1hN1+Qeh"
    "v8gzcxyZOin2O/ONFWEQ4zf8lHTzaostuIMLkKFM3v/ssK+Y4VgS0sPFr7VFByhczwoDx73oh141iDD5ch7tqAgnR7YikrWQDuV2"
    "8FBSB7+MQ3BrBSt6XUvYgcHsGNbg0giE3D3pFDR7uvLfvgjhFFdIqnCWWmPfBgZ7GOD55unG0IUPA2udP65xg88r8TjfpQOUsOnN"
    "KZLOiKovzHrXZX2L0tRGha5nE8ICi427Mc5YF9lm0Ilpo/BrGuuusMG3Zx9RxRyjRa9+zWPANRrvuq7SOVnFvAss4uu9PYf5Orwn"
    "nGFJ0xjP1hn22yXOxRYy62ySQQ1ValpWXjPPXvGKZRiUArCuvrBvDEZoUfJhlcVXs26kYnqRikRLev4GZGvBIDnGHyWLvMo0gx4r"
    "1kBnAGl5WyfZJu5SFywm/o5fMP70zVUoVdY+zC885uqB5pEV+VsAMyy98NumfmBF7kGtIjbt4l+hAqpzM1ME1z0kRqN7n9JEKn0H"
    "DlsQAD9V582O2bBVSTRzF8aFJ7RKohGhrFqyGmYLA+ZAv5Z4kJ1HBZg1LEz2G88MT2vznP6lCEH0m9da48Q5/85pNQ5dlGGEkhpP"
    "I+xazh6EZ0cFNGdNY1vAQY5rBQ8LPwizNSKenYvqwMdobWy8xReIQWCsOk2l4QCtlu8kC+iabWrsCPOfjdQmYTlpI+emQwssWXYU"
    "UQ+wDGl7QWoaSAnwGu+XQMQ9ndaUycPdBu5+cuw815uiNNtIVfeKejFIbLF66Jq/duKfZ9tgWthPzpoM50u9aie2FZwNOuEtaLIv"
    "WcdtBpY9LvnAemSuDUTLLSuiQYwBwRq8HKdFARJ362Wf8gg+ldoeXEV9R6+cscPaeY5j/+E/rsTWyxixg614qiFQyF5xL/7DWjUI"
    "LBuIrmJUIVcu8LelTUslepTGF6CBTJ0RABv8pK7S8dUDy+HXtvT8bmz8l9zOsaTagC1jUpvaeGpi44gorN9KU7z56FMxoKSeKF7M"
    "B2zuTFBN/GekI9gddLfQRaVK/9LAD4j+uBzdt/XICe2kGqjNu3ds9tnx82/9beezRyTRfFQ+p1V/jfaEt8AMTZ5I2IZQAkgkAnXa"
    "KieRKs8qKbkF5yjhtI0krsyg4+BDQZzog+dgkxqYxLDuNyd6hN9HlnMEBAwW8hxWGp4WE8GvL2IU529JVMaBvK/NcrGNnPW/m158"
    "BwaIFSfbSxQbqjv6KKqMi30i2bowXBAJit2YQHb3VN1reFBynM9mRm71PGLAbizdVS5VIqNI5zGquZrgpmTqqMF/AjsWpVSTo6Gi"
    "WD3O2PJWReehkH3aMJMFeu3jv0/Ru+Ihw1zwFcZr0Yh7rK2NEYQyZjcmf5/vX0rBh89HEVdc78j16arXcFWHNZS2lLAm0NC+Xmak"
    "WIP3YUcba6POPYKwqP2huUccyyGg+qFPTcWF488PNm4H86YQRrnUSkPEofjMuaLMrp3vS1TgpjuUZRiZ"
)


VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST: Final[Mapping[str, JsonValue]] = {
    **ANTHROPIC_MESSAGES_REQUEST,
    "anthropic_version": "vertex-2023-10-16",
}


VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "id": "msg_vrtx_011Cg38nogK8THpR3HzMQ1Mt",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "thinking",
            "thinking": (
                "The user asked to read the validation file first, but since reading the tests is independent, I "
                "think it's fine to read both files together rather than strictly sequencing them.\n\n"
            ),
            "signature": VERTEX_AI_ANTHROPIC_THINKING_SIGNATURE_0,
        },
        {
            "type": "thinking",
            "thinking": (
                "I'll check `shop/validation.py` for how `validate_quantity` handles zero, negative, and non-integer "
                "values, and review `tests/test_validation.py` for the expected behavior.\n\n"
            ),
            "signature": VERTEX_AI_ANTHROPIC_THINKING_SIGNATURE_1,
        },
        {
            "type": "tool_use",
            "id": "toolu_vrtx_01SYkCZM7zmQWqcyokJjiAyf",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
        },
        {
            "type": "tool_use",
            "id": "toolu_vrtx_01UhxynEUZYnaNhRkRGC2ZGn",
            "name": "read_file",
            "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 50},
        },
    ],
    "container": None,
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "stop_details": None,
    "usage": {
        "input_tokens": 4,
        "cache_creation_input_tokens": 876,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 425,
        "output_tokens_details": {"thinking_tokens": 229},
    },
}


VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST: Final[Mapping[str, JsonValue]] = {
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "The checkout rejects quantities of zero. Find the validation code and its tests."}],
        },
        {
            "role": "model",
            "parts": [
                {"text": "I will locate the quantity validation and its tests."},
                {
                    "function_call": {
                        "name": "search_files",
                        "args": {"query": "quantity", "paths": ["shop/", "tests/"]},
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
                    "text": "Read the validation file first so we can understand how zero and negative quantities are handled.",
                },
            ],
        },
    ],
    "system_instruction": {
        "parts": [
            {
                "text": (
                    "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                    "Explain which file you need next, then call a read-only tool. Never change files. Use the "
                    "chat_completions client for this project review. Consider zero, negative and non-integer quantities "
                    "before choosing the next file to inspect."
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
                            "query": {"type": "string"},
                            "paths": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["query", "paths"],
                    },
                },
                {
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
                    },
                },
            ],
        },
    ],
    "generationConfig": {
        "max_output_tokens": 1024,
        "thinkingConfig": {"thinkingLevel": "high", "includeThoughts": True},
    },
}


VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": (
                            "**My Thought Process on Analyzing the Request and Planning the Next Action**\n\nRight, so my primary "
                            "objective here is to fulfill the user's request while adhering strictly to the system's operational "
                            'guidelines. The user has specifically asked me to "Read the validation file first so we can '
                            'understand how zero and negative quantities are handled." This is a direct instruction that aligns '
                            'perfectly with the overarching system instruction to "Read the relevant files before suggesting a '
                            'fix."\n\nMy internal checklist, based on the system\'s directives, also reminds me to:\n*   "Explain '
                            'which file you need next, then call a read-only tool."\n*   "Never change files." (This is a '
                            'foundational rule, of course.)\n*   "Use the chat_completions client for this project review." (This '
                            "context is already established.)\n*   Crucially, and directly relevant to the user's query: "
                            '"Consider zero, negative and non-integer quantities before choosing the next file to '
                            'inspect."\n\nNow, looking at the files I have access to, I see `shop/validation.py` and '
                            "`tests/test_validation.py`. The user's request directly points to `shop/validation.py` as the next "
                            "file to examine for validation logic. This makes sense, as the name itself implies it's where the "
                            "core validation rules reside.\n\nBefore I jump straight into invoking a tool, the system requires me "
                            "to articulate *why* I'm selecting this file and *what* I intend to do with it. So, I need to "
                            "explain that I'm going to inspect `shop/validation.py` specifically to understand its handling of "
                            "quantities, particularly concerning zero, negative, and non-integer values, as prompted by the user "
                            "and reinforced by the system's guidance.\n\nFollowing that explanation, I'll then execute the action "
                            "by calling a read-only tool, `read_file`. I'll specify `shop/validation.py` as the path, and to get "
                            "a good initial overview, I'll plan to read a reasonable chunk, say from line 1 to around line 50, "
                            "to capture the initial validation logic. This seems like a good starting point to gather the "
                            "necessary context for the user's specific inquiry.\n\n"
                        ),
                        "thought": True,
                    },
                    {
                        "text": (
                            "To see how quantities (including zero, negative, and non-integer values) are validated, I need to "
                            "read `shop/validation.py` next."
                        ),
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {"start_line": 1, "end_line": 50, "path": "shop/validation.py"},
                            "id": "call_267302",
                        },
                        "thoughtSignature": VERTEX_AI_GEMINI_THOUGHT_SIGNATURE_0,
                    },
                ],
            },
            "finishReason": "STOP",
        },
    ],
    "usageMetadata": {
        "promptTokenCount": 246,
        "candidatesTokenCount": 65,
        "totalTokenCount": 642,
        "trafficType": "ON_DEMAND",
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 246}],
        "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 65}],
        "thoughtsTokenCount": 331,
    },
    "createTime": "2026-10-15T01:13:21.457273Z",
    "responseId": "sSjQarn0G-Wb4_UPmobfOA",
}
