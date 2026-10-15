import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Final
from unittest.mock import ANY

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.process import graceful_stop_seconds
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.case import SSEFrame, TranslationTestCase
from integration.translation.chat_completions.bases.gemini import GEMINI_3_8_FLASH_TEST_CASE
from integration.translation.chat_completions.bases.vertex_ai import (
    GEMINI_3_8_FLASH_TEST_CASE as VERTEX_GEMINI_3_8_FLASH_TEST_CASE,
)
from integration.translation.gemini_thoughts import (
    LONG_SUMMARY,
    gemini_thoughts_proxy,
    sent_thinking_config,
    thoughts_off,
    with_thinking_config,
)
from integration.translation.messages.bases.gemini import (
    GEMINI_3_8_FLASH_TEST_CASE as MESSAGES_GEMINI_3_8_FLASH_TEST_CASE,
)
from integration.translation.responses.bases.gemini import (
    GEMINI_3_8_FLASH_TEST_CASE as RESPONSES_GEMINI_3_8_FLASH_TEST_CASE,
)
from integration.translation.runner import (
    assert_provider_request,
    assert_translation,
    encode_sse_frame,
    parse_sse_frames,
)
from pydantic import JsonValue

__all__ = ("thoughts_off",)

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

GEMINI_3_8_FLASH_THINKING_SHOWN_THOUGHT: Final = "**My Inspection Strategy: Validation First**\n\nOkay, here's how I'm approaching this project review. The instruction specifically mentions considering zero, negative, and non-integer quantities. To efficiently address that, I'm going to start with `shop/validation.py`. My reasoning is simple: the validation logic is the first line of defense. Understanding how the system *expects* to handle these edge cases will inform everything else. It's the logical place to begin my investigation to understand the code. I'll use the `read_file` tool to examine that file. I am **not** changing any files. I will call the `chat_completions` client for this project review. I am now proceeding with reading `shop/validation.py`.\n\n\n"

GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE: Final = (
    "EowMCokMAWkUfRPuOEs7DW4IXhoH/SNh9qYezwNSqhXT2tKoI1mcXDG+yx7R+12sMmzvjHh2OSghAlsvmR6welOzIM1LMw6c"
    "QrSX2IPRo/xbtUvDLlo6aqufzRtini5EF0yxQdmyn9r9xznzDyuEf4eqrbK0KQAArEVM13rm3CTDIjyDRQPawLALfUh/e61/"
    "XVcWrr1PZLR5Jj46ImyrobGGLmJuoytm3ezTEihTb2b0q3/vHOW9cO/gJzMcbCI8koexQLuBB1KLSHUCaZZQYPj4jLx/g0de"
    "2l3H/DYR7Cdroh0P/R3mewkhu+3D2ELnHKt85InrdQOqdXp/XhICsivPgsIEWHMRr6kQ3ZLIUJvJbJ7zo7/4bYnHOm+JqmoX"
    "E+VMobI2YZ+ff9GOYg4vsJRTeF7ff9kZHYBWRM9XvTsZMbeCZHRbCOLNmSSJPfn2ToDOgMxU1u/GqHYmVSV97ybT3Y0mxwtf"
    "2WcVlPUCkFX0GW3GP5IHVkEISo0TznJSI3/yYb+5UmQFMuuIedDdhKa74eU1/K59VuTnd3DzVX/lfhpFbGLDkOSurRsDsViN"
    "69zq+6VbLi+5ARi6SJqUU+1U02QaSIQVp8kxtrhEIsPedjCIYHQQNxR8hFPMhnEssR51THNELWH6sVccxpJIbFfGjl0TZS9d"
    "rfYL2c+IBX56lrH8IX2gz1iMOsESKwTyAECmqJIL7d/HcVdTk404Hd9SYycB+2MP2GN2XA7ivoO/zfkkFcf+PGmXPed8dHlW"
    "bAxrQcOAz89p8kPzb1bou5QUT3feeRG06aJL31gX7BSYj19mvLeo0AmMzx+Wb91NFlwUvlJYlhJo4NfZWg/ymcGclgM8r7iE"
    "MmWZa8wul8g4hbJOg3GE9NV70zvKqEpQrJLTa9jHca0DBvj8N3C+XvpeEg+Dys0cxE7Ak0VVYbvu9r0OHKSWtC7NDYrJsF2n"
    "VFmdavSxpyPEqPaAdWdiewDH0nhP5GCR/ciikokWFQMF5iClWRfL72qEbFx3cQkctqEf+kK2CevIWKPf8bvkrKVDdPqbjk0G"
    "9NWuud+YgNxl4dSCkAWoMrAFlxXozEqMmanMAtZk/7nPeOVFbYEL/xOpSCfxaEKS1zEmMwTb/UNVAogsNY9YdBSFYVNSrS8F"
    "N/IrHBvsP66+YEFuR46Di6cWuuZQdSOfB2uQe1eSC8IgrQ6XUhQHWeXKUXfbcb/1abPOi5PUMYyLBQ7r52UKCEfOBc1j0oNL"
    "wYzI/vWAz74gq8NblQhHB2VlrInJ9rRVY1MehInqUH0kcG6ysQDQLiLJiS30TV0bZh05I86RfcjaN1twbRir82YchRjXG8/1"
    "6iFRJ54Sk7oVzbdnbFOugDea8qIblFmgUTq0+SRLGBFcDMvAqYLNcR2ZO7o0xinMG7IuNRwoSMahnmreWYr9saVJAScd3All"
    "i6RQuxjGIdxG0s0UNuoWiSNIf1Z2ngoJrvZHTc0iZK+A2zv3CBqWAaS8WGwSfcItnlc6hMB0bTWasoCelmRjfv+1jQnKzKMj"
    "DzZlO6skMkv5x9GzLrAvpYoulLYa/wp9+exw/Ll7MxSB22aeAGfai9C7Vj310y8nApsG3QxaoUUO4jg1eCjrrluBLgoVY8cE"
    "/O95IAUa2rFzVrOWbHuob7TEI/8+BwieGeyeHdPGn4fNeKYMkRjFeVJ1GA/2+iCHCjCbSNm4sdzh9188/Es3q4acRfQUiqwR"
    "iQzW2E1fy9mXRDY9y5fdXuzjHtuCkH8MqR0cioN1p4OGZzPi6xXFaqV576NLjySBpfpQEYeXRA3j5NVwTwbeYKnGOot4bEtJ"
    "JQfelwkXLXeAfxzCTD0/vGtdhilzEK8wSYM9a+8llr5D4XwtEsu/68YbXJ2jwzsS3rE/Frulrf6paz9fS45ZXj98+OT5EpFO"
    "o5XHfTXKb5Brzr0tKfPSB2Mx6fXbYP3lZRwfJIaUq8Alwdo7g2jnHQUhCNBB8hkb/IkQrnhsAGS6f0ZEjwf6ErVOa8DkfizN"
    "y4WTdS1D3gOUJxOcszm8J5b085tsfFxEJVgA0VjI60DBY0owWn2I"
)

GEMINI_3_8_FLASH_THOUGHTS_OFF_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": "To understand how zero, negative, and non-integer quantities (such as floats or invalid types) are currently handled and where the checkout rejection is enforced, we need to inspect `shop/validation.py` next.",
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "path": "shop/validation.py",
                                "end_line": 50,
                                "start_line": 1,
                            },
                            "id": "call_330625",
                        },
                        "thoughtSignature": GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE,
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
        "candidatesTokenCount": 77,
        "totalTokenCount": 793,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 358,
            },
        ],
        "thoughtsTokenCount": 358,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.8-flash",
    "responseId": "xy3RapaNEbT4-sAPhMrt6AU",
}

GEMINI_3_8_FLASH_THINKING_OMITTED_THOUGHT_SIGNATURE: Final = (
    "EscECsQEAWkUfROlWYT6hX0DWOKSCg4MSuLzOV7u43EgJDn0dfS2R8v/MVGGcNolGA9rteb/cV3dyCSQ7D2EO2CB5whDkblj"
    "fGAAw6NBhqnidYNQiXbsg7oqynxS116ZQnk302S4KzvAbIvo4gpRbahPVClXKkP8PCQWA5eRQ/e9oQeHr97gbBB7+PLsl0NK"
    "m7YPV6lP3fAXx/r3EADbdNL1diHylKd6tCWUKRea/x2QIOBQZdY5jHSU73z/pnPOotVwggEiasDLSLA+uUR1KllgGc8WKZ4i"
    "uRM3L1gDr1QZ4Rht3quiF8AwfopWE5m2ON+FD78aHUDG/lgDs0vXnJluzri9LpPaexfbpvwMPtAx8Bxrq1/dCeekR2VnVx1c"
    "qZ7TSoWMqk4x98mPpyUFz1hSBRnXaOGvQl5Wk7Iart49tH2LoIBYoV+PkCgNZ2MOi2uzc7OlN/8gvOeBM05xbGAplsLiuKd4"
    "jkgv7Nz8Jby3v/hxtQT9+Ow3mxwSaiIfEkuaxkPiK1cX1nwT92gFzpmhVsrGqpiCUdAfos39Wd2/rcrFbtl5iMAVcHG5JGjw"
    "8AjubFqiWDMqWPC58FfsX9TLC2BMUbPf2xi3hrw+1QWDtSuNl9s+dofkCogXFdLA9lAFc0oRj3MVhzYUhn5eixApJdggDsJF"
    "iJaQtRihs4dstbXX9QXYFnx22Ql3x9TWJibiH8jz4fJVOf0wreMWZ2xVXBiw2xKdy5dB7xRDnrQD/sdcpTpCWURYXVCXglJ9"
    "DD7e7/OCNMLCPw=="
)

GEMINI_3_8_FLASH_THINKING_OMITTED_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": "To understand how the checkout validates quantities\u2014including zero, negative, and non-integer quantities\u2014I need to inspect `shop/validation.py` next.",
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "start_line": 1,
                                "path": "shop/validation.py",
                                "end_line": 40,
                            },
                            "id": "call_745953",
                        },
                        "thoughtSignature": GEMINI_3_8_FLASH_THINKING_OMITTED_THOUGHT_SIGNATURE,
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
        "candidatesTokenCount": 65,
        "totalTokenCount": 531,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 358,
            },
        ],
        "thoughtsTokenCount": 108,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.8-flash",
    "responseId": "zS3RasS-Lp2-jMcPoay98As",
}

GEMINI_3_8_FLASH_THINKING_SHOWN_THOUGHT_SIGNATURE: Final = (
    "EqUECqIEAWkUfRNZEW6qD3+V/1LAc1riV0fvqkg4Bldd7YYDfln+ZHNg6Lbw2T31UZHTum0qYvrAOor848+zSkFG7qBDfdZ9"
    "69FGl/dQGq2llr1yrv23blnbwC7tO0uevr8225EZ/4Pv+hbfP0teFcSZsasE4LQangtxYkewghc6cH1Acz293SIz1cqF/3CM"
    "Ta+FtJ5x106oauWy79+z1VbV9TNyHpNtsrUy3ADR2PFNBjXKf2TvUWygzR4w2RJg581+dfx0sI97O3UlZrhSKRe+p/6siUP7"
    "qnysURAia7k+ew5f05Nuo1aDwD2G20G2s9v7L0PxnCCAay5L4YTub5LXxvUGBXMhSHVb9MmO8+2rNPqDtPWh0bOQdKbkWAV/"
    "dMckMLTKANDbnzBjtCK1HbN7Rw0sIBSnBVy59z7h+0ierHFgA8Vntfp/nvAy0wDpuZu09agdB3T6FO2HCvWCFG2Y+0B4lGJV"
    "ErOknvn50gtn9Tpn1QSGv4VDRrrcv6nCOyohoL9CsBeO1Bs+RItyZSAoTA3bVB5IjcF2kKK27+/+VRJjN0cO4Cm5RD/ETb4+"
    "kI101AFWJ3CYwKEu91SAVTtXLewQYd5LBiJ3Btt68m8d5uUp0a1kAn2cH6g2C7TOfoJ7BgU5+vU81mS8hk9QOFGb9RYUXs8Y"
    "cJl21l6WfMMZri5bMI9qCAFn+YZdBBeIbGq+JJcIgjr/R2tULmU584+5beoEydkS"
)

GEMINI_3_8_FLASH_THINKING_SHOWN_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": GEMINI_3_8_FLASH_THINKING_SHOWN_THOUGHT,
                        "thought": True,
                    },
                    {
                        "text": "I need to inspect `shop/validation.py` next to examine how zero, negative, and non-integer quantities are currently handled in `validate_quantity`.",
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "end_line": 40,
                                "start_line": 1,
                                "path": "shop/validation.py",
                            },
                            "id": "call_386884",
                        },
                        "thoughtSignature": GEMINI_3_8_FLASH_THINKING_SHOWN_THOUGHT_SIGNATURE,
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
        "candidatesTokenCount": 66,
        "totalTokenCount": 529,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 358,
            },
        ],
        "thoughtsTokenCount": 105,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.8-flash",
    "responseId": "0C3RavXPIJWX9MoPtorosAs",
}

GEMINI_2_5_FLASH_THOUGHTS_OFF_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": "I will read the `shop/validation.py` file to understand the quantity validation logic.",
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "start_line": 1,
                                "path": "shop/validation.py",
                                "end_line": 20,
                            },
                        },
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
        "promptTokenCount": 334,
        "candidatesTokenCount": 53,
        "totalTokenCount": 387,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 334,
            },
        ],
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-2.5-flash",
    "responseId": "0y3Rar76LO_K-8YPh6Kd0Ak",
}

VERTEX_GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE: Final = (
    "AY89a18W+yVUlI0b2XsXQujlNKztnSpKXMf0+1JFiprxfp7kc0bG2g9WoPITuNkCqkuBaO5SkZaKDrKQTlJFJKj+H6SnETLk"
    "JrR3SAzp2GTjwM2QRsJjI5ky54oylVT93Jh51PWCMp6vl8tZstMJrois8b1i4/eTOyKaVprypLW8DOIImqN3Cbkea8nEDe9r"
    "XsdpdvsR7i+NvD0eiCW+Z4ZIflx5NVENbyEezjyJCL3lg3bV2Mfb0ytsACHXuqgGQEcH9HBKQKGkF5n7mA0GNLw6wAqlk8FE"
    "JKttP6NgS+HzS/U+xaqxci6B50l0hIMw3pTOHu8Gg5fRKxf/dzvks+s709/ODCGBDrhMzpf/ilnPAMrB67cQrHvQOhnI2pGk"
    "cvkZv+ZGwAQxjJHs/r0aXDGU2mwN9RY+/rSEv8y4MWd5c1KIs7xa1GytDBaaG8Zwh3kZzaSOYjVKIk7cy31mPJ01jCNdW9Pg"
    "1p2r2iwbzJrBHo5Ku3s1yK4rADeY1WOU63sM6evVkWJEtu6KPxAqE9JkQKQoAPOFtoP89RtiTSl0nZZaO0stbqG7G+AkE2iO"
    "5WkKCB1o55R7EEZDalDgFPput8khce5hPrhIGikN3yo11sHsAA2PAuDcHo4L3eWaYPExFiZtB+gVTbBwx42ST3UuBB9lrHtk"
    "GCrp7tlOBHJJY+RlXj1G0TJsSxUZDSEswCB+gjG+1tdEchKGRRKOBUjWGARCozBsIEuBOR0F1ugSCLYn7CumS2bLwmMhKY7+"
    "mfKXEgF2U84MkmwDEOBSTubNbRhy0qNhVPcLT8pJvGvtTvLny1C90TBTHErVzOKb3Z095fFpUyW7MmFQj5J9rX3A160W0SNQ"
    "RKXf2Yr6ikyVhLBSFNvQ00n81iXlyg7bp++ChbbCJBFCUcrGF5WnPLk3+isixodP+G0b+pNtCNo8xMkrvqGqL5AMlmGJ9Qm4"
    "a9S6QgDRaw5AzcV/W90a7/BcaIeqkIly3Z9M0+d3Eep4igYj6AK0JzX1BhchMogq9g/dquTTZZk4edVKflBmQcwg4qGGvioI"
    "X/ae4tKJVqB2U0Ed+A6t3JZ4Ggdpopdnuy29XvdEyTmUg+fF2sTCgAdK07hueKYO2pkhqj86mSy88iHV0/secWQs0ZuHtcSW"
    "D9lAG57yvpMf6RIwccLTyVmM4jcO0A2OB0sTnNIVKYYtswPtUqbrPMTR5saBfEfnULipI521kpb3BseODBKar+G1FISLrb0Q"
    "OjMhNnCWCoCSVDk1FbDmtEvSjfiUzeJtFYoqM8PIXyGiBZnsmjGaOw3QMNvLfkoy4gqE6nvFVgwIk3+J6Z1GdFecCPSSCatz"
    "LRwwQ8B5ZdlJ6+xjTDWIuyaOByD5BsB98e0YHyuLSYP17w6+0pSGXf3M6AdBJXK7hcL/vu3sGzftYARFHgRY1PQsYCqW8RVy"
    "uJL/GwEAT4EgzkDDWufizfUHJEAUVBpQKkxZW7eTbyHyPS/Uiu5a74f5L8lYyep1pAdqt3iK4lvmQ3T9h3DAQIQ15T8PLV78"
    "FjzJvZDQ2uMryvoRS1MGEcK6dxc4VvcWalQTjyczE4jQIxEEiN3U5Yo5CaM6FFGDnjIJBiy0fUD16P8e4ZaEaRFQCY4QZ0sA"
    "bhwSHWhnU7QWvA/CVnqOVni5TpRsY1ShtEaQXKTslUuz96lB6XwEeaDkPqsPdw+mWY9QmmLqXnXERnTq13rQY1s="
)

VERTEX_GEMINI_3_8_FLASH_THOUGHTS_OFF_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": "To understand how zero, negative, and non-integer quantities are handled during checkout validation, we need to inspect `shop/validation.py` next.",
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "start_line": 1,
                                "path": "shop/validation.py",
                                "end_line": 50,
                            },
                            "id": "call_1272842",
                        },
                        "thoughtSignature": VERTEX_GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE,
                    },
                ],
            },
            "finishReason": "STOP",
        },
    ],
    "usageMetadata": {
        "promptTokenCount": 246,
        "candidatesTokenCount": 64,
        "totalTokenCount": 583,
        "trafficType": "ON_DEMAND",
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 246,
            },
        ],
        "candidatesTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 64,
            },
        ],
        "thoughtsTokenCount": 273,
    },
    "modelVersion": "gemini-3.8-flash",
    "createTime": "2026-10-15T20:02:49.484984Z",
    "responseId": "aTHRavjMHYvi6tkPt6OroAo",
}

GEMINI_3_8_FLASH_THOUGHTS_OFF_STREAM_THOUGHT_SIGNATURE: Final = (
    "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"
    "v2UCtcYUZfnJ4FuKNXaZ/AFDT7VxbdnOaS8HVA=="
)

GEMINI_3_8_FLASH_THOUGHTS_OFF_STREAM_FRAMES: Final = (
    SSEFrame(
        data={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "To understand how zero, negative, and non-integer quantities are handled during checkout, I need to inspect `shop/validation.py",
                            },
                        ],
                        "role": "model",
                    },
                    "index": 0,
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 358,
                "candidatesTokenCount": 27,
                "totalTokenCount": 673,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 358,
                    },
                ],
                "thoughtsTokenCount": 288,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "yi3RaufbH6bXjMcPt5mF2Q0",
        }
    ),
    SSEFrame(
        data={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "` next.",
                            },
                        ],
                        "role": "model",
                    },
                    "index": 0,
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 358,
                "candidatesTokenCount": 30,
                "totalTokenCount": 676,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 358,
                    },
                ],
                "thoughtsTokenCount": 288,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "yi3RaufbH6bXjMcPt5mF2Q0",
        }
    ),
    SSEFrame(
        data={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "read_file",
                                    "args": {
                                        "end_line": 50,
                                        "path": "shop/validation.py",
                                        "start_line": 1,
                                    },
                                    "id": "call_317677",
                                },
                                "thoughtSignature": GEMINI_3_8_FLASH_THOUGHTS_OFF_STREAM_THOUGHT_SIGNATURE,
                            },
                        ],
                        "role": "model",
                    },
                    "index": 0,
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 358,
                "candidatesTokenCount": 63,
                "totalTokenCount": 709,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 358,
                    },
                ],
                "thoughtsTokenCount": 288,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "yi3RaufbH6bXjMcPt5mF2Q0",
        }
    ),
    SSEFrame(
        data={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "",
                            },
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 358,
                "candidatesTokenCount": 63,
                "totalTokenCount": 709,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 358,
                    },
                ],
                "thoughtsTokenCount": 288,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "yi3RaufbH6bXjMcPt5mF2Q0",
        }
    ),
)

GEMINI_2_5_FLASH_REQUEST: Final = with_thinking_config(
    {
        **GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request,
        "contents": [
            list_value(GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request["contents"])[0],
            {
                "role": "model",
                "parts": [
                    {"text": "I will locate the quantity validation and its tests."},
                    {
                        "function_call": {
                            "name": "search_files",
                            "args": {"query": "quantity", "paths": ["shop/", "tests/"]},
                        }
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
                        },
                    },
                ],
            },
            list_value(GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request["contents"])[3],
        ],
    },
    {"thinkingBudget": 4096, "includeThoughts": False},
)


def gemini_chat_response(
    *,
    content: str,
    tool_call: Mapping[str, JsonValue],
    message_fields: Mapping[str, JsonValue],
    usage: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    return {
        **GEMINI_3_8_FLASH_TEST_CASE.expected_litellm_response,
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": content,
                    "role": "assistant",
                    "tool_calls": [dict(tool_call)],
                    "images": [],
                    **message_fields,
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            },
        ],
        "usage": dict(usage),
    }


def signed_tool_call(call_id: str, arguments: str, signature: str) -> Mapping[str, JsonValue]:
    return {
        "index": 0,
        "provider_specific_fields": {"thought_signature": signature},
        "function": {"arguments": arguments, "name": "read_file"},
        "id": f"{call_id}__thought__{signature}",
        "type": "function",
    }


THINKING_REQUEST: Final[Mapping[str, JsonValue]] = {
    key: value for key, value in GEMINI_3_8_FLASH_TEST_CASE.litellm_request.items() if key != "reasoning_effort"
}

GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="thoughts_off",
    expected_provider_request=with_thinking_config(
        GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request, {"thinkingLevel": "high", "includeThoughts": False}
    ),
    mock_provider_response=GEMINI_3_8_FLASH_THOUGHTS_OFF_RESPONSE,
    expected_litellm_response=gemini_chat_response(
        content=(
            "To understand how zero, negative, and non-integer quantities (such as floats or invalid types) are "
            "currently handled and where the checkout rejection is enforced, we need to inspect "
            "`shop/validation.py` next."
        ),
        tool_call=signed_tool_call(
            "call_330625",
            '{"path": "shop/validation.py", "end_line": 50, "start_line": 1}',
            GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE,
        ),
        message_fields={
            "thinking_blocks": [],
            "provider_specific_fields": {"thought_signatures": [GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE]},
        },
        usage={
            "completion_tokens": 435,
            "prompt_tokens": 358,
            "total_tokens": 793,
            "completion_tokens_details": {"reasoning_tokens": 358, "text_tokens": 77},
            "prompt_tokens_details": {"text_tokens": 358},
        },
    ),
)

GEMINI_2_5_FLASH_THOUGHTS_OFF_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="thoughts_off",
    litellm_request={**GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "model": "gemini/gemini-2.5-flash"},
    expected_provider_endpoint="/models/gemini-2.5-flash:generateContent",
    expected_provider_request=GEMINI_2_5_FLASH_REQUEST,
    mock_provider_response=GEMINI_2_5_FLASH_THOUGHTS_OFF_RESPONSE,
    expected_litellm_response={
        **gemini_chat_response(
            content="I will read the `shop/validation.py` file to understand the quantity validation logic.",
            tool_call={
                "index": 0,
                "function": {
                    "arguments": '{"start_line": 1, "path": "shop/validation.py", "end_line": 20}',
                    "name": "read_file",
                },
                "id": ANY,
                "type": "function",
            },
            message_fields={"thinking_blocks": []},
            usage={
                "completion_tokens": 53,
                "prompt_tokens": 334,
                "total_tokens": 387,
                "completion_tokens_details": {"text_tokens": 53},
                "prompt_tokens_details": {"text_tokens": 334},
            },
        ),
        "model": "gemini/gemini-2.5-flash",
    },
)

VERTEX_GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE: Final = replace(
    VERTEX_GEMINI_3_8_FLASH_TEST_CASE,
    scenario="thoughts_off",
    expected_provider_request=with_thinking_config(
        VERTEX_GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request, {"thinkingLevel": "high", "includeThoughts": False}
    ),
    mock_provider_response=VERTEX_GEMINI_3_8_FLASH_THOUGHTS_OFF_RESPONSE,
    expected_litellm_response={
        **gemini_chat_response(
            content=(
                "To understand how zero, negative, and non-integer quantities are handled during checkout validation, "
                "we need to inspect `shop/validation.py` next."
            ),
            tool_call=signed_tool_call(
                "call_1272842",
                '{"start_line": 1, "path": "shop/validation.py", "end_line": 50}',
                VERTEX_GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE,
            ),
            message_fields={
                "thinking_blocks": [],
                "provider_specific_fields": {
                    "thought_signatures": [VERTEX_GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE]
                },
            },
            usage={
                "completion_tokens": 337,
                "prompt_tokens": 246,
                "total_tokens": 583,
                "completion_tokens_details": {"reasoning_tokens": 273, "text_tokens": 64},
                "prompt_tokens_details": {"text_tokens": 246},
            },
        ),
        "model": "vertex_ai/gemini-3.8-flash",
    },
)

GEMINI_3_8_FLASH_THINKING_OMITTED_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="thinking_omitted",
    litellm_request={
        **THINKING_REQUEST,
        "thinking": {"type": "enabled", "budget_tokens": 1024, "display": "omitted"},
    },
    expected_provider_request=with_thinking_config(
        GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request, {"includeThoughts": False}
    ),
    mock_provider_response=GEMINI_3_8_FLASH_THINKING_OMITTED_RESPONSE,
    expected_litellm_response=gemini_chat_response(
        content=(
            "To understand how the checkout validates quantities—including zero, negative, and non-integer "
            "quantities—I need to inspect `shop/validation.py` next."
        ),
        tool_call=signed_tool_call(
            "call_745953",
            '{"start_line": 1, "path": "shop/validation.py", "end_line": 40}',
            GEMINI_3_8_FLASH_THINKING_OMITTED_THOUGHT_SIGNATURE,
        ),
        message_fields={
            "thinking_blocks": [],
            "provider_specific_fields": {"thought_signatures": [GEMINI_3_8_FLASH_THINKING_OMITTED_THOUGHT_SIGNATURE]},
        },
        usage={
            "completion_tokens": 173,
            "prompt_tokens": 358,
            "total_tokens": 531,
            "completion_tokens_details": {"reasoning_tokens": 108, "text_tokens": 65},
            "prompt_tokens_details": {"text_tokens": 358},
        },
    ),
)

GEMINI_3_8_FLASH_THINKING_SHOWN_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="thinking_shown",
    litellm_request={
        **THINKING_REQUEST,
        "thinking": {"type": "enabled", "budget_tokens": 1024, "display": "summarized"},
    },
    expected_provider_request=with_thinking_config(
        GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request, {"includeThoughts": True}
    ),
    mock_provider_response=GEMINI_3_8_FLASH_THINKING_SHOWN_RESPONSE,
    expected_litellm_response=gemini_chat_response(
        content=(
            "I need to inspect `shop/validation.py` next to examine how zero, negative, and non-integer quantities "
            "are currently handled in `validate_quantity`."
        ),
        tool_call=signed_tool_call(
            "call_386884",
            '{"end_line": 40, "start_line": 1, "path": "shop/validation.py"}',
            GEMINI_3_8_FLASH_THINKING_SHOWN_THOUGHT_SIGNATURE,
        ),
        message_fields={
            "reasoning_content": GEMINI_3_8_FLASH_THINKING_SHOWN_THOUGHT,
            "thinking_blocks": [{"type": "thinking", "thinking": GEMINI_3_8_FLASH_THINKING_SHOWN_THOUGHT}],
            "provider_specific_fields": {"thought_signatures": [GEMINI_3_8_FLASH_THINKING_SHOWN_THOUGHT_SIGNATURE]},
        },
        usage={
            "completion_tokens": 171,
            "prompt_tokens": 358,
            "total_tokens": 529,
            "completion_tokens_details": {"reasoning_tokens": 105, "text_tokens": 66},
            "prompt_tokens_details": {"text_tokens": 358},
        },
    ),
)


def with_request(case: TranslationTestCase, scenario: str, **fields: JsonValue) -> TranslationTestCase:
    return replace(case, scenario=scenario, litellm_request={**case.litellm_request, **fields})


def with_thinking(case: TranslationTestCase, scenario: str, display: JsonValue) -> TranslationTestCase:
    return with_request(case, scenario, thinking={"type": "enabled", "budget_tokens": 1024, "display": display})


def with_summary(case: TranslationTestCase, scenario: str, summary: JsonValue) -> TranslationTestCase:
    return with_request(case, scenario, reasoning_effort={"effort": "high", "summary": summary})


@pytest.mark.parametrize(
    "case",
    (
        GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE,
        GEMINI_2_5_FLASH_THOUGHTS_OFF_TEST_CASE,
        VERTEX_GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE,
    ),
    ids=(
        GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE.id,
        GEMINI_2_5_FLASH_THOUGHTS_OFF_TEST_CASE.id,
        VERTEX_GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE.id,
    ),
)
def test_gemini_keeps_thinking_and_billing_reasoning_without_thought_text_when_the_gateway_turns_it_off(
    case: TranslationTestCase, thoughts_off: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, thoughts_off, provider)


SUMMARY_ASKS: Final = (
    with_summary(GEMINI_3_8_FLASH_TEST_CASE, "summary_auto", "auto"),
    with_summary(GEMINI_3_8_FLASH_TEST_CASE, "summary_empty", ""),
    with_summary(GEMINI_3_8_FLASH_TEST_CASE, "summary_long", LONG_SUMMARY),
    with_summary(GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE, "summary_int", 7),
    with_summary(GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE, "summary_list", ["auto"]),
    with_summary(GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE, "summary_null", None),
)


@pytest.mark.parametrize("case", SUMMARY_ASKS, ids=tuple(case.id for case in SUMMARY_ASKS))
def test_only_a_string_reasoning_summary_brings_thought_text_back_over_the_gateway_default(
    case: TranslationTestCase, thoughts_off: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, thoughts_off, provider)


DISPLAY_ASKS: Final = (
    GEMINI_3_8_FLASH_THINKING_SHOWN_TEST_CASE,
    with_thinking(GEMINI_3_8_FLASH_THINKING_SHOWN_TEST_CASE, "display_updates", "updates"),
    with_thinking(GEMINI_3_8_FLASH_THINKING_SHOWN_TEST_CASE, "display_int", 7),
    with_thinking(GEMINI_3_8_FLASH_THINKING_SHOWN_TEST_CASE, "display_empty", ""),
    with_thinking(GEMINI_3_8_FLASH_THINKING_SHOWN_TEST_CASE, "display_list", ["omitted"]),
    with_thinking(GEMINI_3_8_FLASH_THINKING_SHOWN_TEST_CASE, "display_long", LONG_SUMMARY),
    GEMINI_3_8_FLASH_THINKING_OMITTED_TEST_CASE,
)


@pytest.mark.parametrize("case", DISPLAY_ASKS, ids=tuple(case.id for case in DISPLAY_ASKS))
def test_any_thinking_display_but_omitted_brings_thought_text_back_over_the_gateway_default(
    case: TranslationTestCase, thoughts_off: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, thoughts_off, provider)


DEFAULT_GATEWAY_DISPLAYS: Final = (
    with_thinking(GEMINI_3_8_FLASH_THINKING_SHOWN_TEST_CASE, "display_omitted_default", "omitted"),
    GEMINI_3_8_FLASH_THINKING_SHOWN_TEST_CASE,
)


@pytest.mark.parametrize("case", DEFAULT_GATEWAY_DISPLAYS, ids=tuple(case.id for case in DEFAULT_GATEWAY_DISPLAYS))
def test_thinking_display_never_turns_thought_text_off_on_the_default_gateway(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)


def test_a_streamed_turn_keeps_thinking_and_billing_reasoning_without_thought_text(
    thoughts_off: Gateway, provider: SharedProvider
) -> None:
    provider.expect(
        Reply(
            content_type="text/event-stream",
            chunks=tuple(encode_sse_frame(frame) for frame in GEMINI_3_8_FLASH_THOUGHTS_OFF_STREAM_FRAMES),
        )
    )
    response: Final = thoughts_off.request(
        "POST",
        "/v1/chat/completions",
        {**GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "stream": True, "stream_options": {"include_usage": True}},
    )
    assert_provider_request(
        provider.received(),
        "/models/gemini-3.8-flash:streamGenerateContent?alt=sse",
        GEMINI_3_8_FLASH_TEST_CASE.expected_provider_headers,
        GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE.expected_provider_request,
    )
    assert response.status_code == 200, response.text
    frames: Final = tuple(frame.data for frame in parse_sse_frames(response.text) if frame.data != "[DONE]")
    deltas: Final = tuple(
        object_value(object_value(choice)["delta"])
        for frame in frames
        for choice in list_value(frame.get("choices", []))
    )
    assert "".join(str(delta.get("content") or "") for delta in deltas) == (
        "To understand how zero, negative, and non-integer quantities are handled during checkout, I need to "
        "inspect `shop/validation.py` next."
    )
    assert [delta for delta in deltas if delta.get("reasoning_content") or delta.get("thinking_blocks")] == []
    usage: Final = [object_value(frame["usage"]) for frame in frames if frame.get("usage")][-1]
    assert {field: value for field, value in usage.items() if field != "cost"} == {
        "completion_tokens": 351,
        "prompt_tokens": 358,
        "total_tokens": 709,
        "completion_tokens_details": {"reasoning_tokens": 288, "text_tokens": 63},
        "prompt_tokens_details": {"text_tokens": 358},
    }
    assert isinstance(usage["cost"], float) and usage["cost"] > 0


@pytest.mark.parametrize(
    ("configured", "case"),
    (("false", GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE), (None, GEMINI_3_8_FLASH_TEST_CASE)),
    ids=("string-false", "null"),
)
def test_a_flag_written_as_text_or_left_null_still_sends_gemini_a_boolean(
    configured: JsonValue, case: TranslationTestCase, gateway: Gateway, provider: SharedProvider, tmp_path: Path
) -> None:
    with gemini_thoughts_proxy(gateway, tmp_path, configured) as owned:
        assert_translation(case, owned.gateway, provider)


BURST_ASKS: Final[tuple[tuple[str, Mapping[str, JsonValue], Mapping[str, JsonValue]], ...]] = (
    (
        "/v1/chat/completions",
        {**GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "reasoning_effort": {"effort": "low", "summary": "auto"}},
        {"thinkingLevel": "low", "includeThoughts": True},
    ),
    (
        "/v1/chat/completions",
        GEMINI_3_8_FLASH_TEST_CASE.litellm_request,
        {"thinkingLevel": "high", "includeThoughts": False},
    ),
    (
        "/v1/messages",
        MESSAGES_GEMINI_3_8_FLASH_TEST_CASE.litellm_request,
        {"thinkingLevel": "high", "includeThoughts": True},
    ),
    (
        "/v1/messages",
        {**MESSAGES_GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "thinking": {"type": "adaptive"}},
        {"thinkingLevel": "high", "includeThoughts": False},
    ),
    (
        "/v1/responses",
        {**RESPONSES_GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "reasoning": {"effort": "high", "summary": "auto"}},
        {"thinkingLevel": "high", "includeThoughts": True},
    ),
    (
        "/v1/responses",
        RESPONSES_GEMINI_3_8_FLASH_TEST_CASE.litellm_request,
        {"thinkingLevel": "high", "includeThoughts": False},
    ),
)


def _assert_each_request_follows_its_own_ask(gateway: Gateway, provider: SharedProvider) -> None:
    burst: Final = BURST_ASKS * 4
    for _ in burst:
        provider.expect(Reply(body=json.dumps(GEMINI_3_8_FLASH_THOUGHTS_OFF_RESPONSE).encode()))
    with ThreadPoolExecutor(max_workers=12) as pool:
        responses: Final = tuple(pool.map(lambda ask: gateway.request("POST", ask[0], ask[1]), burst))
    assert [response.status_code for response in responses] == [200] * len(burst), [
        response.text for response in responses if response.status_code != 200
    ]
    sent: Final = sorted(json.dumps(sent_thinking_config(request), sort_keys=True) for request in provider.received())
    assert sent == sorted(json.dumps(expected, sort_keys=True) for _, _, expected in burst)


def test_every_worker_honors_the_flag_on_every_endpoint_under_a_burst_and_after_a_worker_respawns(
    gateway: Gateway, provider: SharedProvider, tmp_path: Path
) -> None:
    with gemini_thoughts_proxy(gateway, tmp_path, False, workers=2) as owned:
        _assert_each_request_follows_its_own_ask(owned.gateway, provider)
        owned.kill_worker(min(owned.worker_pids()))
        _assert_each_request_follows_its_own_ask(owned.gateway, provider)
