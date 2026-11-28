from collections.abc import Mapping
from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase
from integration.translation.responses.templates import RESPONSES_REQUEST, RESPONSES_RESPONSE
from integration.translation.templates.gemini import GEMINI_GENERATE_CONTENT_REQUEST, GEMINI_GENERATE_CONTENT_RESPONSE
from pydantic import JsonValue

GEMINI_RESPONSES_RESPONSE: Final[Mapping[str, JsonValue]] = {
    key: value for key, value in RESPONSES_RESPONSE.items() if key != "provider_specific_fields"
}

GEMINI_3_8_FLASH_THOUGHT: Final = (
    "**My Approach to Analyzing Quantity Validation**\n\n"
    "Okay, the user wants me to understand how zero and negative quantities are handled, and they’ve "
    "given me a specific constraint: explain my next move and then call a read-only tool. I need to b"
    "e careful to avoid making any changes. The overall goal is to review the project, so it's a good"
    " approach to use the responses client to keep track of the process. I also need to be mindful of"
    " zero, negative, and non-integer quantities in my analysis.\n\n"
    'The user specifically mentions "reading the validation file first." Let\'s start with `shop/valid'
    "ation.py`. I need to figure out how they are checked in the `validate_quantity` function, so it "
    "makes sense to examine that file. But before diving too deep into all lines, I should try to min"
    "imize how much data I read. Given that the tool's performance might become degraded when reading"
    " large files, I should consider reading only a portion of the file. To get an overview, I will r"
    "ead the first 50 lines to see if that's enough information.\n\n"
    "Following the instructions, I need to clearly state my next step before executing it. Therefore,"
    " I'll explain that I need to examine `shop/validation.py` to understand how the system handles z"
    "ero, negative, and non-integer quantities, focusing specifically on the `validate_quantity` func"
    'tion, then I\'ll use `read_file(path="shop/validation.py", start_line=1, end_line=50)` to read th'
    "e start of the file.\n\n\n"
)

GEMINI_3_8_FLASH_THOUGHT_SIGNATURE: Final = (
    "Er8KCrwKAWkUfROyraCcwrIYpIws/Jh5cMTrVJ2ShZLiHvXJuIq/KdxErXHBEeWq4H9FXkFOU5eZFSvo6lnkwhw09W+nukH2"
    "Jgcjan4KoNsFIGvNKhF/mT66xCr40t/LHRSIxu4o585L29auyheBJmh3cR4NjnZPYwLKzmnI3W2ePhcD5tjLU/pt6LOX4IBN"
    "jsDho8ep9wW2foJDkEorvAYRI8RiXJ2loNEjagJAaSo2Rt3URPz3KzvwKo8QlJAYGiQ+5eUkVnwcCsNs0NNY5+RVggqnJflK"
    "okqjOsaIXAxo2jupI4amtI+NpQ59K5TddnJ8eh+fiQSphB2Ex2AlhczWhoXbYqdkjRfYtXF1KiTO4w4L74ApYd7iMSm1Ix5U"
    "76yYiB5+sRaZYzuBIdD01O+SMqPXil9Ft73kzuH6CROVi5Xf3RMRAQvmuJcRYciprH0kcba7E2Z6c0v7Ht6kegzoWrpAtGgW"
    "hL2KTMewyWQF7IZmYLGoo20hniIliBs+7bLAPJ3DEsp1HU55dAsdvV1IwQZtFuqKffE7MB0uQws/snCf6w9mpCbVOYRVuzZe"
    "LTbWzar84bmsp1dfHGS2gOkvsi5fi8IUs/be9aK/budI6nkSXborMoESSG3Z5BTXgi5U0vJcS2FTO6oFMpE1lPBL/vIpPDKR"
    "ujjqtUT+QA89SJmLZDORPc3aLd36dEErhlSZ+9LYsU3EkKNruomCdiSrbClIRK50YYu/ocG9lXF4yhUp3dgKbMWCw9jGZv+a"
    "KPz8sXVgbXSm+zLuvtByXHSfiVmu7hP1Oc7BmdL+lJW03rgMAxtDb39rbP8l+5sm4EDJil9mcGGOYpcBSKRmDacoqk9D8AWG"
    "PwlYJ6L1EfsxKXQNFXa9l4kEv4QXn2d5BQnfCjFwsnolQmeSWP2XYh8O+R/YmhhFIAZhvzMctHnQ/XRRMnjADlpnM1mioyy4"
    "poDsUDKIyr92JvDK+T+1NRU/uQva9ScCF9YJCM3uLdxkocwx8eBN0hsOZYosrUq9FLL0+aNgHgybUyn4Kef197W4fht3yS2Z"
    "AUuNm4nM3VXRKlLgP3MYJnkOlBTVooyEoqC6raGH2RMdIegAs5JSIyd6xHkg8J2DaLhy+7WbpwqcHHCnM67sVyfRqQC9TP1a"
    "D0C0saoNL8sAZfHiDQPOLmvxt9z2+QpYeNZprWgGIGw5iZmLPexGHNLFd+i063HUpf5dYmjPwdL7970CW8PPQgjxW7OMOTlm"
    "Gkbf4YMsyiTrHbmVBjVPm3dA93q7wE3zsuhJ2icHkT7zlvHzz+nYtgJlWyoMtOPOjUunNM2pVXmMtXou0Rp0aR7YDIy9uqnY"
    "GIfBKYriMp/wL1apCyzih0EqcGhqNVspDDPJV0Slj3VKP7LaBTqz0LdTFo1WZAZNUZJW2sOHC8c9oXjNCM/o25qOPHowmzN9"
    "kWFBAtuetPHn5lD5PFvIdYMyjOtwRpFsPyAM8E6Nb7hP3pQnW8zvQ9YrfT6nqst39f5JHyjXh0HtVJgF+EjETS8H2EJSY3a9"
    "ExXPCRmCyXnKR9haJ95o+XRVMlJRucw9C34RwZ/po7oivV5jV+hSSOnTZuxx7rGH6md0ELhMKXO3PwaNE/s14SqOCBc9s9yj"
    "vGbC9fVe89KWNigVeejbIVlSs8uoHld9Bhespx8zZVheVQP9Oc9lVLXzmLteo4GNKkFYoJ7a5kCvx85vj9vAqKISOBuc6Vfu"
    "zH3cl8BYRVo5JiaxWRQxcTAtTbhH/RxOXHm+gc8gtdisilWLHaS13D0gmaq28gkUTCg="
)

GEMINI_3_8_FLASH_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
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
                            "I need to inspect `shop/validation.py` next to see how `validate_quantity` handles zero, negativ"
                            "e, and non-integer quantities."
                        ),
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "path": "shop/validation.py",
                                "start_line": 1,
                                "end_line": 50,
                            },
                            "id": "call_23592",
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
        "promptTokenCount": 354,
        "candidatesTokenCount": 64,
        "totalTokenCount": 750,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 354,
            },
        ],
        "thoughtsTokenCount": 332,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.8-flash",
    "responseId": "-CbQavzyBouHz7IP-Ma2uQw",
}

GEMINI_3_8_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "gemini/gemini-3.8-flash"},
    expected_provider_endpoint="/models/gemini-3.8-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        **GEMINI_GENERATE_CONTENT_REQUEST,
        "system_instruction": {"parts": [{"text": RESPONSES_REQUEST["instructions"]}]},
    },
    mock_provider_response={**GEMINI_GENERATE_CONTENT_RESPONSE, **GEMINI_3_8_FLASH_RESPONSE_FIELDS},
    expected_litellm_response={
        **GEMINI_RESPONSES_RESPONSE,
        "model": "gemini/gemini-3.8-flash",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": GEMINI_3_8_FLASH_THOUGHT,
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "I need to inspect `shop/validation.py` next to see how `validate_quantity` handles zero, negativ"
                            "e, and non-integer quantities."
                        ),
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 50}',
                "call_id": "call_23592__thought__" + GEMINI_3_8_FLASH_THOUGHT_SIGNATURE,
                "name": "read_file",
                "type": "function_call",
                "id": "fc_call_23592__thought__" + GEMINI_3_8_FLASH_THOUGHT_SIGNATURE,
                "namespace": None,
                "status": "completed",
                "provider_specific_fields": {
                    "thought_signature": GEMINI_3_8_FLASH_THOUGHT_SIGNATURE,
                },
            },
        ],
        "usage": {
            "input_tokens": 354,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 354,
                "video_tokens": None,
            },
            "output_tokens": 396,
            "output_tokens_details": {
                "audio_tokens": None,
                "reasoning_tokens": 332,
                "text_tokens": 64,
            },
            "total_tokens": 750,
            "cost": None,
        },
    },
)

GEMINI_3_1_PRO_PREVIEW_THOUGHT: Final = (
    "Okay, here's my summary, taking on the persona you requested:\n\n"
    "**Navigating the Validation Landscape**\n\n"
    "Alright, before I dive into the core logic, I really need to understand how the system handles t"
    "he edge cases – specifically, zero and negative quantities. I've got a gut feeling those could b"
    "e potential vulnerabilities or, at the very least, a source of unexpected behavior. So, my initi"
    "al move is clear: I'm going to dig into `shop/validation.py`. Understanding the validation strat"
    "egy upfront is critical; it informs everything downstream. This will give me a clear picture of "
    "how the system is designed to handle invalid or potentially malicious inputs and will set the st"
    "age for a more focused review of the main logic. I'll read that file first.\n\n\n"
)

GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE: Final = (
    "Ev8BCvwBAWkUfRM62a1j2olCssVKs9GECDQ5L1C7yrRsGmPVGbcIm+MrafD721G6/xzw4hoxWRaLYt24NwcS8gLJMwZMKF2U"
    "xOKF2qPxQIrxVkY0JG8s9A1vVBlu4mNsDGoBMPLc3CEX/Wb7sJ8u/oaXz8n/4yWDc9NgLwxdpzrDH9An0fmu2MGpZ+rL1nHj"
    "E+ZoNp6gKpaiRe3MmdBF2+Oy3MI6JZQnU9C/T9InT2F6NyzdZMhqSXhTf5qddxt9XLeabg8TKO4+GLKlmjB/T9NBOP00fQqS"
    "rLtPHkoQgxhn3S0Qj5kDAOgNXvJM4lyhlGyWIfBuHNN3VI9LpfVMhKU2"
)

GEMINI_3_1_PRO_PREVIEW_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": GEMINI_3_1_PRO_PREVIEW_THOUGHT,
                        "thought": True,
                    },
                    {
                        "text": "I will read the validation code first.",
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "end_line": 50,
                                "start_line": 1,
                                "path": "shop/validation.py",
                            },
                            "id": "call_25410",
                        },
                        "thoughtSignature": GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE,
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
        "promptTokenCount": 354,
        "candidatesTokenCount": 41,
        "totalTokenCount": 429,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 354,
            },
        ],
        "thoughtsTokenCount": 34,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.1-pro-preview",
    "responseId": "_ibQasKwGeCBqtsP84vwkAk",
}

GEMINI_3_1_PRO_PREVIEW_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "gemini/gemini-3.1-pro-preview"},
    expected_provider_endpoint="/models/gemini-3.1-pro-preview:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        **GEMINI_GENERATE_CONTENT_REQUEST,
        "system_instruction": {"parts": [{"text": RESPONSES_REQUEST["instructions"]}]},
    },
    mock_provider_response={**GEMINI_GENERATE_CONTENT_RESPONSE, **GEMINI_3_1_PRO_PREVIEW_RESPONSE_FIELDS},
    expected_litellm_response={
        **GEMINI_RESPONSES_RESPONSE,
        "model": "gemini/gemini-3.1-pro-preview",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": GEMINI_3_1_PRO_PREVIEW_THOUGHT,
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I will read the validation code first.",
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"end_line": 50, "start_line": 1, "path": "shop/validation.py"}',
                "call_id": "call_25410__thought__" + GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE,
                "name": "read_file",
                "type": "function_call",
                "id": "fc_call_25410__thought__" + GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE,
                "namespace": None,
                "status": "completed",
                "provider_specific_fields": {
                    "thought_signature": GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE,
                },
            },
        ],
        "usage": {
            "input_tokens": 354,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 354,
                "video_tokens": None,
            },
            "output_tokens": 75,
            "output_tokens_details": {
                "audio_tokens": None,
                "reasoning_tokens": 34,
                "text_tokens": 41,
            },
            "total_tokens": 429,
            "cost": None,
        },
    },
)

GEMINI_4_ARGON_THOUGHT: Final = (
    "**My Approach to Understanding Quantity Validation**\n\nOkay, so the user wants a focused review, "
    "and I need to start by understanding how this system handles potentially tricky quantity inputs."
    " The instructions are clear: first, I need to look at the validation, especially focusing on zer"
    "os, negatives, and non-integers. The prompt explicitly directs me to the `shop/validation.py` fi"
    "le to get this information.\n\nTherefore, I will begin by explaining that I need to examine `shop/"
    "validation.py` to understand the quantity validation logic. Then, I will use the `read_file` too"
    "l to examine the file content.\n\n\n"
)

GEMINI_4_ARGON_THOUGHT_SIGNATURE: Final = (
    "EvAFCu0FAWkUfROlLQU9gMYhPSBBS11ewOx/lkms/eSAHLg8GTk/GDF6MvJhDsQ30oCot3b97yi1ezgROfDed8moTDDJfXuy"
    "NG8P64+pIiiFeK6u08SH3/0urhf86d+iXJbesQFWRKw6nm4kO7opHmYyQcG+R3GVNeQmEUMc9ESoa2RyIbqCwtyg6GYdf6O5"
    "cyBQZNLH57JtXMbX+LrREJX5YpLoITdEu05L/d8GhtsSg0SdzTleD1pA61v3nppnmZP6bui6vGXiQY0ajCpxrAb28UgjgRvl"
    "w+R+rBniGL+7tVimEXLXF/i/IRtEvr27umOI+adyQrQOgQFzh8GN1qlpmrHnTamh+evurSEUukKBeUza17CsbERRZO7kSEx6"
    "N6EvZt1s4fiDaltSyvxAaxEJxZ/i/yj4DD3c5t1pLCtGCuOQ5yAZ2NBe42u7PDC/J19z+HUOE+EtyJYLlAbBpSHgSeTdHacw"
    "D6A3o3sQ8djcbA0Av+S/kkHK7atlowxZMpNYA1u/G2kJKMQtmpZEEAeLidiY0q2FR4oZfg7UjwBFJJE0wdEhPkY9teg2HZnw"
    "X4HoivSb9jYjihhupsfz55r33Re59L8BEv5PR4ES+Qh7XFSuq2qD2diAjbpDY7Dq3VjNJyCiSOqaDBFcF4E/CNo3SLvnL2Dq"
    "smLNoEPJuJUPku20SEcw0epMr6X3x1AcjIRW/0NLjSs6CqBR1j1EsZHiYhEiNNhtHvdke/KRywSFmh3Lu0x89YruvsvDf6aI"
    "rGLq+Dx8dKbKRvCz3GSdo9XhhRRr/Zq+P+YHk7lS3FFBtXiRXFMial3nOLD8o2g50nU6wcg9cX0+1nkPCx/zLO6poyC06EwP"
    "qteE+kUUPd18H5B0qLOKdnDm8n/8FhHl9VaVh7tLvAn2pob6Vu4PkLYCHzP1mqm0qXZ7DLqJSSeoBvstrg/WUqJIwuE18Ps9"
    "8Ph1VD5vaw3Igp0C+VV+DL74WF/SiXwW8kRvl3CxhA74Fo0="
)

GEMINI_4_ARGON_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": GEMINI_4_ARGON_THOUGHT, "thought": True},
                    {
                        "text": (
                            "I will read `shop/validation.py` to examine how zero, negative, and non-integer quantities are c"
                            "urrently handled in `validate_quantity`."
                        )
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {"start_line": 1, "path": "shop/validation.py", "end_line": 50},
                            "id": "call_18163055417662353838",
                        },
                        "thoughtSignature": GEMINI_4_ARGON_THOUGHT_SIGNATURE,
                    },
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
            "finishMessage": "Model generated function call(s).",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 412,
        "candidatesTokenCount": 70,
        "totalTokenCount": 613,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 412}],
        "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 70}],
        "thoughtsTokenCount": 131,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-4-argon",
    "responseId": "1AEDa_bbIt6yjdoPnJGJoAo",
}

GEMINI_4_ARGON_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "gemini/gemini-4-argon"},
    expected_provider_endpoint="/models/gemini-4-argon:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        **GEMINI_GENERATE_CONTENT_REQUEST,
        "system_instruction": {"parts": [{"text": RESPONSES_REQUEST["instructions"]}]},
    },
    mock_provider_response={**GEMINI_GENERATE_CONTENT_RESPONSE, **GEMINI_4_ARGON_RESPONSE_FIELDS},
    expected_litellm_response={
        **GEMINI_RESPONSES_RESPONSE,
        "model": "gemini/gemini-4-argon",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": GEMINI_4_ARGON_THOUGHT, "annotations": []}],
                "phase": None,
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "I will read `shop/validation.py` to examine how zero, negative, and non-integer quantities are c"
                            "urrently handled in `validate_quantity`."
                        ),
                        "annotations": [],
                    }
                ],
                "phase": None,
            },
            {
                "arguments": '{"start_line": 1, "path": "shop/validation.py", "end_line": 50}',
                "call_id": "call_18163055417662353838__thought__" + GEMINI_4_ARGON_THOUGHT_SIGNATURE,
                "name": "read_file",
                "type": "function_call",
                "id": "fc_call_18163055417662353838__thought__" + GEMINI_4_ARGON_THOUGHT_SIGNATURE,
                "namespace": None,
                "status": "completed",
                "provider_specific_fields": {"thought_signature": GEMINI_4_ARGON_THOUGHT_SIGNATURE},
            },
        ],
        "usage": {
            "input_tokens": 412,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 412,
                "video_tokens": None,
            },
            "output_tokens": 201,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 131, "text_tokens": 70},
            "total_tokens": 613,
            "cost": None,
        },
    },
)

GEMINI_3_5_FLASH_THOUGHT_SIGNATURE: Final = "ErkECrYEAWkUfRORlQ5j/gkLmyx08FVeIMgg57FJXrjcLwBlt9iuELI6Uc5h+NR/Vm+gEsbdZ5mgFTmKjPF9K/eft8IlXMKm6odJjQOgOYsbt2JHzANo0bpfTmlA6fIi0G2zLbvBVASA6Bdxu1aPQuO4voioQRwm2vomRxH1YbWp8sKXk0DBVbefosldrL0zLJFsi5dFYCtPvw0n9olPVgptHzEdiyXqG+63aPxTooARRQutUH0XWAKR0V+P7qWPt55QKlLaKQqKeBndg9JrEplJihg1sp++y+NAi18fsqXteUS2zIeDtdePGM/GS5oVibwE25zJoziPRdtJhGasFSaA7a3znhW9PF0pBIPAIRPKE4NsQ0FRhpy7ksIXY+0uJ4N+WPPejrtKK6z5x+P0tFkFP0ZNNPM8FZbir1ncVhVxkZS/wWmhc/8TZoRA9ghlTpYhHJ+C4fRVqQqnyRR3SDpVTzB4/sCjBlb434dTH0U3jB4h6V9b/Zx4k4pwUwZTNr2FfgOt2bR7u05DOa+H73OzsNG6zBnMYgBndQdRgk58+l4+UcZdpGKB0lkbHdfD2bminBypEmeJKNRpuc7Smuu0YxcZiY03tzzhHdrUmItqC39OEr2CzRcT9DjFpiWydo3ej9ZkEXeyxoMCckpMmGWh6xbjAnX9gkrPFmUE2rJqblDJWa51i6u/p9Y6ciCq6j4lAy9eBfULIRGQt9pKOXagltOnX0vR0MTDYgWe4dVDSzCI1TV7X/SQYjo="

GEMINI_3_5_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "gemini/gemini-3.5-flash",
        "max_output_tokens": 1024,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/models/gemini-3.5-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        "contents": [{"parts": [{"text": "Say hello."}], "role": "user"}],
        "generationConfig": {"max_output_tokens": 1024},
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello.", "thoughtSignature": GEMINI_3_5_FLASH_THOUGHT_SIGNATURE}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 2,
            "totalTokenCount": 137,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
            "thoughtsTokenCount": 125,
            "serviceTier": "standard",
        },
        "modelVersion": "gemini-3.5-flash",
        "responseId": "4hjEav32Mt6P6dkPn5iDqAg",
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "gemini/gemini-3.5-flash",
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
        "max_output_tokens": 1024,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 10,
                "video_tokens": None,
            },
            "output_tokens": 127,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 125, "text_tokens": 2},
            "total_tokens": 137,
            "cost": None,
        },
        "user": None,
        "store": None,
    },
)
