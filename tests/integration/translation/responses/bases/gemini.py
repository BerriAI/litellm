from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase

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
        "generationConfig": {"max_output_tokens": 1024, "temperature": 1.0},
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

GEMINI_3_8_FLASH_THOUGHT_SIGNATURE: Final = "EtsDCtgDAWkUfRODYoLRWC+DajYQxOvsLgPh0m8j4NTnd7BflzgPBKFfPW+PU1XsMQuEzviI1qk5mYI0qCfOQNf84PAXXvFA5hMYl+YObaND4G+ZtCdYcolFVfPJQqgK6Kpv20n9hZfLt5JzOS2+HRCLZaokIsZFadN++wqEeEkWQhnKdLGH1lM0fn8Fj/pYq95YLGnB90B8Oaj4qyG6ost2dzRAeAzFSXAko1mD/IgsDrDhEumngCqotdAbPW4jUGYOGDpoXLrBzQZvGa9blRC3ep6NLT0EYMnXImLFoZaLLIBMzVDsmmL0qOg4Gu+uNJlY6cDmtqRgkrcvuvGhh8+lrjUMJVigSsTAoKsTnT3OyCqdNqa+R2aD4WTl1uBFyGY7yXpZ9skQPkV210QNOylZ6exaMA51+W/mohL5j5+OJX7xtVfRIpjp4e0PLkEPxnuviX6OU4ykWZSSiztSXzogbrmnwP7faclRXTXHE5pFlM8y7gDJ9NEwxd5vvIMLdaK2dRWBNtvjI3Cj56sLKygH/j0mQSnt3PonE6Jv6XMJDPbuIzsjzX8t/j3u6WP0bVGnifNNEfIqt6YcKgeWlUTnabRGkWg/fBSyYbiTTEIGhketv57hA9oH7jJZbw=="

GEMINI_3_8_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "gemini/gemini-3.8-flash",
        "max_output_tokens": 1024,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/models/gemini-3.8-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        "contents": [{"parts": [{"text": "Say hello."}], "role": "user"}],
        "generationConfig": {"max_output_tokens": 1024, "temperature": 1.0},
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello.", "thoughtSignature": GEMINI_3_8_FLASH_THOUGHT_SIGNATURE}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 2,
            "totalTokenCount": 120,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
            "thoughtsTokenCount": 108,
            "serviceTier": "standard",
        },
        "modelVersion": "gemini-3.8-flash",
        "responseId": "5BjEaunaCtadz7IP-8WKsAk",
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "gemini/gemini-3.8-flash",
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
            "output_tokens": 110,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 108, "text_tokens": 2},
            "total_tokens": 120,
            "cost": None,
        },
        "user": None,
        "store": None,
    },
)

GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE: Final = "Ev8DCvwDAWkUfRMrwgm49bezSdfm90OeW6KeR4kKqExF8s+EorBlW6NWc6XLqdQ2dA5xZ4p0CPNrjuxemR5dt9ch3qMTSgIHTcKKAnxJ0eqsjR61EJWFP4JNdigiymymc7UNs/zLZer+qPH2XQLD9r85O3NVBeupYk6xy6395CZYygF9oVJD3WNXwlefvThnvH/3rDsnO0FBfcrvxRHiSVTD1Moe+uTVV2w3vKKSCxUb64w5lquEjFx+AO/jiJIc3McPvAOvUr0I/2fMCWLcO7Y5sV6zuN8qpaQKitC/Ev09cl3SAbKzwjO3gBgYVmF5PcPY5HT8S3bwCez2aOgX7BCN+FItlZ4wMsZStLIY38XMQtbibVRHmiufN86IMkoD8Yxb1lnK7aaS+anSPkn4M4zkkiwAUdjt14k6JvMBk9J4duvTHLP0BiSaLLkTe6Ufj9cRMxZKCP0ew4DiLMwejuQoSpC4aLP1gmli4eOUcyq/g2/o0kByY8Fl2vb54eiXADp4fhhIEAZFe4J/0x29ZZVKx8KEInDahP7tslzwah3PWfX/K1jXzqX93mo4a/0Ec1bM/sqwiEWSHNhrPpyXcTkmWQ1Bot+PTARnzAjeXvoTR5sajbNKj1SdCC0RY7zb8OqGZHDQxgNw7ghTjYMTdBJsdCMQfnl3dvqmw6zM/aplZg=="

GEMINI_3_1_PRO_PREVIEW_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "gemini/gemini-3.1-pro-preview",
        "max_output_tokens": 1024,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/models/gemini-3.1-pro-preview:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        "contents": [{"parts": [{"text": "Say hello."}], "role": "user"}],
        "generationConfig": {"max_output_tokens": 1024, "temperature": 1.0},
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello.", "thoughtSignature": GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 2,
            "totalTokenCount": 115,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
            "thoughtsTokenCount": 103,
            "serviceTier": "standard",
        },
        "modelVersion": "gemini-3.1-pro-preview",
        "responseId": "5RjEaqfwC7XYqtsPp6TuoAY",
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "gemini/gemini-3.1-pro-preview",
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
            "output_tokens": 105,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 103, "text_tokens": 2},
            "total_tokens": 115,
            "cost": None,
        },
        "user": None,
        "store": None,
    },
)
