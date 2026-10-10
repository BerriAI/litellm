"""
Tests for litellm.litellm_core_utils.llm_response_utils.get_api_base

`optional_params` is documented as accepting either a plain dict or a
LiteLLM_Params object, so both shapes must resolve `stream` the same way and
report the same streaming vs non-streaming Gemini / Vertex URL.
"""

import pytest

from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base
from litellm.types.router import LiteLLM_Params

GEMINI_STREAM = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent"
GEMINI_NON_STREAM = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"


@pytest.mark.parametrize(
    "optional_params, expected_api_base",
    [
        ({"stream": True}, GEMINI_STREAM),
        ({"stream": False}, GEMINI_NON_STREAM),
        ({}, GEMINI_NON_STREAM),
        (LiteLLM_Params(model="gemini/gemini-pro", stream=True), GEMINI_STREAM),
        (LiteLLM_Params(model="gemini/gemini-pro"), GEMINI_NON_STREAM),
    ],
)
def test_get_api_base_gemini_stream(
    optional_params: dict[str, bool] | LiteLLM_Params, expected_api_base: str
) -> None:
    assert get_api_base(model="gemini/gemini-pro", optional_params=optional_params) == expected_api_base


@pytest.mark.parametrize("stream, suffix", [(True, "streamGenerateContent"), (False, "generateContent")])
def test_get_api_base_vertex_stream_from_dict(stream: bool, suffix: str) -> None:
    api_base = get_api_base(
        model="gemini-pro",
        optional_params={
            "stream": stream,
            "vertex_location": "us-central1",
            "vertex_project": "my-project",
        },
    )

    assert api_base is not None
    assert api_base.endswith(f":{suffix}")
