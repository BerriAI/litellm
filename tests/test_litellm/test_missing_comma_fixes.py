"""
Tests to verify that string lists/Literals do not contain
implicit string concatenation caused by missing commas.
"""

import os
import sys
import typing

sys.path.insert(0, os.path.abspath("../.."))

from litellm.constants import clarifai_models
from litellm.types.llms.openai import OpenAIImageEditOptionalParams


def test_clarifai_models_no_implicit_concat():
    """
    Verify clarifai_models contains the two individual model strings,
    not a single concatenated string caused by a missing comma.

    Bug: missing comma between
        "clarifai/qwen.qwenLM.Qwen3-30B-A3B-Thinking-2507"
        "clarifai/openai.chat-completion.gpt-5-nano"
    resulted in the concatenated string:
        "clarifai/qwen.qwenLM.Qwen3-30B-A3B-Thinking-2507clarifai/openai.chat-completion.gpt-5-nano"
    """
    assert "clarifai/qwen.qwenLM.Qwen3-30B-A3B-Thinking-2507" in clarifai_models
    assert "clarifai/openai.chat-completion.gpt-5-nano" in clarifai_models

    # The concatenated (invalid) string should NOT be present
    bad_concat = (
        "clarifai/qwen.qwenLM.Qwen3-30B-A3B-Thinking-2507"
        "clarifai/openai.chat-completion.gpt-5-nano"
    )
    assert bad_concat not in clarifai_models


def test_openai_image_edit_params_no_implicit_concat():
    """
    Verify OpenAIImageEditOptionalParams Literal contains both "mask"
    and "output_compression" as separate values, not a concatenated
    "maskoutput_compression" caused by a missing comma.
    """
    valid_values = typing.get_args(OpenAIImageEditOptionalParams)

    assert "mask" in valid_values, (
        f'"mask" not found in OpenAIImageEditOptionalParams. Got: {valid_values}'
    )
    assert "output_compression" in valid_values, (
        f'"output_compression" not found in OpenAIImageEditOptionalParams. Got: {valid_values}'
    )

    # The concatenated (invalid) string should NOT be present
    assert "maskoutput_compression" not in valid_values, (
        '"maskoutput_compression" found — likely a missing comma between "mask" and "output_compression"'
    )
