"""
Unit tests for Cerebras chat configuration.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # project root for ``litellm`` import

from litellm.llms.cerebras.chat import CerebrasConfig


class TestCerebrasGetSupportedOpenAIParams:
    """Validates the OpenAI-param allowlist exposed by ``CerebrasConfig``."""

    def test_max_retries_in_supported_params(self):
        """``max_retries`` is a standard OpenAI client parameter that Cerebras
        accepts. It must appear in the supported-param allowlist so callers
        can override the default retry count per request without LiteLLM
        stripping the value silently.
        """
        config = CerebrasConfig()
        params = config.get_supported_openai_params(model="llama3.1-8b")
        assert "max_retries" in params, (
            f"max_retries should be in the Cerebras supported_params allowlist; "
            f"got: {params!r}"
        )

    def test_core_openai_params_still_supported(self):
        """Regression guard: the standard OpenAI params Cerebras has always
        accepted must remain in the allowlist after the ``max_retries``
        addition (no accidental removals)."""
        config = CerebrasConfig()
        params = config.get_supported_openai_params(model="llama3.1-8b")
        for expected in (
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "seed",
            "stop",
            "stream",
            "temperature",
            "top_p",
            "tool_choice",
            "tools",
            "user",
        ):
            assert expected in params, (
                f"{expected!r} unexpectedly missing from Cerebras supported_params: "
                f"{params!r}"
            )
