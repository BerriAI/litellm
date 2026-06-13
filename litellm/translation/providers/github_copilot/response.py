"""github_copilot response parsing: the shared openai parser verbatim.

On the SDK chat path the LIVE normalizer is
``convert_to_model_response_object`` (``GithubCopilotConfig.transform_response``
and its Anthropic-native-body synthesis are DEAD on this path — they wake only
under the EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER env secret, researcher-5
§1.1). The response model is the seam preset + cdr re-prefix ->
``github_copilot/{wire_model}`` (the compat_sdk seam arm; construction arm
"openai" — NO parser-side prefix, the same as every other SDK-path member).
"""

from __future__ import annotations

from ..openai_compat.response import parse_response

__all__ = ("parse_response",)
