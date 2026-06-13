"""deepseek chat-completion response JSON -> IR ``ChatResponse``.

deepseek rides the httpx path with NO ``transform_response`` override: the
inherited base GPT transform runs ``convert_to_model_response_object`` over
a FRESH ``ModelResponse`` (model=None), so the response model is the BARE
wire model (the xai R4 / cometapi pattern; no ``deepseek/`` prefix —
pinned by test_differential_deepseek_response.py). The shared openai parser
is that normalizer's mirror, verbatim.
"""

from __future__ import annotations

from ..openai_compat.response import parse_response

__all__ = ("parse_response",)
