"""sagemaker_chat response parsing: the shared openai parser verbatim.

``SagemakerChatConfig`` inherits the base ``transform_response`` (LIVE on
the httpx path) -> ``convert_to_model_response_object`` over a fresh
ModelResponse with model None: the BARE wire model rides (the
cometapi/xai R4 shape; NO seam preset, construction arm "openai")."""

from __future__ import annotations

from ..openai_compat.response import parse_response

__all__ = ("parse_response",)
