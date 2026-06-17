"""Shared helpers for the Amazon Bedrock Mantle OpenAI-compatible provider."""

from typing import Optional


def mantle_base_segment(model: Optional[str], model_cost: dict) -> str:
    """Return the base path segment for a Bedrock Mantle model's OpenAI surface.

    Data-driven from the model's price-map use_openai_responses_path flag
    (overridable via register_model / proxy model_info). Per the AWS model cards,
    gpt-5.x and the google gemma-4-* family carry that flag and are served on the
    /openai/v1 base (.../openai/v1/responses and .../openai/v1/chat/completions);
    every other model including gpt-oss uses the standard /v1 base. The segment is
    the base for the model's whole OpenAI-compatible surface, so both the chat and
    responses configs derive from it -- there is no separate model-name rule.
    """
    entry = model_cost.get(f"bedrock_mantle/{model}", {})
    return "openai/v1" if entry.get("use_openai_responses_path") is True else "v1"
