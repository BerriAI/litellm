"""Shared helpers for the Amazon Bedrock Mantle OpenAI-compatible provider.

Both helpers are pure functions of (model, model_cost) so the routing rules can be
unit-tested without patching global state.
"""


def mantle_supports_responses(model: str | None, model_cost: dict) -> bool:
    """Whether a Bedrock Mantle model can serve the native Responses API.

    Purely data-driven from the model's price-map capability signal -- either
    /v1/responses in supported_endpoints, or mode=responses -- both overridable
    via register_model and proxy model_info, so onboarding a model is a JSON
    change, never a code change. There is deliberately NO model-name match here:
    capability is per-model, not per-family (openai.gpt-oss-120b supports
    Responses while openai.gpt-oss-safeguard-120b does not, despite sharing the
    gpt-oss substring), so a substring gate would be wrong. A model absent from
    model_cost simply has no signal and returns False (chat-completions emulation).
    """
    entry = model_cost.get(f"bedrock_mantle/{model}", {})
    if "/v1/responses" in (entry.get("supported_endpoints") or []):
        return True
    return entry.get("mode") == "responses"


def mantle_base_segment(model: str | None, model_cost: dict) -> str:
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
