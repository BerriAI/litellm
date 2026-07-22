"""
Helper util for handling bedrock-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Optional, Tuple

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token

if TYPE_CHECKING:
    from litellm.types.utils import Usage

# Routing prefixes stripped before model-info lookup, ordered longest-first so
# that "bedrock/converse/" is matched before the shorter "bedrock/" prefix.
_BEDROCK_ROUTING_PREFIXES = ("bedrock/converse/", "bedrock/", "converse/")


def _strip_bedrock_routing_prefix(model: str) -> str:
    """Return *model* with any leading Bedrock routing prefix removed.

    litellm may pass model strings that still carry provider or converse routing
    prefixes (e.g. ``bedrock/converse/us.anthropic.claude-sonnet-4-6``).
    ``get_model_info`` looks up the canonical key without these prefixes, so
    stripping them first ensures the correct price entry — including cross-region
    keys such as ``us.*`` — is found and used.
    """
    for prefix in _BEDROCK_ROUTING_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def cost_per_token(model: str, usage: "Usage", service_tier: Optional[str] = None) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Follows the same logic as Anthropic's cost per token calculation.

    Cross-region inference model prices (``us.*``, ``eu.*``, ``ap.*``) are
    already encoded in ``model_prices_and_context_window.json`` with the correct
    surcharge applied — no additional multiplier is needed here.  The prefix
    stripping ensures that callers using ``bedrock/converse/us.*`` style strings
    still resolve to the correct cross-region price entry.
    """
    return generic_cost_per_token(
        model=_strip_bedrock_routing_prefix(model),
        usage=usage,
        custom_llm_provider="bedrock",
        service_tier=service_tier,
    )
