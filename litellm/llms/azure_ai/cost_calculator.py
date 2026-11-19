"""
Azure AI cost calculation helper.
Handles Azure AI Foundry Model Router flat cost and other Azure AI specific pricing.
"""

from dataclasses import replace
from typing import Final

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import (
    AdjustFacts,
    BranchFacts,
    CustomPricing,
    ReceiptFacts,
    branch_receipt,
    custom_pricing_usage,
    custom_receipt,
)
from litellm.rust_bridge._native import NativeReceipt
from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info


def is_azure_model_router(model: str) -> bool:
    """
    Check if the model is Azure AI Foundry Model Router.

    Detects patterns like:
    - "azure-model-router"
    - "model-router"
    - "model_router/<actual-model>"
    - "model-router/<actual-model>"

    Args:
        model: The model name

    Returns:
        bool: True if this is a model router model
    """
    model_lower: Final = model.lower()
    return "model-router" in model_lower or "model_router" in model_lower or model_lower == "azure-model-router"


ROUTER_FEE_ENTRY_NAMES: Final = frozenset({"model-router", "model_router"})


def is_router_fee_entry(model: str) -> bool:
    return model.lower().removeprefix("azure_ai/") in ROUTER_FEE_ENTRY_NAMES


def _router_fee_entry_name(model: str) -> str:
    entry_name: Final = model.lower().removeprefix("azure_ai/")
    return entry_name if entry_name in ROUTER_FEE_ENTRY_NAMES else "model_router"


def router_fee_row(model: str) -> ModelInfo:
    """The cost map row of the Model Router fee ``model`` names. Raises when the cost map has none."""
    return get_model_info(model=_router_fee_entry_name(model), custom_llm_provider="azure_ai")


def router_fee_name(model: str, request_model: str | None) -> str | None:
    """The Model Router name whose fee the priced ``model`` adds to its prompt cost, ``None`` for none: a response
    priced as the router entry already carries it."""
    if is_router_fee_entry(model):
        return None
    if is_azure_model_router(model):
        return model
    if request_model is not None and is_azure_model_router(request_model):
        return request_model
    return None


def _response_model_row(model: str) -> ModelInfo | None:
    try:
        return get_model_info(model=model, custom_llm_provider="azure_ai")
    except Exception as e:  # noqa: BLE001  # get_model_info raises a bare Exception for an unmapped model
        if not is_azure_model_router(model):
            raise
        verbose_logger.debug(
            "Azure AI Model Router: model '%s' not in cost map, only the routing fee applies. Error: %s", model, e
        )
        return None


def receipt(
    model: str,
    usage: Usage,
    request_model: str | None = None,
    service_tier: str | None = None,
    data_residency: str | None = None,
    vertex_location: str | None = None,
    facts: ReceiptFacts | None = None,
    adjust: AdjustFacts | None = None,
) -> NativeReceipt:
    """The response model's own tokens priced by ``generic_cost_per_token`` with the service tier, plus the Model Router
    fee on the prompt once when the priced name or ``request_model`` is a router name. A router name missing from the
    cost map bills the fee alone."""
    fee_name: Final = router_fee_name(model=model, request_model=request_model)
    beside: Final = facts or ReceiptFacts()
    with_fee: Final = (
        beside if fee_name is None else replace(beside, router_fee="prompt", fee_row=router_fee_row(fee_name))
    )
    row: Final = _response_model_row(model)
    if row is None:
        return custom_receipt(
            model,
            "azure_ai",
            custom_pricing_usage(
                prompt_tokens=usage.prompt_tokens, completion_tokens=0, cached_tokens=0, cache_creation_tokens=0
            ),
            CustomPricing(per_token=None, per_second=0.0),
            None,
            service_tier=service_tier,
            data_residency=data_residency,
            vertex_location=vertex_location,
            receipt=with_fee,
            adjust=adjust,
        )
    return branch_receipt(
        model,
        "azure_ai",
        usage,
        row,
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        branch=BranchFacts(rules="generic_tier"),
        receipt=with_fee,
        adjust=adjust,
    )


def cost_per_token(
    model: str,
    usage: Usage,
    response_time_ms: float | None = 0.0,
    request_model: str | None = None,
    service_tier: str | None = None,
) -> tuple[float, float]:
    """
    Price the response model's own tokens for Azure AI, plus the Model Router fee exactly once when either the
    priced name or request_model is a Model Router name.

    A response priced as the router entry itself already carries the fee, so nothing is added on top of it. A
    router deployment name that is missing from the cost map prices at the fee alone.

    Args:
        model: str, the model name without provider prefix (from response)
        usage: LiteLLM Usage block
        response_time_ms: Optional response time in milliseconds
        request_model: Optional[str], the original request model name; a Model Router name adds the routing fee
        service_tier: Optional service tier the request was priced on

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd

    Raises:
        ValueError: If a model that is not a Model Router name is missing from the cost map
    """
    priced: Final = receipt(model=model, usage=usage, request_model=request_model, service_tier=service_tier)
    return priced.prompt_cost, priced.completion_cost
