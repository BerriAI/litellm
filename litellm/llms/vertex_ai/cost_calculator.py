# What is this?
## Cost calculation for Google AI Studio / Vertex AI models
from collections.abc import Sequence
from typing import Final, Literal

import litellm
from litellm import verbose_logger
from litellm._cost import billing_context, price, require_priced, token_usage
from litellm._internal_context import current_billing_time
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    generic_cost_per_token,
    get_vertex_regional_endpoint_uplift,
    is_above_128k,
)
from litellm.rust_bridge import _native
from litellm.types.utils import ModelInfo, Usage

"""
Gemini pricing covers: 
- token
- image
- audio
- video
"""

"""
Vertex AI -> character based pricing 

Google AI Studio -> token based pricing
"""

models_without_dynamic_pricing: Final = ["gemini-1.0-pro", "gemini-pro", "gemini-2"]


def cost_router(
    model: str,
    custom_llm_provider: str,
    call_type: Literal["embedding", "aembedding"] | str,
) -> Literal["cost_per_character", "cost_per_token"]:
    """
    Route the cost calc to the right place, based on model/call_type/etc.

    Returns
        - str, the specific google cost calc function it should route to.
    """
    if (
        custom_llm_provider == "vertex_ai"
        and (
            "claude" in model
            or "llama" in model
            or "mistral" in model
            or "jamba" in model
            or "codestral" in model
            or "gemma" in model
        )
        or custom_llm_provider == "vertex_ai"
        and (call_type == "embedding" or call_type == "aembedding")
        or custom_llm_provider == "vertex_ai"
        and ("gemini-2" in model)
    ):
        return "cost_per_token"
    return "cost_per_character"


def _token_component_cost(
    model: str,
    model_info: ModelInfo,
    usage: Usage,
    direction: Literal["input", "output"],
    service_tier: str | None,
) -> float:
    if (
        model_info.get("input_cost_per_token_above_128k_tokens") is not None
        or model_info.get("output_cost_per_token_above_128k_tokens") is not None
    ):
        input_cost, output_cost = _handle_128k_pricing(model_info, usage, direction=direction)
        return input_cost if direction == "input" else output_cost
    priced: Final = require_priced(
        price(
            model_info,
            token_usage(usage, model_info, direction=direction),
            billing_context(current_billing_time(), service_tier),
        ),
        model,
    )
    return priced.input if direction == "input" else priced.output


def cost_per_character(
    model: str,
    custom_llm_provider: str,
    usage: Usage,
    prompt_characters: float | None = None,
    completion_characters: float | None = None,
    service_tier: str | None = None,
    vertex_location: str | None = None,
) -> tuple[float, float]:
    """
    Calculates the cost per character for a given VertexAI model, input messages, and response object.

    Input:
        - model: str, the model name without provider prefix
        - custom_llm_provider: str, "vertex_ai-*"
        - prompt_characters: float, the number of input characters
        - completion_characters: float, the number of output characters
        - service_tier: optional tier derived from Gemini trafficType
          ("priority" for ON_DEMAND_PRIORITY, "flex" for FLEX/batch).
        - vertex_location: the Vertex AI location serving the request; non-global
          locations apply the model's regional-endpoint uplift multiplier

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd

    Raises:
        Exception if model requires >128k pricing, but model cost not mapped
    """
    ## GET MODEL INFO
    model_info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)

    ## CALCULATE INPUT COST
    if prompt_characters is None:
        prompt_cost = _token_component_cost(model, model_info, usage, "input", service_tier)
    else:
        try:
            if (
                is_above_128k(tokens=prompt_characters * 4)  # 1 token = 4 char
                and model not in models_without_dynamic_pricing
            ):
                ## check if character pricing, else default to token pricing
                assert (
                    "input_cost_per_character_above_128k_tokens" in model_info
                    and model_info["input_cost_per_character_above_128k_tokens"] is not None
                ), (
                    f"model info for model={model} does not have 'input_cost_per_character_above_128k_tokens'-pricing for > 128k tokens\nmodel_info={model_info}"
                )
                prompt_cost = prompt_characters * model_info["input_cost_per_character_above_128k_tokens"]
            else:
                assert (
                    "input_cost_per_character" in model_info and model_info["input_cost_per_character"] is not None
                ), (
                    f"model info for model={model} does not have 'input_cost_per_character'-pricing\nmodel_info={model_info}"
                )
                prompt_cost = prompt_characters * model_info["input_cost_per_character"]
        except Exception as e:
            verbose_logger.debug(
                "litellm.litellm_core_utils.llm_cost_calc.google.py::cost_per_character(): Exception occured - %s\nDefaulting to None",
                e,
            )
            prompt_cost = _token_component_cost(model, model_info, usage, "input", service_tier)

    ## CALCULATE OUTPUT COST
    if completion_characters is None:
        completion_cost = _token_component_cost(model, model_info, usage, "output", service_tier)
    else:
        completion_tokens: Final = usage.completion_tokens
        try:
            if (
                is_above_128k(tokens=completion_characters * 4)  # 1 token = 4 char
                and model not in models_without_dynamic_pricing
            ):
                assert (
                    "output_cost_per_character_above_128k_tokens" in model_info
                    and model_info["output_cost_per_character_above_128k_tokens"] is not None
                ), (
                    f"model info for model={model} does not have 'output_cost_per_character_above_128k_tokens' pricing\nmodel_info={model_info}"
                )
                completion_cost = completion_tokens * model_info["output_cost_per_character_above_128k_tokens"]
            else:
                assert (
                    "output_cost_per_character" in model_info and model_info["output_cost_per_character"] is not None
                ), (
                    f"model info for model={model} does not have 'output_cost_per_character'-pricing\nmodel_info={model_info}"
                )
                completion_cost = completion_characters * model_info["output_cost_per_character"]
        except Exception as e:
            verbose_logger.debug(
                "litellm.litellm_core_utils.llm_cost_calc.google.py::cost_per_character(): Exception occured - %s\nDefaulting to None",
                e,
            )
            completion_cost = _token_component_cost(model, model_info, usage, "output", service_tier)

    vertex_uplift: Final = get_vertex_regional_endpoint_uplift(model_info, vertex_location)
    return prompt_cost * vertex_uplift, completion_cost * vertex_uplift


def _handle_128k_pricing(
    model_info: ModelInfo,
    usage: Usage,
    *,
    factors: Sequence[float] = (),
    direction: Literal["input", "output"] | None = None,
) -> tuple[float, float]:
    result: Final = require_priced(
        price(
            model_info,
            _native.UsageInput(
                {
                    **({"input_cost_per_token": usage.prompt_tokens} if direction != "output" else {}),
                    **({"output_cost_per_token": usage.completion_tokens} if direction != "input" else {}),
                },
                prompt_tokens=usage.prompt_tokens,
            ),
            billing_context(current_billing_time()),
            factors=factors,
            directional_token_threshold=128000,
        ),
        model_info.get("key", "vertex_ai"),
    )
    return result.input, result.output


def cost_per_token(
    model: str,
    custom_llm_provider: str,
    usage: Usage,
    service_tier: str | None = None,
    vertex_location: str | None = None,
) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - custom_llm_provider: str, either "vertex_ai-*" or "gemini"
        - prompt_tokens: float, the number of input tokens
        - completion_tokens: float, the number of output tokens
        - service_tier: optional tier derived from Gemini trafficType
          ("priority" for ON_DEMAND_PRIORITY, "flex" for FLEX/batch).
        - vertex_location: the Vertex AI location serving the request; non-global
          locations apply the model's regional-endpoint uplift multiplier

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd

    Raises:
        Exception if model requires >128k pricing, but model cost not mapped
    """

    ## GET MODEL INFO
    model_info: Final = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)

    ## HANDLE 128k+ PRICING
    input_cost_per_token_above_128k_tokens: Final = model_info.get("input_cost_per_token_above_128k_tokens")
    output_cost_per_token_above_128k_tokens: Final = model_info.get("output_cost_per_token_above_128k_tokens")
    if input_cost_per_token_above_128k_tokens is not None or output_cost_per_token_above_128k_tokens is not None:
        return _handle_128k_pricing(
            model_info=model_info,
            usage=usage,
            factors=(get_vertex_regional_endpoint_uplift(model_info, vertex_location),),
        )

    return generic_cost_per_token(
        model=model,
        custom_llm_provider=custom_llm_provider,
        usage=usage,
        service_tier=service_tier,
        vertex_location=vertex_location,
    )
