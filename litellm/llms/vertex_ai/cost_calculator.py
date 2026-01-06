# What is this?
## Cost calculation for Google AI Studio / Vertex AI models
from typing import Literal, Optional, Tuple, Union

import litellm
from litellm import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _is_above_128k,
    generic_cost_per_token,
)
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

models_without_dynamic_pricing = ["gemini-1.0-pro", "gemini-pro", "gemini-2"]


def cost_router(
    model: str,
    custom_llm_provider: str,
    call_type: Union[Literal["embedding", "aembedding"], str],
) -> Literal["cost_per_character", "cost_per_token"]:
    """
    Route the cost calc to the right place, based on model/call_type/etc.

    Returns
        - str, the specific google cost calc function it should route to.
    """
    if custom_llm_provider == "vertex_ai" and (
        "claude" in model
        or "llama" in model
        or "mistral" in model
        or "jamba" in model
        or "codestral" in model
        or "gemma" in model
    ):
        return "cost_per_token"
    elif custom_llm_provider == "vertex_ai" and (
        call_type == "embedding" or call_type == "aembedding"
    ):
        return "cost_per_token"
    elif custom_llm_provider == "vertex_ai" and ("gemini-2" in model):
        return "cost_per_token"
    return "cost_per_character"


def cost_per_character(
    model: str,
    custom_llm_provider: str,
    usage: Usage,
    prompt_characters: Optional[float] = None,
    completion_characters: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculates the cost per character for a given VertexAI model, input messages, and response object.

    Input:
        - model: str, the model name without provider prefix
        - custom_llm_provider: str, "vertex_ai-*"
        - prompt_characters: float, the number of input characters
        - completion_characters: float, the number of output characters

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd

    Raises:
        Exception if model requires >128k pricing, but model cost not mapped
    """
    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )

    ## GET MODEL INFO
    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )

    ## CALCULATE INPUT COST
    if prompt_characters is None:
        prompt_cost, _ = cost_per_token(
            model=model,
            custom_llm_provider=custom_llm_provider,
            usage=usage,
        )
    else:
        try:
            if (
                _is_above_128k(tokens=prompt_characters * 4)  # 1 token = 4 char
                and model not in models_without_dynamic_pricing
            ):
                ## check if character pricing, else default to token pricing
                assert (
                    "input_cost_per_character_above_128k_tokens" in model_info
                    and model_info["input_cost_per_character_above_128k_tokens"]
                    is not None
                ), "model info for model={} does not have 'input_cost_per_character_above_128k_tokens'-pricing for > 128k tokens\nmodel_info={}".format(
                    model, model_info
                )
                prompt_cost = (
                    prompt_characters
                    * model_info["input_cost_per_character_above_128k_tokens"]
                )
            else:
                assert (
                    "input_cost_per_character" in model_info
                    and model_info["input_cost_per_character"] is not None
                ), "model info for model={} does not have 'input_cost_per_character'-pricing\nmodel_info={}".format(
                    model, model_info
                )
                prompt_cost = prompt_characters * model_info["input_cost_per_character"]
        except Exception as e:
            verbose_logger.debug(
                "litellm.litellm_core_utils.llm_cost_calc.google.py::cost_per_character(): Exception occured - {}\nDefaulting to None".format(
                    str(e)
                )
            )
            prompt_cost, _ = cost_per_token(
                model=model,
                custom_llm_provider=custom_llm_provider,
                usage=usage,
            )

    ## CALCULATE OUTPUT COST
    if completion_characters is None:
        _, completion_cost = cost_per_token(
            model=model,
            custom_llm_provider=custom_llm_provider,
            usage=usage,
        )
    else:
        completion_tokens = usage.completion_tokens
        try:
            if (
                _is_above_128k(tokens=completion_characters * 4)  # 1 token = 4 char
                and model not in models_without_dynamic_pricing
            ):
                assert (
                    "output_cost_per_character_above_128k_tokens" in model_info
                    and model_info["output_cost_per_character_above_128k_tokens"]
                    is not None
                ), "model info for model={} does not have 'output_cost_per_character_above_128k_tokens' pricing\nmodel_info={}".format(
                    model, model_info
                )
                completion_cost = (
                    completion_tokens
                    * model_info["output_cost_per_character_above_128k_tokens"]
                )
            else:
                assert (
                    "output_cost_per_character" in model_info
                    and model_info["output_cost_per_character"] is not None
                ), "model info for model={} does not have 'output_cost_per_character'-pricing\nmodel_info={}".format(
                    model, model_info
                )
                completion_cost = (
                    completion_characters * model_info["output_cost_per_character"]
                )
        except Exception as e:
            verbose_logger.debug(
                "litellm.litellm_core_utils.llm_cost_calc.google.py::cost_per_character(): Exception occured - {}\nDefaulting to None".format(
                    str(e)
                )
            )
            _, completion_cost = cost_per_token(
                model=model,
                custom_llm_provider=custom_llm_provider,
                usage=usage,
            )

    return prompt_cost, completion_cost


def _handle_128k_pricing(
    model_info: ModelInfo,
    usage: Usage,
) -> Tuple[float, float]:
    ## CALCULATE INPUT COST
    input_cost_per_token_above_128k_tokens = model_info.get(
        "input_cost_per_token_above_128k_tokens"
    )
    output_cost_per_token_above_128k_tokens = model_info.get(
        "output_cost_per_token_above_128k_tokens"
    )

    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    if (
        _is_above_128k(tokens=prompt_tokens)
        and input_cost_per_token_above_128k_tokens is not None
    ):
        prompt_cost = prompt_tokens * input_cost_per_token_above_128k_tokens
    else:
        prompt_cost = prompt_tokens * model_info["input_cost_per_token"]

    ## CALCULATE OUTPUT COST
    output_cost_per_token_above_128k_tokens = model_info.get(
        "output_cost_per_token_above_128k_tokens"
    )
    if (
        _is_above_128k(tokens=completion_tokens)
        and output_cost_per_token_above_128k_tokens is not None
    ):
        completion_cost = completion_tokens * output_cost_per_token_above_128k_tokens
    else:
        completion_cost = completion_tokens * model_info["output_cost_per_token"]

    return prompt_cost, completion_cost


def cost_per_token(
    model: str,
    custom_llm_provider: str,
    usage: Usage,
) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - custom_llm_provider: str, either "vertex_ai-*" or "gemini"
        - prompt_tokens: float, the number of input tokens
        - completion_tokens: float, the number of output tokens

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd

    Raises:
        Exception if model requires >128k pricing, but model cost not mapped
    """

    ## GET MODEL INFO
    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )

    ## HANDLE 128k+ PRICING
    input_cost_per_token_above_128k_tokens = model_info.get(
        "input_cost_per_token_above_128k_tokens"
    )
    output_cost_per_token_above_128k_tokens = model_info.get(
        "output_cost_per_token_above_128k_tokens"
    )
    if (
        input_cost_per_token_above_128k_tokens is not None
        or output_cost_per_token_above_128k_tokens is not None
    ):
        return _handle_128k_pricing(
            model_info=model_info,
            usage=usage,
        )

    return generic_cost_per_token(
        model=model,
        custom_llm_provider=custom_llm_provider,
        usage=usage,
    )
