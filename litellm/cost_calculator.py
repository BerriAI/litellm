# What is this?
## File for 'response_cost' calculation in Logging
import time
import traceback
from typing import Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel

import litellm
import litellm._logging
from litellm import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.google import (
    cost_per_character as google_cost_per_character,
)
from litellm.litellm_core_utils.llm_cost_calc.google import (
    cost_per_token as google_cost_per_token,
)
from litellm.litellm_core_utils.llm_cost_calc.google import (
    cost_router as google_cost_router,
)
from litellm.litellm_core_utils.llm_cost_calc.utils import _generic_cost_per_character
from litellm.llms.anthropic.cost_calculation import (
    cost_per_token as anthropic_cost_per_token,
)
from litellm.llms.azure_ai.cost_calculator import (
    cost_per_query as azure_ai_rerank_cost_per_query,
)
from litellm.llms.cohere.cost_calculator import (
    cost_per_query as cohere_rerank_cost_per_query,
)
from litellm.llms.databricks.cost_calculator import (
    cost_per_token as databricks_cost_per_token,
)
from litellm.llms.fireworks_ai.cost_calculator import (
    cost_per_token as fireworks_ai_cost_per_token,
)
from litellm.llms.together_ai.cost_calculator import get_model_params_and_category
from litellm.rerank_api.types import RerankResponse
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.router import SPECIAL_MODEL_INFO_PARAMS
from litellm.types.utils import PassthroughCallTypes, Usage
from litellm.utils import (
    CallTypes,
    CostPerToken,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    TextCompletionResponse,
    TranscriptionResponse,
    print_verbose,
    token_counter,
)


def _cost_per_token_custom_pricing_helper(
    prompt_tokens: float = 0,
    completion_tokens: float = 0,
    response_time_ms=None,
    ### CUSTOM PRICING ###
    custom_cost_per_token: Optional[CostPerToken] = None,
    custom_cost_per_second: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Internal helper function for calculating cost, if custom pricing given"""
    if custom_cost_per_token is None and custom_cost_per_second is None:
        return None

    if custom_cost_per_token is not None:
        input_cost = custom_cost_per_token["input_cost_per_token"] * prompt_tokens
        output_cost = custom_cost_per_token["output_cost_per_token"] * completion_tokens
        return input_cost, output_cost
    elif custom_cost_per_second is not None:
        output_cost = custom_cost_per_second * response_time_ms / 1000  # type: ignore
        return 0, output_cost

    return None


def cost_per_token(
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    response_time_ms=None,
    custom_llm_provider: Optional[str] = None,
    region_name=None,
    ### CHARACTER PRICING ###
    prompt_characters: int = 0,
    completion_characters: int = 0,
    ### PROMPT CACHING PRICING ### - used for anthropic
    cache_creation_input_tokens: Optional[int] = 0,
    cache_read_input_tokens: Optional[int] = 0,
    ### CUSTOM PRICING ###
    custom_cost_per_token: Optional[CostPerToken] = None,
    custom_cost_per_second: Optional[float] = None,
    ### NUMBER OF QUERIES ###
    number_of_queries: Optional[int] = None,
    ### CALL TYPE ###
    call_type: Literal[
        "embedding",
        "aembedding",
        "completion",
        "acompletion",
        "atext_completion",
        "text_completion",
        "image_generation",
        "aimage_generation",
        "moderation",
        "amoderation",
        "atranscription",
        "transcription",
        "aspeech",
        "speech",
        "rerank",
        "arerank",
    ] = "completion",
) -> Tuple[float, float]:  # type: ignore
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Parameters:
        model (str): The name of the model to use. Default is ""
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the completion.
        response_time (float): The amount of time, in milliseconds, it took the call to complete.
        prompt_characters (float): The number of characters in the prompt. Used for vertex ai cost calculation.
        completion_characters (float): The number of characters in the completion response. Used for vertex ai cost calculation.
        custom_llm_provider (str): The llm provider to whom the call was made (see init.py for full list)
        custom_cost_per_token: Optional[CostPerToken]: the cost per input + output token for the llm api call.
        custom_cost_per_second: Optional[float]: the cost per second for the llm api call.
        call_type: Optional[str]: the call type

    Returns:
        tuple: A tuple containing the cost in USD dollars for prompt tokens and completion tokens, respectively.
    """
    if model is None:
        raise Exception("Invalid arg. Model cannot be none.")

    ## RECONSTRUCT USAGE BLOCK ##
    usage_block = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )

    ## CUSTOM PRICING ##
    response_cost = _cost_per_token_custom_pricing_helper(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_time_ms=response_time_ms,
        custom_cost_per_second=custom_cost_per_second,
        custom_cost_per_token=custom_cost_per_token,
    )

    if response_cost is not None:
        return response_cost[0], response_cost[1]

    # given
    prompt_tokens_cost_usd_dollar: float = 0
    completion_tokens_cost_usd_dollar: float = 0
    model_cost_ref = litellm.model_cost
    model_with_provider = model
    if custom_llm_provider is not None:
        model_with_provider = custom_llm_provider + "/" + model
        if region_name is not None:
            model_with_provider_and_region = (
                f"{custom_llm_provider}/{region_name}/{model}"
            )
            if (
                model_with_provider_and_region in model_cost_ref
            ):  # use region based pricing, if it's available
                model_with_provider = model_with_provider_and_region
    else:
        _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model)

    model_without_prefix = model
    model_parts = model.split("/", 1)
    if len(model_parts) > 1:
        model_without_prefix = model_parts[1]
    else:
        model_without_prefix = model
    """
    Code block that formats model to lookup in litellm.model_cost
    Option1. model = "bedrock/ap-northeast-1/anthropic.claude-instant-v1". This is the most accurate since it is region based. Should always be option 1
    Option2. model = "openai/gpt-4"       - model = provider/model
    Option3. model = "anthropic.claude-3" - model = model
    """
    if (
        model_with_provider in model_cost_ref
    ):  # Option 2. use model with provider, model = "openai/gpt-4"
        model = model_with_provider
    elif model in model_cost_ref:  # Option 1. use model passed, model="gpt-4"
        model = model
    elif (
        model_without_prefix in model_cost_ref
    ):  # Option 3. if user passed model="bedrock/anthropic.claude-3", use model="anthropic.claude-3"
        model = model_without_prefix

    # see this https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models
    print_verbose(f"Looking up model={model} in model_cost_map")
    if custom_llm_provider == "vertex_ai":
        cost_router = google_cost_router(
            model=model_without_prefix,
            custom_llm_provider=custom_llm_provider,
            prompt_characters=prompt_characters,
            completion_characters=completion_characters,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            call_type=call_type,
        )
        if cost_router == "cost_per_character":
            return google_cost_per_character(
                model=model_without_prefix,
                custom_llm_provider=custom_llm_provider,
                prompt_characters=prompt_characters,
                completion_characters=completion_characters,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        elif cost_router == "cost_per_token":
            return google_cost_per_token(
                model=model_without_prefix,
                custom_llm_provider=custom_llm_provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
    elif custom_llm_provider == "anthropic":
        return anthropic_cost_per_token(model=model, usage=usage_block)
    elif custom_llm_provider == "databricks":
        return databricks_cost_per_token(model=model, usage=usage_block)
    elif custom_llm_provider == "fireworks_ai":
        return fireworks_ai_cost_per_token(model=model, usage=usage_block)
    elif custom_llm_provider == "gemini":
        return google_cost_per_token(
            model=model_without_prefix,
            custom_llm_provider=custom_llm_provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    elif call_type == "speech" or call_type == "aspeech":
        prompt_cost, completion_cost = _generic_cost_per_character(
            model=model_without_prefix,
            custom_llm_provider=custom_llm_provider,
            prompt_characters=prompt_characters,
            completion_characters=completion_characters,
            custom_prompt_cost=None,
            custom_completion_cost=0,
        )
        if prompt_cost is None or completion_cost is None:
            raise ValueError(
                "cost for tts call is None. prompt_cost={}, completion_cost={}, model={}, custom_llm_provider={}, prompt_characters={}, completion_characters={}".format(
                    prompt_cost,
                    completion_cost,
                    model_without_prefix,
                    custom_llm_provider,
                    prompt_characters,
                    completion_characters,
                )
            )
        return prompt_cost, completion_cost
    elif call_type == "arerank" or call_type == "rerank":
        return rerank_cost(
            model=model,
            custom_llm_provider=custom_llm_provider,
        )
    elif model in model_cost_ref:
        print_verbose(f"Success: model={model} in model_cost_map")
        print_verbose(
            f"prompt_tokens={prompt_tokens}; completion_tokens={completion_tokens}"
        )
        if (
            model_cost_ref[model].get("input_cost_per_token", None) is not None
            and model_cost_ref[model].get("output_cost_per_token", None) is not None
        ):
            ## COST PER TOKEN ##
            prompt_tokens_cost_usd_dollar = (
                model_cost_ref[model]["input_cost_per_token"] * prompt_tokens
            )
            completion_tokens_cost_usd_dollar = (
                model_cost_ref[model]["output_cost_per_token"] * completion_tokens
            )
        elif (
            model_cost_ref[model].get("output_cost_per_second", None) is not None
            and response_time_ms is not None
        ):
            print_verbose(
                f"For model={model} - output_cost_per_second: {model_cost_ref[model].get('output_cost_per_second')}; response time: {response_time_ms}"
            )
            ## COST PER SECOND ##
            prompt_tokens_cost_usd_dollar = 0
            completion_tokens_cost_usd_dollar = (
                model_cost_ref[model]["output_cost_per_second"]
                * response_time_ms
                / 1000
            )
        elif (
            model_cost_ref[model].get("input_cost_per_second", None) is not None
            and response_time_ms is not None
        ):
            print_verbose(
                f"For model={model} - input_cost_per_second: {model_cost_ref[model].get('input_cost_per_second')}; response time: {response_time_ms}"
            )
            ## COST PER SECOND ##
            prompt_tokens_cost_usd_dollar = (
                model_cost_ref[model]["input_cost_per_second"] * response_time_ms / 1000
            )
            completion_tokens_cost_usd_dollar = 0.0
        print_verbose(
            f"Returned custom cost for model={model} - prompt_tokens_cost_usd_dollar: {prompt_tokens_cost_usd_dollar}, completion_tokens_cost_usd_dollar: {completion_tokens_cost_usd_dollar}"
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    elif "ft:gpt-3.5-turbo" in model:
        print_verbose(f"Cost Tracking: {model} is an OpenAI FinteTuned LLM")
        # fuzzy match ft:gpt-3.5-turbo:abcd-id-cool-litellm
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref["ft:gpt-3.5-turbo"]["input_cost_per_token"] * prompt_tokens
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref["ft:gpt-3.5-turbo"]["output_cost_per_token"]
            * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    elif "ft:gpt-4-0613" in model:
        print_verbose(f"Cost Tracking: {model} is an OpenAI FinteTuned LLM")
        # fuzzy match ft:gpt-4-0613:abcd-id-cool-litellm
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref["ft:gpt-4-0613"]["input_cost_per_token"] * prompt_tokens
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref["ft:gpt-4-0613"]["output_cost_per_token"] * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    elif "ft:gpt-4o-2024-05-13" in model:
        print_verbose(f"Cost Tracking: {model} is an OpenAI FinteTuned LLM")
        # fuzzy match ft:gpt-4o-2024-05-13:abcd-id-cool-litellm
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref["ft:gpt-4o-2024-05-13"]["input_cost_per_token"]
            * prompt_tokens
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref["ft:gpt-4o-2024-05-13"]["output_cost_per_token"]
            * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar

    elif "ft:davinci-002" in model:
        print_verbose(f"Cost Tracking: {model} is an OpenAI FinteTuned LLM")
        # fuzzy match ft:davinci-002:abcd-id-cool-litellm
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref["ft:davinci-002"]["input_cost_per_token"] * prompt_tokens
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref["ft:davinci-002"]["output_cost_per_token"]
            * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    elif "ft:babbage-002" in model:
        print_verbose(f"Cost Tracking: {model} is an OpenAI FinteTuned LLM")
        # fuzzy match ft:babbage-002:abcd-id-cool-litellm
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref["ft:babbage-002"]["input_cost_per_token"] * prompt_tokens
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref["ft:babbage-002"]["output_cost_per_token"]
            * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    elif model in litellm.azure_llms:
        verbose_logger.debug(f"Cost Tracking: {model} is an Azure LLM")
        model = litellm.azure_llms[model]
        verbose_logger.debug(
            f"applying cost={model_cost_ref[model]['input_cost_per_token']} for prompt_tokens={prompt_tokens}"
        )
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref[model]["input_cost_per_token"] * prompt_tokens
        )
        verbose_logger.debug(
            f"applying cost={model_cost_ref[model]['output_cost_per_token']} for completion_tokens={completion_tokens}"
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref[model]["output_cost_per_token"] * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    elif model in litellm.azure_embedding_models:
        verbose_logger.debug(f"Cost Tracking: {model} is an Azure Embedding Model")
        model = litellm.azure_embedding_models[model]
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref[model]["input_cost_per_token"] * prompt_tokens
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref[model]["output_cost_per_token"] * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    else:
        # if model is not in model_prices_and_context_window.json. Raise an exception-let users know
        error_str = f"Model not in model_prices_and_context_window.json. You passed model={model}, custom_llm_provider={custom_llm_provider}. Register pricing for model - https://docs.litellm.ai/docs/proxy/custom_pricing\n"
        raise litellm.exceptions.NotFoundError(  # type: ignore
            message=error_str,
            model=model,
            llm_provider="",
        )


def get_replicate_completion_pricing(completion_response: dict, total_time=0.0):
    # see https://replicate.com/pricing
    # for all litellm currently supported LLMs, almost all requests go to a100_80gb
    a100_80gb_price_per_second_public = (
        0.001400  # assume all calls sent to A100 80GB for now
    )
    if total_time == 0.0:  # total time is in ms
        start_time = completion_response.get("created", time.time())
        end_time = getattr(completion_response, "ended", time.time())
        total_time = end_time - start_time

    return a100_80gb_price_per_second_public * total_time / 1000


def has_hidden_params(obj: Any) -> bool:
    return hasattr(obj, "_hidden_params")


def _select_model_name_for_cost_calc(
    model: Optional[str],
    completion_response: Union[BaseModel, dict, str],
    base_model: Optional[str] = None,
    custom_pricing: Optional[bool] = None,
) -> Optional[str]:
    """
    1. If custom pricing is true, return received model name
    2. If base_model is set (e.g. for azure models), return that
    3. If completion response has model set return that
    4. If model is passed in return that
    """
    if custom_pricing is True:
        return model

    if base_model is not None:
        return base_model

    return_model = model
    if isinstance(completion_response, str):
        return return_model

    elif return_model is None and hasattr(completion_response, "get"):
        return_model = completion_response.get("model", "")  # type: ignore
    hidden_params = getattr(completion_response, "_hidden_params", None)

    if hidden_params is not None:
        if (
            hidden_params.get("model", None) is not None
            and len(hidden_params["model"]) > 0
        ):
            return_model = hidden_params.get("model", model)

    return return_model


def completion_cost(
    completion_response=None,
    model: Optional[str] = None,
    prompt="",
    messages: List = [],
    completion="",
    total_time=0.0,  # used for replicate, sagemaker
    call_type: Literal[
        "embedding",
        "aembedding",
        "completion",
        "acompletion",
        "atext_completion",
        "text_completion",
        "image_generation",
        "aimage_generation",
        "moderation",
        "amoderation",
        "atranscription",
        "transcription",
        "aspeech",
        "speech",
        "rerank",
        "arerank",
    ] = "completion",
    ### REGION ###
    custom_llm_provider=None,
    region_name=None,  # used for bedrock pricing
    ### IMAGE GEN ###
    size=None,
    quality=None,
    n=None,  # number of images
    ### CUSTOM PRICING ###
    custom_cost_per_token: Optional[CostPerToken] = None,
    custom_cost_per_second: Optional[float] = None,
) -> float:
    """
    Calculate the cost of a given completion call fot GPT-3.5-turbo, llama2, any litellm supported llm.

    Parameters:
        completion_response (litellm.ModelResponses): [Required] The response received from a LiteLLM completion request.

        [OPTIONAL PARAMS]
        model (str): Optional. The name of the language model used in the completion calls
        prompt (str): Optional. The input prompt passed to the llm
        completion (str): Optional. The output completion text from the llm
        total_time (float): Optional. (Only used for Replicate LLMs) The total time used for the request in seconds
        custom_cost_per_token: Optional[CostPerToken]: the cost per input + output token for the llm api call.
        custom_cost_per_second: Optional[float]: the cost per second for the llm api call.

    Returns:
        float: The cost in USD dollars for the completion based on the provided parameters.

    Exceptions:
        Raises exception if model not in the litellm model cost map. Register model, via custom pricing or PR - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json


    Note:
        - If completion_response is provided, the function extracts token information and the model name from it.
        - If completion_response is not provided, the function calculates token counts based on the model and input text.
        - The cost is calculated based on the model, prompt tokens, and completion tokens.
        - For certain models containing "togethercomputer" in the name, prices are based on the model size.
        - For un-mapped Replicate models, the cost is calculated based on the total time used for the request.
    """
    try:
        if (
            (call_type == "aimage_generation" or call_type == "image_generation")
            and model is not None
            and isinstance(model, str)
            and len(model) == 0
            and custom_llm_provider == "azure"
        ):
            model = "dall-e-2"  # for dall-e-2, azure expects an empty model name
        # Handle Inputs to completion_cost
        prompt_tokens = 0
        prompt_characters = 0
        completion_tokens = 0
        completion_characters = 0
        cache_creation_input_tokens: Optional[int] = None
        cache_read_input_tokens: Optional[int] = None
        if completion_response is not None and (
            isinstance(completion_response, BaseModel)
            or isinstance(completion_response, dict)
        ):  # tts returns a custom class

            usage_obj: Optional[Union[dict, litellm.Usage]] = completion_response.get(  # type: ignore
                "usage", {}
            )
            if isinstance(usage_obj, BaseModel) and not isinstance(
                usage_obj, litellm.Usage
            ):
                setattr(
                    completion_response,
                    "usage",
                    litellm.Usage(**usage_obj.model_dump()),
                )
            if usage_obj is None:
                _usage = {}
            elif isinstance(usage_obj, BaseModel):
                _usage = usage_obj.model_dump()
            else:
                _usage = usage_obj
            # get input/output tokens from completion_response
            prompt_tokens = _usage.get("prompt_tokens", 0)
            completion_tokens = _usage.get("completion_tokens", 0)
            cache_creation_input_tokens = _usage.get("cache_creation_input_tokens", 0)
            cache_read_input_tokens = _usage.get("cache_read_input_tokens", 0)

            total_time = getattr(completion_response, "_response_ms", 0)
            verbose_logger.debug(
                f"completion_response response ms: {getattr(completion_response, '_response_ms', None)} "
            )
            model = _select_model_name_for_cost_calc(
                model=model, completion_response=completion_response
            )
            hidden_params = getattr(completion_response, "_hidden_params", None)
            if hidden_params is not None:
                custom_llm_provider = hidden_params.get(
                    "custom_llm_provider", custom_llm_provider or None
                )
                region_name = hidden_params.get("region_name", region_name)
                size = hidden_params.get("optional_params", {}).get(
                    "size", "1024-x-1024"
                )  # openai default
                quality = hidden_params.get("optional_params", {}).get(
                    "quality", "standard"
                )  # openai default
                n = hidden_params.get("optional_params", {}).get(
                    "n", 1
                )  # openai default
        else:
            if model is None:
                raise ValueError(
                    f"Model is None and does not exist in passed completion_response. Passed completion_response={completion_response}, model={model}"
                )
            if len(messages) > 0:
                prompt_tokens = token_counter(model=model, messages=messages)
            elif len(prompt) > 0:
                prompt_tokens = token_counter(model=model, text=prompt)
            completion_tokens = token_counter(model=model, text=completion)
        if model is None:
            raise ValueError(
                f"Model is None and does not exist in passed completion_response. Passed completion_response={completion_response}, model={model}"
            )

        if custom_llm_provider is None:
            try:
                _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model)
            except Exception as e:
                verbose_logger.error(
                    "litellm.cost_calculator.py::completion_cost() - Error inferring custom_llm_provider - {}".format(
                        str(e)
                    )
                )
        if (
            call_type == CallTypes.image_generation.value
            or call_type == CallTypes.aimage_generation.value
            or call_type == PassthroughCallTypes.passthrough_image_generation.value
        ):
            ### IMAGE GENERATION COST CALCULATION ###
            if custom_llm_provider == "vertex_ai":
                # https://cloud.google.com/vertex-ai/generative-ai/pricing
                # Vertex Charges Flat $0.20 per image
                return 0.020

            if size is None:
                size = "1024-x-1024"  # openai default
            # fix size to match naming convention
            if "x" in size and "-x-" not in size:
                size = size.replace("x", "-x-")
            image_gen_model_name = f"{size}/{model}"
            image_gen_model_name_with_quality = image_gen_model_name
            if quality is not None:
                image_gen_model_name_with_quality = f"{quality}/{image_gen_model_name}"
            size = size.split("-x-")
            height = int(size[0])  # if it's 1024-x-1024 vs. 1024x1024
            width = int(size[1])
            verbose_logger.debug(f"image_gen_model_name: {image_gen_model_name}")
            verbose_logger.debug(
                f"image_gen_model_name_with_quality: {image_gen_model_name_with_quality}"
            )
            if image_gen_model_name in litellm.model_cost:
                return (
                    litellm.model_cost[image_gen_model_name]["input_cost_per_pixel"]
                    * height
                    * width
                    * n
                )
            elif image_gen_model_name_with_quality in litellm.model_cost:
                return (
                    litellm.model_cost[image_gen_model_name_with_quality][
                        "input_cost_per_pixel"
                    ]
                    * height
                    * width
                    * n
                )
            else:
                raise Exception(
                    f"Model={image_gen_model_name} not found in completion cost model map"
                )
        elif (
            call_type == CallTypes.speech.value or call_type == CallTypes.aspeech.value
        ):
            prompt_characters = litellm.utils._count_characters(text=prompt)
        elif (
            call_type == CallTypes.rerank.value or call_type == CallTypes.arerank.value
        ):
            if completion_response is not None and isinstance(
                completion_response, RerankResponse
            ):
                meta_obj = completion_response.meta
                billed_units = meta_obj.get("billed_units", {}) or {}
                search_units = (
                    billed_units.get("search_units") or 1
                )  # cohere charges per request by default.
                completion_tokens = search_units
        # Calculate cost based on prompt_tokens, completion_tokens
        if (
            "togethercomputer" in model
            or "together_ai" in model
            or custom_llm_provider == "together_ai"
        ):
            # together ai prices based on size of llm
            # get_model_params_and_category takes a model name and returns the category of LLM size it is in model_prices_and_context_window.json

            model = get_model_params_and_category(model, call_type=CallTypes(call_type))

        # replicate llms are calculate based on time for request running
        # see https://replicate.com/pricing
        elif (
            model in litellm.replicate_models or "replicate" in model
        ) and model not in litellm.model_cost:
            # for unmapped replicate model, default to replicate's time tracking logic
            return get_replicate_completion_pricing(completion_response, total_time)  # type: ignore

        if model is None:
            raise ValueError(
                f"Model is None and does not exist in passed completion_response. Passed completion_response={completion_response}, model={model}"
            )

        if custom_llm_provider is not None and custom_llm_provider == "vertex_ai":
            # Calculate the prompt characters + response characters
            if len(messages) > 0:
                prompt_string = litellm.utils.get_formatted_prompt(
                    data={"messages": messages}, call_type="completion"
                )
            else:
                prompt_string = ""

            prompt_characters = litellm.utils._count_characters(text=prompt_string)
            if completion_response is not None and isinstance(
                completion_response, ModelResponse
            ):
                completion_string = litellm.utils.get_response_string(
                    response_obj=completion_response
                )
                completion_characters = litellm.utils._count_characters(
                    text=completion_string
                )

        (
            prompt_tokens_cost_usd_dollar,
            completion_tokens_cost_usd_dollar,
        ) = cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            custom_llm_provider=custom_llm_provider,
            response_time_ms=total_time,
            region_name=region_name,
            custom_cost_per_second=custom_cost_per_second,
            custom_cost_per_token=custom_cost_per_token,
            prompt_characters=prompt_characters,
            completion_characters=completion_characters,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            call_type=call_type,
        )
        _final_cost = prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar

        return _final_cost
    except Exception as e:
        raise e


def response_cost_calculator(
    response_object: Union[
        ModelResponse,
        EmbeddingResponse,
        ImageResponse,
        TranscriptionResponse,
        TextCompletionResponse,
        HttpxBinaryResponseContent,
        RerankResponse,
    ],
    model: str,
    custom_llm_provider: Optional[str],
    call_type: Literal[
        "embedding",
        "aembedding",
        "completion",
        "acompletion",
        "atext_completion",
        "text_completion",
        "image_generation",
        "aimage_generation",
        "moderation",
        "amoderation",
        "atranscription",
        "transcription",
        "aspeech",
        "speech",
        "rerank",
        "arerank",
    ],
    optional_params: dict,
    cache_hit: Optional[bool] = None,
    base_model: Optional[str] = None,
    custom_pricing: Optional[bool] = None,
) -> Optional[float]:
    """
    Returns
    - float or None: cost of response
    """
    try:
        response_cost: float = 0.0
        if cache_hit is not None and cache_hit is True:
            response_cost = 0.0
        else:
            if isinstance(response_object, BaseModel):
                response_object._hidden_params["optional_params"] = optional_params
            if isinstance(response_object, ImageResponse):
                response_cost = completion_cost(
                    completion_response=response_object,
                    model=model,
                    call_type=call_type,
                    custom_llm_provider=custom_llm_provider,
                )
            else:
                if custom_pricing is True:  # override defaults if custom pricing is set
                    base_model = model
                # base_model defaults to None if not set on model_info
                response_cost = completion_cost(
                    completion_response=response_object,
                    call_type=call_type,
                    model=base_model,
                    custom_llm_provider=custom_llm_provider,
                )
        return response_cost
    except Exception as e:
        raise e


def rerank_cost(
    model: str,
    custom_llm_provider: Optional[str],
) -> Tuple[float, float]:
    """
    Returns
    - float or None: cost of response OR none if error.
    """
    default_num_queries = 1
    _, custom_llm_provider, _, _ = litellm.get_llm_provider(
        model=model, custom_llm_provider=custom_llm_provider
    )

    try:
        if custom_llm_provider == "cohere":
            return cohere_rerank_cost_per_query(
                model=model, num_queries=default_num_queries
            )
        elif custom_llm_provider == "azure_ai":
            return azure_ai_rerank_cost_per_query(
                model=model, num_queries=default_num_queries
            )
        raise ValueError(
            f"invalid custom_llm_provider for rerank model: {model}, custom_llm_provider: {custom_llm_provider}"
        )
    except Exception as e:
        raise e
