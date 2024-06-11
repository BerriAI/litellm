# What is this?
## File for 'response_cost' calculation in Logging
from typing import Optional, Union, Literal, List
import litellm._logging
from litellm.utils import (
    ModelResponse,
    EmbeddingResponse,
    ImageResponse,
    TranscriptionResponse,
    TextCompletionResponse,
    CallTypes,
    cost_per_token,
    print_verbose,
    CostPerToken,
    token_counter,
)
import litellm
from litellm import verbose_logger


# Extract the number of billion parameters from the model name
# only used for together_computer LLMs
def get_model_params_and_category(model_name) -> str:
    """
    Helper function for calculating together ai pricing.

    Returns
    - str - model pricing category if mapped else received model name
    """
    import re

    model_name = model_name.lower()
    re_params_match = re.search(
        r"(\d+b)", model_name
    )  # catch all decimals like 3b, 70b, etc
    category = None
    if re_params_match is not None:
        params_match = str(re_params_match.group(1))
        params_match = params_match.replace("b", "")
        if params_match is not None:
            params_billion = float(params_match)
        else:
            return model_name
        # Determine the category based on the number of parameters
        if params_billion <= 4.0:
            category = "together-ai-up-to-4b"
        elif params_billion <= 8.0:
            category = "together-ai-4.1b-8b"
        elif params_billion <= 21.0:
            category = "together-ai-8.1b-21b"
        elif params_billion <= 41.0:
            category = "together-ai-21.1b-41b"
        elif params_billion <= 80.0:
            category = "together-ai-41.1b-80b"
        elif params_billion <= 110.0:
            category = "together-ai-81.1b-110b"
        if category is not None:
            return category

    return model_name


def get_replicate_completion_pricing(completion_response=None, total_time=0.0):
    # see https://replicate.com/pricing
    # for all litellm currently supported LLMs, almost all requests go to a100_80gb
    a100_80gb_price_per_second_public = (
        0.001400  # assume all calls sent to A100 80GB for now
    )
    if total_time == 0.0:  # total time is in ms
        start_time = completion_response["created"]
        end_time = getattr(completion_response, "ended", time.time())
        total_time = end_time - start_time

    return a100_80gb_price_per_second_public * total_time / 1000


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
        completion_tokens = 0
        custom_llm_provider = None
        if completion_response is not None:
            # get input/output tokens from completion_response
            prompt_tokens = completion_response.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = completion_response.get("usage", {}).get(
                "completion_tokens", 0
            )
            total_time = completion_response.get("_response_ms", 0)
            verbose_logger.debug(
                f"completion_response response ms: {completion_response.get('_response_ms')} "
            )
            model = model or completion_response.get(
                "model", None
            )  # check if user passed an override for model, if it's none check completion_response['model']
            if hasattr(completion_response, "_hidden_params"):
                if (
                    completion_response._hidden_params.get("model", None) is not None
                    and len(completion_response._hidden_params["model"]) > 0
                ):
                    model = completion_response._hidden_params.get("model", model)
                custom_llm_provider = completion_response._hidden_params.get(
                    "custom_llm_provider", ""
                )
                region_name = completion_response._hidden_params.get(
                    "region_name", region_name
                )
                size = completion_response._hidden_params.get(
                    "optional_params", {}
                ).get(
                    "size", "1024-x-1024"
                )  # openai default
                quality = completion_response._hidden_params.get(
                    "optional_params", {}
                ).get(
                    "quality", "standard"
                )  # openai default
                n = completion_response._hidden_params.get("optional_params", {}).get(
                    "n", 1
                )  # openai default
        else:
            if len(messages) > 0:
                prompt_tokens = token_counter(model=model, messages=messages)
            elif len(prompt) > 0:
                prompt_tokens = token_counter(model=model, text=prompt)
            completion_tokens = token_counter(model=model, text=completion)
        if model is None:
            raise ValueError(
                f"Model is None and does not exist in passed completion_response. Passed completion_response={completion_response}, model={model}"
            )

        if (
            call_type == CallTypes.image_generation.value
            or call_type == CallTypes.aimage_generation.value
        ):
            ### IMAGE GENERATION COST CALCULATION ###
            if custom_llm_provider == "vertex_ai":
                # https://cloud.google.com/vertex-ai/generative-ai/pricing
                # Vertex Charges Flat $0.20 per image
                return 0.020

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
        # Calculate cost based on prompt_tokens, completion_tokens
        if (
            "togethercomputer" in model
            or "together_ai" in model
            or custom_llm_provider == "together_ai"
        ):
            # together ai prices based on size of llm
            # get_model_params_and_category takes a model name and returns the category of LLM size it is in model_prices_and_context_window.json
            model = get_model_params_and_category(model)
        # replicate llms are calculate based on time for request running
        # see https://replicate.com/pricing
        elif (
            model in litellm.replicate_models or "replicate" in model
        ) and model not in litellm.model_cost:
            # for unmapped replicate model, default to replicate's time tracking logic
            return get_replicate_completion_pricing(completion_response, total_time)

        if model is None:
            raise ValueError(
                f"Model is None and does not exist in passed completion_response. Passed completion_response={completion_response}, model={model}"
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
        )
        _final_cost = prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar
        print_verbose(
            f"final cost: {_final_cost}; prompt_tokens_cost_usd_dollar: {prompt_tokens_cost_usd_dollar}; completion_tokens_cost_usd_dollar: {completion_tokens_cost_usd_dollar}"
        )
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
    ],
    model: str,
    custom_llm_provider: str,
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
    ],
    optional_params: dict,
    cache_hit: Optional[bool] = None,
    base_model: Optional[str] = None,
    custom_pricing: Optional[bool] = None,
) -> Optional[float]:
    try:
        response_cost: float = 0.0
        if cache_hit is not None and cache_hit is True:
            response_cost = 0.0
        else:
            response_object._hidden_params["optional_params"] = optional_params
            if isinstance(response_object, ImageResponse):
                response_cost = completion_cost(
                    completion_response=response_object,
                    model=model,
                    call_type=call_type,
                    custom_llm_provider=custom_llm_provider,
                )
            else:
                if (
                    model in litellm.model_cost
                    and custom_pricing is not None
                    and custom_llm_provider is True
                ):  # override defaults if custom pricing is set
                    base_model = model
                # base_model defaults to None if not set on model_info
                response_cost = completion_cost(
                    completion_response=response_object,
                    call_type=call_type,
                    model=base_model,
                    custom_llm_provider=custom_llm_provider,
                )
        return response_cost
    except litellm.NotFoundError as e:
        print_verbose(
            f"Model={model} for LLM Provider={custom_llm_provider} not found in completion cost map."
        )
        return None
