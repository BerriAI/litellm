# What is this?
## File for 'response_cost' calculation in Logging
from typing import Optional, Union, Literal
from litellm.utils import (
    ModelResponse,
    EmbeddingResponse,
    ImageResponse,
    TranscriptionResponse,
    TextCompletionResponse,
    CallTypes,
    completion_cost,
    print_verbose,
)
import litellm


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
        if cache_hit is not None and cache_hit == True:
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
                    and custom_llm_provider == True
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
