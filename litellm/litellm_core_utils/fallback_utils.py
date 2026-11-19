from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.core_helpers import (
    filter_internal_params,
    safe_deep_copy,
)
from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
)
from litellm.types.utils import CustomPricingLiteLLMParams

from .asyncify import run_async_function

_MODEL_BOUND_PARAMS: Final = frozenset(CustomPricingLiteLLMParams.model_fields) | {"base_model"}
_PER_ATTEMPT_KWARGS: Final = frozenset({"model", "litellm_logging_obj"})
_REQUEST_KWARGS: Final = TypeAdapter(Mapping[str, object])


def _bills_as_a2a_agent(logging_obj: object) -> bool:
    from litellm.litellm_core_utils.litellm_logging import Logging

    return isinstance(logging_obj, Logging) and logging_obj.a2a_agent_billing is not None


def _inherited_kwargs(
    base_kwargs: Mapping[str, object], original_model: object, model: object, logging_obj: object
) -> Mapping[str, object]:
    if model == original_model or not _bills_as_a2a_agent(logging_obj):
        return base_kwargs
    return {key: value for key, value in base_kwargs.items() if key not in _MODEL_BOUND_PARAMS}


async def _fallback_response(
    base_kwargs: Mapping[str, object],
    original_model: object,
    model: object,
    overrides: Mapping[str, object],
    logging_obj: object,
) -> object:
    from litellm.litellm_core_utils.litellm_logging import Logging

    inherited: Final = safe_deep_copy(dict(_inherited_kwargs(base_kwargs, original_model, model, logging_obj)))
    completion_kwargs: Final = filter_internal_params(_REQUEST_KWARGS.validate_python({**inherited, **overrides}))
    attempt_kwargs: Final = {**completion_kwargs, "model": model, "litellm_logging_obj": logging_obj}
    if not isinstance(logging_obj, Logging):
        return await litellm.acompletion(**attempt_kwargs)
    if model == original_model:
        logging_obj.resume_billing_as_a2a_agent()
        return await litellm.acompletion(**attempt_kwargs)
    logging_obj.stop_billing_as_a2a_agent()
    response: Final = await litellm.acompletion(**attempt_kwargs)
    logging_obj.stop_attributing_to_a2a_agent()
    return response


async def async_completion_with_fallbacks(**kwargs):
    """
    Asynchronously attempts completion with fallback models if the primary model fails.

    Args:
        **kwargs: Keyword arguments for completion, including:
            - model (str): Primary model to use
            - fallbacks (List[Union[str, dict]]): List of fallback models/configs
            - Other completion parameters

    Returns:
        ModelResponse: The completion response from the first successful model

    Raises:
        Exception: If all models fail and no response is generated
    """
    # Extract and prepare parameters
    nested_kwargs: Final = kwargs.pop("kwargs", {})
    original_model: Final = kwargs["model"]
    model = original_model
    fallbacks: Final = [original_model] + nested_kwargs.pop("fallbacks", [])
    kwargs.pop("acompletion", None)  # Remove to prevent keyword conflicts
    litellm_call_id: Final = str(uuid.uuid4())
    merged_kwargs: Final = _REQUEST_KWARGS.validate_python({**kwargs, **nested_kwargs})
    base_kwargs: Final = {
        **{key: value for key, value in merged_kwargs.items() if key not in _PER_ATTEMPT_KWARGS},
        "litellm_call_id": litellm_call_id,
    }
    litellm_logging_obj: Final = merged_kwargs.get("litellm_logging_obj")

    # Try each fallback model
    most_recent_exception_str: str | None = None
    for attempted_fallbacks, fallback in enumerate(fallbacks):
        try:
            # Handle dictionary fallback configurations
            fallback_config = safe_deep_copy(dict(fallback)) if isinstance(fallback, dict) else {}
            model = fallback_config.pop("model", original_model) if isinstance(fallback, dict) else fallback
            response = await _fallback_response(
                base_kwargs, original_model, model, fallback_config, litellm_logging_obj
            )

            if response is not None:
                return add_fallback_headers_to_response(
                    response=response,
                    attempted_fallbacks=attempted_fallbacks,
                )

        except Exception as e:
            verbose_logger.exception("Fallback attempt failed for model %s: %s", model, e)
            most_recent_exception_str = str(e)
            continue

    raise Exception(
        f"{most_recent_exception_str}. All fallback attempts failed. Enable verbose logging with `litellm.set_verbose=True` for details."
    )


def completion_with_fallbacks(**kwargs):
    return run_async_function(async_function=async_completion_with_fallbacks, **kwargs)
