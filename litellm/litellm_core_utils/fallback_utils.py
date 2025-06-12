import uuid
from copy import deepcopy
from typing import Optional

import litellm
from litellm._logging import verbose_logger

from litellm.types.utils import FileTypes

from .asyncify import run_async_function


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
    nested_kwargs = kwargs.pop("kwargs", {})
    original_model = kwargs["model"]
    model = original_model
    fallbacks = [original_model] + nested_kwargs.pop("fallbacks", [])
    kwargs.pop("acompletion", None)  # Remove to prevent keyword conflicts
    litellm_call_id = str(uuid.uuid4())
    base_kwargs = {**kwargs, **nested_kwargs, "litellm_call_id": litellm_call_id}

    # fields to remove
    base_kwargs.pop("model", None)  # Remove model as it will be set per fallback
    litellm_logging_obj = base_kwargs.pop("litellm_logging_obj", None)

    # Try each fallback model
    most_recent_exception_str: Optional[str] = None
    for fallback in fallbacks:
        try:
            completion_kwargs = deepcopy(base_kwargs)
            # Handle dictionary fallback configurations
            if isinstance(fallback, dict):
                model = fallback.pop("model", original_model)
                completion_kwargs.update(fallback)
            else:
                model = fallback

            response = await litellm.acompletion(
                **completion_kwargs,
                model=model,
                litellm_logging_obj=litellm_logging_obj,
            )

            if response is not None:
                return response

        except Exception as e:
            verbose_logger.exception(
                f"Fallback attempt failed for model {model}: {str(e)}"
            )
            most_recent_exception_str = str(e)
            continue

    raise Exception(
        f"{most_recent_exception_str}. All fallback attempts failed. Enable verbose logging with `litellm.set_verbose=True` for details."
    )


def completion_with_fallbacks(**kwargs):
    return run_async_function(async_function=async_completion_with_fallbacks, **kwargs)


# ------------------------------
# Audio Transcription Fallbacks
# ------------------------------


async def async_transcription_with_fallbacks(model, file, **kwargs):
    """Asynchronously attempt transcription with fallback models if the primary fails.

    Parameters
    ----------
    model : str
        Primary model name.
    file : FileTypes
        Audio file like object passed to the transcription endpoint.
    **kwargs : dict
        Additional parameters. A nested ``kwargs`` dict can include the key
        ``fallbacks`` which should be a list of models / dict configs, exactly
        like `completion_with_fallbacks`.
    """

    # Extract nested kwargs coming from main.litellm.atranscription wrapper
    nested_kwargs = kwargs.pop("kwargs", {})
    fallbacks = [model] + nested_kwargs.pop("fallbacks", [])

    # Remove potential recursion flags
    nested_kwargs.pop("fallbacks", None)
    kwargs.pop("fallbacks", None)

    # Remove indicators that can conflict with recursive calls
    nested_kwargs.pop("atranscription", None)
    kwargs.pop("atranscription", None)

    # Generate a new call-id so downstream logging treats each attempt uniquely
    litellm_call_id = str(uuid.uuid4())

    base_kwargs = {
        **kwargs,
        **nested_kwargs,
        "litellm_call_id": litellm_call_id,
    }

    # We will pass model + file explicitly for each attempt, so remove them from kwargs
    base_kwargs.pop("model", None)
    base_kwargs.pop("file", None)

    litellm_logging_obj = base_kwargs.pop("litellm_logging_obj", None)

    most_recent_exception_str: Optional[str] = None

    for fb in fallbacks:
        try:
            attempt_kwargs = deepcopy(base_kwargs)

            if isinstance(fb, dict):
                attempt_model = fb.pop("model", model)
                attempt_kwargs.update(fb)
            else:
                attempt_model = fb

            # Ensure fallbacks param is not forwarded further to avoid infinite recursion
            attempt_kwargs.pop("fallbacks", None)

            response = await litellm.atranscription(
                model=attempt_model,
                file=file,
                **attempt_kwargs,
            )

            if response is not None:
                return response

        except Exception as e:
            verbose_logger.exception(
                f"Transcription fallback attempt failed for model {fb}: {str(e)}"
            )
            most_recent_exception_str = str(e)

    raise Exception(
        f"{most_recent_exception_str}. All transcription fallback attempts failed. Enable verbose logging with `litellm.set_verbose=True` for details."
    )


def transcription_with_fallbacks(**kwargs):
    """Synchronous wrapper around :pyfunc:`async_transcription_with_fallbacks`."""

    return run_async_function(async_function=async_transcription_with_fallbacks, **kwargs)
