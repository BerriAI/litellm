import importlib
import os
from typing import TYPE_CHECKING, Dict, Optional, Type

from litellm._logging import verbose_logger
from litellm.types.utils import CallTypes

from . import *

if TYPE_CHECKING:
    from litellm.llms.base_llm.guardrail_translation.base_translation import (
        BaseTranslation,
    )
    from litellm.types.utils import ModelInfo, Usage


def get_cost_for_web_search_request(
    custom_llm_provider: str, usage: "Usage", model_info: "ModelInfo"
) -> Optional[float]:
    """
    Get the cost for a web search request for a given model.

    Args:
        custom_llm_provider: The custom LLM provider.
        usage: The usage object.
        model_info: The model info.
    """
    if custom_llm_provider == "gemini":
        from .gemini.cost_calculator import cost_per_web_search_request

        return cost_per_web_search_request(usage=usage, model_info=model_info)
    elif custom_llm_provider == "anthropic":
        from .anthropic.cost_calculation import get_cost_for_anthropic_web_search

        return get_cost_for_anthropic_web_search(model_info=model_info, usage=usage)
    elif custom_llm_provider.startswith("vertex_ai"):
        from .vertex_ai.gemini.cost_calculator import (
            cost_per_web_search_request as cost_per_web_search_request_vertex_ai,
        )

        return cost_per_web_search_request_vertex_ai(usage=usage, model_info=model_info)
    elif custom_llm_provider == "perplexity":
        # Perplexity handles search costs internally in its own cost calculator
        # Return 0.0 to indicate costs are already accounted for
        return 0.0
    elif custom_llm_provider == "xai":
        from .xai.cost_calculator import cost_per_web_search_request
        return cost_per_web_search_request(usage=usage, model_info=model_info)
    else:
        return None


def discover_guardrail_translation_mappings() -> (
    Dict[CallTypes, Type["BaseTranslation"]]
):
    """
    Discover guardrail translation mappings by scanning the llms directory structure.

    Scans for modules with guardrail_translation_mappings dictionaries and aggregates them.

    Returns:
        Dict[CallTypes, Type[BaseTranslation]]: A dictionary mapping call types to their translation handler classes
    """
    discovered_mappings: Dict[CallTypes, Type["BaseTranslation"]] = {}

    try:
        # Get the path to the llms directory
        current_dir = os.path.dirname(__file__)
        llms_dir = current_dir

        if not os.path.exists(llms_dir):
            verbose_logger.debug("llms directory not found")
            return discovered_mappings

        # Recursively scan for guardrail_translation directories
        for root, dirs, files in os.walk(llms_dir):
            # Skip __pycache__ and base_llm directories
            dirs[:] = [d for d in dirs if not d.startswith("__") and d != "base_llm"]

            # Check if this is a guardrail_translation directory with __init__.py
            if (
                os.path.basename(root) == "guardrail_translation"
                and "__init__.py" in files
            ):
                # Build the module path relative to litellm
                rel_path = os.path.relpath(root, os.path.dirname(llms_dir))
                module_path = "litellm." + rel_path.replace(os.sep, ".")

                try:
                    # Import the module
                    verbose_logger.debug(
                        f"Discovering guardrail translations in: {module_path}"
                    )

                    module = importlib.import_module(module_path)

                    # Check for guardrail_translation_mappings dictionary
                    if hasattr(module, "guardrail_translation_mappings"):
                        mappings = getattr(module, "guardrail_translation_mappings")
                        if isinstance(mappings, dict):
                            discovered_mappings.update(mappings)
                            verbose_logger.debug(
                                f"Found guardrail_translation_mappings in {module_path}: {list(mappings.keys())}"
                            )

                except ImportError as e:
                    verbose_logger.error(f"Could not import {module_path}: {e}")
                    continue
                except Exception as e:
                    verbose_logger.error(f"Error processing {module_path}: {e}")
                    continue

        verbose_logger.debug(
            f"Discovered {len(discovered_mappings)} guardrail translation mappings: {list(discovered_mappings.keys())}"
        )

    except Exception as e:
        verbose_logger.error(f"Error discovering guardrail translation mappings: {e}")

    return discovered_mappings


# Cache the discovered mappings
endpoint_guardrail_translation_mappings: Optional[
    Dict[CallTypes, Type["BaseTranslation"]]
] = None


def load_guardrail_translation_mappings():
    global endpoint_guardrail_translation_mappings
    if endpoint_guardrail_translation_mappings is None:
        endpoint_guardrail_translation_mappings = (
            discover_guardrail_translation_mappings()
        )
    return endpoint_guardrail_translation_mappings


def get_guardrail_translation_mapping(call_type: CallTypes) -> Type["BaseTranslation"]:
    """
    Get the guardrail translation handler for a given call type.

    Args:
        call_type: The type of call (e.g., completion, acompletion, anthropic_messages)

    Returns:
        The translation handler class for the given call type

    Raises:
        ValueError: If no translation mapping exists for the given call type
    """
    global endpoint_guardrail_translation_mappings

    # Lazy load the mappings on first access
    if endpoint_guardrail_translation_mappings is None:
        endpoint_guardrail_translation_mappings = (
            discover_guardrail_translation_mappings()
        )

    # Get the translation handler class for the call type
    if call_type not in endpoint_guardrail_translation_mappings:
        raise ValueError(
            f"No guardrail translation mapping found for call_type: {call_type}. "
            f"Available mappings: {list(endpoint_guardrail_translation_mappings.keys())}"
        )

    # Return the handler class directly
    return endpoint_guardrail_translation_mappings[call_type]
