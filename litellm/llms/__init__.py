import importlib
import os
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import attribute_or_none
from litellm.types.utils import CallTypes

from . import *

if TYPE_CHECKING:
    from collections.abc import Callable

    from litellm.llms.base_llm.guardrail_translation.base_translation import (
        BaseTranslation,
    )
    from litellm.types.utils import ModelInfo, Usage


def get_cost_for_google_maps_grounding_request(
    custom_llm_provider: str, usage: "Usage", model_info: "ModelInfo"
) -> float | None:
    """
    Get the cost of Grounding with Google Maps for a given model. Only Gemini models on the
    Gemini API and Vertex AI can populate the Maps grounding counter, so every other provider
    returns None.
    """
    from litellm.llms.gemini.cost_calculator import maps_grounding_cost

    return maps_grounding_cost(usage, model_info, custom_llm_provider)


def get_cost_for_web_search_request(
    custom_llm_provider: str,
    usage: "Usage",
    model_info: "ModelInfo",
    response_model: str | None = None,
    lookup_model_info: "Callable[[str, str | None], ModelInfo | None] | None" = None,
) -> float | None:
    """
    Get the cost for a web search request for a given model, priced in Rust by the provider's own calculator.
    ``None`` for a provider without one.

    Args:
        custom_llm_provider: The custom LLM provider.
        usage: The usage object.
        model_info: The model info.
        response_model: The model the provider response names.
        lookup_model_info: Looks up a model's cost map entry, returning None when it is not mapped.
    """
    from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import price_web_search_dispatch

    return price_web_search_dispatch(custom_llm_provider, usage, model_info, response_model, lookup_model_info)


def discover_guardrail_translation_mappings() -> dict[CallTypes, type["BaseTranslation"]]:
    """
    Discover guardrail translation mappings by scanning the llms directory structure.

    Scans for modules with guardrail_translation_mappings dictionaries and aggregates them.

    Returns:
        Dict[CallTypes, Type[BaseTranslation]]: A dictionary mapping call types to their translation handler classes
    """
    discovered_mappings: Final[dict[CallTypes, type[BaseTranslation]]] = {}

    try:
        # Get the path to the llms directory
        current_dir: Final = os.path.dirname(__file__)
        llms_dir: Final = current_dir

        if not os.path.exists(llms_dir):
            verbose_logger.debug("llms directory not found")
            return discovered_mappings

        # Recursively scan for guardrail_translation directories
        for root, dirs, files in os.walk(llms_dir):
            # Skip __pycache__ and base_llm directories
            dirs[:] = [d for d in dirs if not d.startswith("__") and d != "base_llm"]

            # Check if this is a guardrail_translation directory with __init__.py
            if os.path.basename(root) == "guardrail_translation" and "__init__.py" in files:
                # Build the module path relative to litellm
                rel_path = os.path.relpath(root, os.path.dirname(llms_dir))
                module_path = "litellm." + rel_path.replace(os.sep, ".")

                try:
                    # Import the module
                    verbose_logger.debug("Discovering guardrail translations in: %s", module_path)

                    module = importlib.import_module(module_path)

                    # Check for guardrail_translation_mappings dictionary
                    if hasattr(module, "guardrail_translation_mappings"):
                        mappings = attribute_or_none(module, "guardrail_translation_mappings")
                        if isinstance(mappings, dict):
                            discovered_mappings.update(mappings)
                            verbose_logger.debug(
                                "Found guardrail_translation_mappings in %s: %s", module_path, list(mappings.keys())
                            )

                except ImportError as e:
                    verbose_logger.error("Could not import %s: %s", module_path, e)
                    continue
                except Exception as e:
                    verbose_logger.error("Error processing %s: %s", module_path, e)
                    continue

        try:
            from litellm.proxy.mcp_server.guardrail_translation import (
                guardrail_translation_mappings as mcp_guardrail_translation_mappings,
            )

            discovered_mappings.update(mcp_guardrail_translation_mappings)
            verbose_logger.debug(
                "Loaded MCP guardrail translation mappings: %s",
                list(mcp_guardrail_translation_mappings.keys()),
            )
        except ImportError:
            verbose_logger.debug("MCP guardrail translation mappings not available; skipping")

        verbose_logger.debug(
            "Discovered %s guardrail translation mappings: %s",
            len(discovered_mappings),
            list(discovered_mappings.keys()),
        )

    except Exception as e:
        verbose_logger.error("Error discovering guardrail translation mappings: %s", e)

    return discovered_mappings


# Cache the discovered mappings
endpoint_guardrail_translation_mappings: dict[CallTypes, type["BaseTranslation"]] | None = None


def load_guardrail_translation_mappings():
    global endpoint_guardrail_translation_mappings
    if endpoint_guardrail_translation_mappings is None:
        endpoint_guardrail_translation_mappings = discover_guardrail_translation_mappings()
    return endpoint_guardrail_translation_mappings


def get_guardrail_translation_mapping(call_type: CallTypes) -> type["BaseTranslation"]:
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
        endpoint_guardrail_translation_mappings = discover_guardrail_translation_mappings()

    # Get the translation handler class for the call type
    if call_type not in endpoint_guardrail_translation_mappings:
        raise ValueError(
            f"No guardrail translation mapping found for call_type: {call_type}. "
            f"Available mappings: {list(endpoint_guardrail_translation_mappings.keys())}"
        )

    # Return the handler class directly
    return endpoint_guardrail_translation_mappings[call_type]
