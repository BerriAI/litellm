import importlib
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_logger
from litellm.types.utils import CallTypes

from . import *

if TYPE_CHECKING:
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
    if custom_llm_provider != "gemini" and not custom_llm_provider.startswith("vertex_ai"):
        return None
    from .gemini.cost_calculator import cost_per_google_maps_grounding_request

    return cost_per_google_maps_grounding_request(usage=usage, model_info=model_info)


def get_cost_for_web_search_request(custom_llm_provider: str, usage: "Usage", model_info: "ModelInfo") -> float | None:
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
        # Anthropic Claude models on Vertex AI populate server_tool_use.web_search_requests
        # (same as the direct Anthropic API), not prompt_tokens_details.web_search_requests
        # (which is the Gemini field). Route claude-* models to the Anthropic calculator.
        model_key: Final[str] = model_info.get("key", "") if model_info else ""
        if "claude" in model_key.lower():
            from .anthropic.cost_calculation import get_cost_for_anthropic_web_search

            verbose_logger.debug("vertex_ai/claude model detected — routing web search cost to Anthropic calculator")
            return get_cost_for_anthropic_web_search(model_info=model_info, usage=usage)

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
    elif custom_llm_provider == "groq":
        from .groq.cost_calculator import (
            cost_per_web_search_request as groq_cost_per_web_search_request,
        )

        return groq_cost_per_web_search_request(usage=usage, model_info=model_info)
    else:
        return None


_GUARDRAIL_TRANSLATION_PACKAGE: Final = "guardrail_translation"
_MCP_GUARDRAIL_TRANSLATION_MODULE: Final = "litellm.proxy._experimental.mcp_server.guardrail_translation"


@dataclass(frozen=True, slots=True)
class GuardrailTranslationDiscovery:
    """
    The outcome of one scan for guardrail translation handlers.

    unavailable_modules names the bundled packages that failed to import, which is what tells a complete
    result apart from one that is missing handlers and therefore must not be cached.
    """

    mappings: Mapping[CallTypes, type["BaseTranslation"]]
    unavailable_modules: tuple[str, ...]


def _bundled_guardrail_translation_modules() -> Iterator[str]:
    """Yield the import path of every guardrail_translation package shipped under litellm/llms."""
    llms_dir: Final = os.path.dirname(__file__)
    for root, dirs, files in os.walk(llms_dir):
        dirs[:] = [d for d in dirs if not d.startswith("__") and d != "base_llm"]
        if os.path.basename(root) == _GUARDRAIL_TRANSLATION_PACKAGE and "__init__.py" in files:
            yield "litellm." + os.path.relpath(root, os.path.dirname(llms_dir)).replace(os.sep, ".")


def _guardrail_translation_mappings_of(module_path: str) -> Mapping[CallTypes, type["BaseTranslation"]] | None:
    """Import one guardrail_translation package, returning None when it could not be imported at all."""
    try:
        module: Final = importlib.import_module(module_path)
    except Exception as e:
        verbose_logger.error("Could not import guardrail translations from %s: %s", module_path, e)
        return None
    mappings: Final = getattr(module, "guardrail_translation_mappings", None)
    if not isinstance(mappings, dict):
        return {}
    declared: Final[Mapping[CallTypes, type[BaseTranslation]]] = mappings
    return declared


def _optional_mcp_guardrail_translation_mappings() -> Mapping[CallTypes, type["BaseTranslation"]]:
    """MCP call types live outside litellm/llms and are absent from installs without the MCP server."""
    try:
        from litellm.proxy._experimental.mcp_server.guardrail_translation import (
            guardrail_translation_mappings as mcp_guardrail_translation_mappings,
        )
    except ImportError:
        verbose_logger.debug("%s not available; skipping", _MCP_GUARDRAIL_TRANSLATION_MODULE)
        return {}
    return mcp_guardrail_translation_mappings


def discover_guardrail_translations() -> GuardrailTranslationDiscovery:
    """
    Scan the llms tree, plus the optional MCP package, for guardrail translation handlers.

    Returns:
        GuardrailTranslationDiscovery: the handlers found, and the bundled packages that failed to import
    """
    bundled: Final = tuple(
        (module_path, _guardrail_translation_mappings_of(module_path))
        for module_path in _bundled_guardrail_translation_modules()
    )
    found: Final = (
        *(mappings for _, mappings in bundled if mappings is not None),
        _optional_mcp_guardrail_translation_mappings(),
    )
    return GuardrailTranslationDiscovery(
        mappings=MappingProxyType(
            {call_type: handler for mappings in found for call_type, handler in mappings.items()}
        ),
        unavailable_modules=tuple(module_path for module_path, mappings in bundled if mappings is None),
    )


def discover_guardrail_translation_mappings() -> dict[CallTypes, type["BaseTranslation"]]:
    """
    Discover guardrail translation mappings by scanning the llms directory structure.

    Returns:
        Dict[CallTypes, Type[BaseTranslation]]: A dictionary mapping call types to their translation handler classes
    """
    return dict(discover_guardrail_translations().mappings)


endpoint_guardrail_translation_mappings: dict[CallTypes, type["BaseTranslation"]] | None = None


def load_guardrail_translation_mappings() -> dict[CallTypes, type["BaseTranslation"]]:
    """
    Return the guardrail translation handlers, caching only a discovery that imported every bundled package.

    An incomplete scan is served but never cached: caching one would silently strip the missing call types
    off every guardrail for the rest of the process, so the next call retries the packages that failed.
    """
    global endpoint_guardrail_translation_mappings
    if endpoint_guardrail_translation_mappings is not None:
        return endpoint_guardrail_translation_mappings

    discovery: Final = discover_guardrail_translations()
    if discovery.unavailable_modules:
        verbose_logger.error(
            "Found only %s guardrail translation handlers because %s could not be imported. "
            "Not caching this result: guardrails for the missing call types cannot run until the import succeeds.",
            len(discovery.mappings),
            ", ".join(discovery.unavailable_modules),
        )
        return dict(discovery.mappings)

    endpoint_guardrail_translation_mappings = dict(discovery.mappings)
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
    mappings: Final = load_guardrail_translation_mappings()
    if call_type not in mappings:
        raise ValueError(
            f"No guardrail translation mapping found for call_type: {call_type}. "
            f"Available mappings: {list(mappings.keys())}"
        )
    return mappings[call_type]
