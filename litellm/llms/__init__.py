import importlib
import importlib.util
import os
from collections.abc import Iterable, Iterator, Mapping
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
_NO_MAPPINGS: Final[Mapping[CallTypes, type["BaseTranslation"]]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class GuardrailTranslationDiscovery:
    """
    The outcome of one scan for guardrail translation handlers.

    unavailable maps each package that failed to import to the reason, which is what tells a complete
    result apart from one that is missing handlers and therefore has to be retried.
    """

    mappings: Mapping[CallTypes, type["BaseTranslation"]]
    unavailable: Mapping[str, str]


def _bundled_guardrail_translation_modules() -> Iterator[str]:
    """Yield the import path of every guardrail_translation package shipped under litellm/llms."""
    llms_dir: Final = os.path.dirname(__file__)
    for root, dirs, files in os.walk(llms_dir):
        dirs[:] = tuple(d for d in dirs if not d.startswith("__") and d != "base_llm")
        if os.path.basename(root) == _GUARDRAIL_TRANSLATION_PACKAGE and "__init__.py" in files:
            yield "litellm." + os.path.relpath(root, os.path.dirname(llms_dir)).replace(os.sep, ".")


@dataclass(frozen=True, slots=True)
class _UnavailablePackage:
    """
    Why one guardrail_translation package could not be imported.

    missing_dependency is set when the package asked for a module this install does not have at all, which is
    the one failure that says the package is absent rather than momentarily unimportable.
    """

    reason: str
    missing_dependency: bool


def _is_absent(module_name: str) -> bool:
    """Whether this install has no module of that name, as opposed to one that is present but failed to import."""
    root: Final = module_name.partition(".")[0]
    try:
        return importlib.util.find_spec(root) is None
    except (ImportError, ValueError):
        return False


def _import_guardrail_translations(
    module_path: str,
) -> Mapping[CallTypes, type["BaseTranslation"]] | _UnavailablePackage:
    """Import one guardrail_translation package, reporting why that failed instead of raising."""
    try:
        module: Final = importlib.import_module(module_path)
    except Exception as e:  # noqa: BLE001  # a package failing at import time for any reason is unavailable, not fatal
        return _UnavailablePackage(
            reason=f"{type(e).__name__}: {e}",
            missing_dependency=isinstance(e, ModuleNotFoundError) and _is_absent(e.name or ""),
        )
    mappings: Final = getattr(module, "guardrail_translation_mappings", None)
    if not isinstance(mappings, dict):
        return _NO_MAPPINGS
    declared: Final[Mapping[CallTypes, type[BaseTranslation]]] = mappings
    return declared


def _guardrail_translations_from(
    module_path: str,
) -> Mapping[CallTypes, type["BaseTranslation"]] | _UnavailablePackage:
    """
    Import one package's handlers, tolerating an install that does not ship the optional MCP server.

    litellm ships every package under llms, so a failure there is a gap to retry rather than a fact about the
    install. The MCP package instead arrives with the proxy extra, and a dependency this install does not have
    at all means it serves no MCP endpoints for a guardrail to scan, so there is nothing to retry or report. A
    dependency that is installed and still fails to import is a broken install, which is reported and retried.
    """
    result: Final = _import_guardrail_translations(module_path)
    if (
        module_path == _MCP_GUARDRAIL_TRANSLATION_MODULE
        and isinstance(result, _UnavailablePackage)
        and result.missing_dependency
    ):
        verbose_logger.debug("%s is not installed: %s", module_path, result.reason)
        return _NO_MAPPINGS
    return result


def _guardrail_translation_modules() -> Iterator[str]:
    """Yield every module that can declare guardrail translation handlers, the optional MCP one last."""
    yield from _bundled_guardrail_translation_modules()
    yield _MCP_GUARDRAIL_TRANSLATION_MODULE


def _discover(
    module_paths: Iterable[str], already_found: Mapping[CallTypes, type["BaseTranslation"]]
) -> GuardrailTranslationDiscovery:
    imported: Final = tuple((module_path, _guardrail_translations_from(module_path)) for module_path in module_paths)
    found: Final = (
        already_found,
        *(result for _, result in imported if not isinstance(result, _UnavailablePackage)),
    )
    return GuardrailTranslationDiscovery(
        mappings=MappingProxyType(
            {call_type: handler for mappings in found for call_type, handler in mappings.items()}
        ),
        unavailable=MappingProxyType(
            {module_path: result.reason for module_path, result in imported if isinstance(result, _UnavailablePackage)}
        ),
    )


def discover_guardrail_translations() -> GuardrailTranslationDiscovery:
    """
    Scan the llms tree, plus the optional MCP package, for guardrail translation handlers.

    Returns:
        GuardrailTranslationDiscovery: the handlers found, and the packages that failed to import
    """
    return _discover(_guardrail_translation_modules(), already_found=_NO_MAPPINGS)


def discover_guardrail_translation_mappings() -> Mapping[CallTypes, type["BaseTranslation"]]:
    """
    Discover guardrail translation mappings by scanning the llms directory structure.

    Returns:
        Mapping[CallTypes, Type[BaseTranslation]]: the call types that have a translation handler class
    """
    return discover_guardrail_translations().mappings


def _announce_discovery(previous: GuardrailTranslationDiscovery | None, current: GuardrailTranslationDiscovery) -> None:
    if previous is None and not current.unavailable:
        return
    if previous is None:
        verbose_logger.error(
            "Could not import guardrail translation handlers from %s; guardrails cannot run for their call types "
            "until the import succeeds, which every lookup retries. %s",
            ", ".join(current.unavailable),
            "; ".join(f"{module_path}: {reason}" for module_path, reason in current.unavailable.items()),
        )
        return
    recovered: Final = tuple(
        module_path for module_path in previous.unavailable if module_path not in current.unavailable
    )
    if not recovered:
        return
    verbose_logger.warning("Guardrail translation handlers from %s are available again.", ", ".join(recovered))


guardrail_translation_discovery: GuardrailTranslationDiscovery | None = None


def load_guardrail_translation_mappings() -> Mapping[CallTypes, type["BaseTranslation"]]:
    """
    Return the guardrail translation handlers, retrying any bundled package that could not be imported last time.

    Serving an incomplete scan as if it were complete would silently strip the missing call types off every
    guardrail for the rest of the process, so the packages that failed are imported again on each lookup, and
    only the part that succeeded is kept.
    """
    global guardrail_translation_discovery
    cached: Final = guardrail_translation_discovery
    if cached is not None and not cached.unavailable:
        return cached.mappings
    discovery: Final = (
        discover_guardrail_translations()
        if cached is None
        else _discover(cached.unavailable, already_found=cached.mappings)
    )
    _announce_discovery(previous=cached, current=discovery)
    guardrail_translation_discovery = discovery
    return discovery.mappings


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
