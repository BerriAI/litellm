"""Azure OpenAI parameter gates: api-version legality and deployment families.

v1's azure chain is the openai passthrough plus an api-version-aware
``AzureOpenAIConfig.map_openai_params`` (llms/azure/chat/gpt_transformation.py
:153-248): ``tool_choice`` needs 2023-12-01-preview or later and
``tool_choice='required'`` needs >= 2024-06; ``response_format`` support is
decided by model (gpt-3.5/gpt-35 excluded) AND api-version, with unsupported
combinations taking the synthetic json-tool strategy (``json_mode=True``).
Every gated shape returns a fallback reason here: the json-tool strategy and
the drop/raise interplay stay v1's, never a silent divergence. Family
detection runs on ``base_model or model`` exactly like v1 (azure.py:245-247,
utils.py:4831-4881), with azure's own substring predicates, which differ from
openai's prefix checks.
"""

from __future__ import annotations

import re

from expression import Option

from ...deps import TranslationDeps
from ...ir import ChatRequest, ToolChoice

# Mirrors litellm/types/llms/azure.py:1-2 (not importable here: providers may
# only import litellm.constants); the azure response_format differential rows
# pin these values.
_RESPONSE_FORMAT_SUPPORTED_YEAR = 2024
_RESPONSE_FORMAT_SUPPORTED_MONTH = 8


def detection_model(model: str, deps: TranslationDeps) -> str:
    # v1 is truthiness, not None-ness (azure.py:246 `base_model or model`):
    # base_model: "" in a config must fall to the deployment name, or the
    # family/response_format gates fail OPEN (verifier-azure S1).
    return deps.base_model or model


def unsupported_model_family(model: str, deps: TranslationDeps) -> str | None:
    detected = detection_model(model, deps)
    if (
        "o1" in detected
        or "o3" in detected
        or "o4" in detected
        or "o_series/" in detected
    ):
        return (
            f"azure o-series model {detected}: v1's AzureOpenAIO1Config "
            "owns its param rewrites (o_series_transformation.py:107-108)"
        )
    if (
        "gpt-5" in detected and not detected.split("/")[-1].startswith("gpt-5-chat")
    ) or "gpt5_series" in detected:
        return (
            f"azure gpt-5 model {detected}: v1's AzureOpenAIGPT5Config "
            "owns its param rewrites (gpt_5_transformation.py:37-59)"
        )
    return None


_API_VERSION_UNWIRED = (
    "azure api_version is not wired into TranslationDeps; v1 always resolves "
    "a string through the default chain (utils.py:4865-4875), so None is a "
    "seam wiring bug, not v1's unparseable-string passthrough"
)


def _api_version_parts(api_version: str) -> tuple[str, str, str] | None:
    """v1 splits the api_version on ``-`` and treats fewer than three
    segments as unparseable (every gate then passes through)."""
    parts = api_version.split("-")
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def unsupported_tool_choice(request: ChatRequest, deps: TranslationDeps) -> str | None:
    match request.tool_choice:
        case Option(tag="some", some=choice):
            pass
        case _:
            return None
    if deps.api_version is None:
        return _API_VERSION_UNWIRED
    parts = _api_version_parts(deps.api_version)
    if parts is None:
        return None
    year, month, day = parts
    # v1 compares the segments as STRINGS (az gpt_transformation.py:188-196);
    # mirrored verbatim so zero-padding behaves identically.
    if (
        year < "2023"
        or (year == "2023" and month < "12")
        or (year == "2023" and month == "12" and day < "01")
    ):
        return (
            f"tool_choice needs api_version 2023-12-01-preview or later, got "
            f"{deps.api_version}; v1 drops or raises (az gpt_transformation.py:181-205)"
        )
    if _is_required(choice) and year == "2024" and month <= "05":
        return (
            f"tool_choice='required' is unsupported for api_version "
            f"{deps.api_version}; v1 drops or raises (az gpt_transformation.py:206-217)"
        )
    return None


def _is_required(choice: ToolChoice) -> bool:
    return choice.tag == "required"


def _response_format_supported_model(detected: str) -> bool:
    """v1 ``_is_response_format_supported_model``: digit-dash-digit runs
    normalize to dots (the same regex), then gpt-3.5 (normalized) / gpt-35
    (raw) are out."""
    normalized = re.sub(r"(\d)-(\d)", r"\1.\2", detected)
    return not ("gpt-3.5" in normalized or "gpt-35" in detected)


def unsupported_response_format(
    request: ChatRequest, deps: TranslationDeps
) -> str | None:
    if request.response_format.is_none():
        return None
    detected = detection_model(request.model, deps)
    if not _response_format_supported_model(detected):
        return (
            f"response_format on {detected} takes v1's synthetic json-tool "
            "strategy (json_mode), unported (az gpt_transformation.py:220-242)"
        )
    if deps.api_version is None:
        return _API_VERSION_UNWIRED
    parts = _api_version_parts(deps.api_version)
    if parts is None:
        return None
    try:
        year, month = int(parts[0]), int(parts[1])
    except ValueError:
        return (
            f"api_version {deps.api_version!r} has non-numeric segments; "
            "v1's response_format gate raises on it"
        )
    supported = year > _RESPONSE_FORMAT_SUPPORTED_YEAR or (
        year == _RESPONSE_FORMAT_SUPPORTED_YEAR
        and month >= _RESPONSE_FORMAT_SUPPORTED_MONTH
    )
    if not supported:
        return (
            f"response_format needs api_version >= 2024-08, got "
            f"{deps.api_version}; v1 takes the json-tool strategy "
            "(az gpt_transformation.py:131-151)"
        )
    return None
