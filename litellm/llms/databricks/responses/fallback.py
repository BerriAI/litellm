"""
Databricks `responses()` surface fallback chain.

No single Databricks responses surface covers every model (verified live — see
``_local/litellm-unity-ai-gateway/probes``):

* native OpenAI Responses (``/ai-gateway/openai/v1/responses``) — GPT only
* Supervisor (``/ai-gateway/mlflow/v1/responses``) — allowlist (Claude + gpt-5 full + qwen35)
* Open Responses (``/serving-endpoints/open-responses``) — Gemini, gpt-oss, qwen3-next, …

So `responses(model="databricks/...")` tries the per-family ordered chain below
and, when every responses surface rejects the model (e.g. older Llama/Gemma), the
caller falls through to chat-completions emulation. The chain is consumed by the
error-driven retry in :func:`litellm.responses.main.responses`.
"""

from typing import TYPE_CHECKING, List

from litellm.llms.databricks.ai_gateway import ProviderFamily, detect_family

if TYPE_CHECKING:
    from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig

# Statuses that can indicate "this surface does not serve this model".
_FALLBACK_STATUSES = {400, 404, 500, 501}
# Substrings (lowercased) seen in Databricks "model not served here" errors.
_FALLBACK_MARKERS = (
    "invalid_parameter_value",
    "not supported",
    "unknown field",
    "endpoint_not_found",
    "internal_error",
    "resource_does_not_exist",
)


def databricks_responses_config_chain(model: str) -> List["BaseResponsesAPIConfig"]:
    """Ordered responses configs to try for ``model``. An empty tail (chain
    exhausted) means the caller should emulate via chat completions."""
    from litellm.llms.databricks.responses.transformation import (
        DatabricksOpenResponsesAPIConfig,
        DatabricksResponsesAPIConfig,
        DatabricksSupervisorResponsesAPIConfig,
    )

    family = detect_family(model)
    if family == ProviderFamily.OPENAI_RESPONSES:
        # gpt-N (non-oss): native first; gpt-5 full sizes also work on Supervisor.
        return [
            DatabricksResponsesAPIConfig(),
            DatabricksSupervisorResponsesAPIConfig(),
        ]
    if family == ProviderFamily.ANTHROPIC:
        # Claude: Supervisor (ai-gateway) primary, Open Responses fallback.
        return [
            DatabricksSupervisorResponsesAPIConfig(),
            DatabricksOpenResponsesAPIConfig(),
        ]
    if family == ProviderFamily.GEMINI:
        # Gemini: only Open Responses (Supervisor rejects it; no native responses).
        return [DatabricksOpenResponsesAPIConfig()]
    # OPENAI catch-all (gpt-oss, qwen, llama, gemma, …): Open Responses primary;
    # Supervisor catches the qwen35-style allowlisted models.
    return [
        DatabricksOpenResponsesAPIConfig(),
        DatabricksSupervisorResponsesAPIConfig(),
    ]


def is_surface_unavailable_error(exc: Exception) -> bool:
    """True if ``exc`` looks like "this responses surface does not serve this
    model" (so we should try the next surface), as opposed to a genuine request
    error that would fail everywhere."""
    status = getattr(exc, "status_code", None)
    try:
        status_int = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_int = None

    msg = (str(getattr(exc, "message", "") or "") + " " + str(exc)).lower()
    has_marker = any(m in msg for m in _FALLBACK_MARKERS)

    if status_int == 404:
        return True
    if status_int in _FALLBACK_STATUSES and has_marker:
        return True
    # Some transports surface the status only in the message body.
    return has_marker
