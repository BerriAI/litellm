from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .conduct import ConductGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


# ─── Constants ────────────────────────────────────────────────────────
# Kept as module-level names so the intent is obvious in review — no
# `getattr(..., 8.0)` default that gets silently discarded because the
# field always exists as None (cursor[bot] finding).
_DEFAULT_TIMEOUT_SECONDS = 8.0
_DEFAULT_UNREACHABLE_FALLBACK = "fail_closed"


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    """Initialize the Conduct guardrail from LiteLLM's config block.

    Maps LiteLLM's typed guardrail fields onto the Conduct constructor:

      LiteLLM field                  → Conduct kwarg
      ───────────────────────────────────────────────
      api_base                       → api_url
      api_key                        → agent_token
      workspace_id (extra)           → workspace_id
      unreachable_fallback           → fail_mode (fail_closed | fail_open)
      timeout                        → timeout

    The typed ``unreachable_fallback`` field replaces the free-form
    ``fail_mode`` this shim previously read. A typo on the old field
    silently defaulted the plugin to fail-open behavior; using the
    typed field forces Pydantic validation upstream (yucheng-berri,
    devin-ai-integration findings).
    """
    import litellm

    # ``getattr(..., default)`` only fires when the attribute is missing;
    # ``LitellmParams`` always defines ``timeout`` and defaults it to
    # ``None``, so the default was never applied. Use ``or`` so an
    # explicit ``None`` (or ``0``) also falls through to the intended
    # 8-second budget (cursor[bot] finding).
    timeout = getattr(litellm_params, "timeout", None) or _DEFAULT_TIMEOUT_SECONDS
    unreachable_fallback = (
        getattr(litellm_params, "unreachable_fallback", None)
        or _DEFAULT_UNREACHABLE_FALLBACK
    )

    _conduct_callback: Final = ConductGuardrail(
        api_url=getattr(litellm_params, "api_base", None),
        agent_token=getattr(litellm_params, "api_key", None),
        workspace_id=getattr(litellm_params, "workspace_id", None),
        # Conduct's constructor argument is still ``fail_mode`` — mapped
        # from the typed LiteLLM field above.
        fail_mode=unreachable_fallback,
        timeout=timeout,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_conduct_callback)

    return _conduct_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.CONDUCT.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.CONDUCT.value: ConductGuardrail,
}
