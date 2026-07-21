"""Rubrik guardrail integration for LiteLLM."""

from typing import TYPE_CHECKING

from litellm.integrations.rubrik import RubrikLogger
from litellm.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> RubrikLogger:
    """Create and register a RubrikLogger instance.

    The ``mode`` field in the guardrail config controls which surfaces are
    moderated:
    - ``pre_call`` (or a mode that includes it): prompt moderation via the
      ``/v1/before_prompt/openai/v1`` webhook.
    - ``post_call`` (the default when ``mode`` is omitted): response and tool
      call moderation via the ``/v1/after_completion/openai/v1`` webhook.

    Both hooks are active when ``mode`` covers both ``pre_call`` and
    ``post_call``.
    """
    import litellm

    # `LitellmParams.__init__` converts an omitted `default_on` to `False`
    # before we receive it, so `litellm_params.default_on` is always `bool`
    # and never `None`. Read the raw config dict to distinguish an explicit
    # `default_on: false` from the absent case, which for Rubrik should
    # default to `True` (the guardrail is meant to be on by default).
    raw_litellm_params = guardrail.get("litellm_params") or {}
    raw_default_on = (
        raw_litellm_params.get("default_on")
        if isinstance(raw_litellm_params, dict)
        else getattr(raw_litellm_params, "default_on", None)
    )
    default_on = litellm_params.default_on if raw_default_on is not None else True

    rubrik_callback = RubrikLogger(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(rubrik_callback)
    return rubrik_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.RUBRIK.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.RUBRIK.value: RubrikLogger,
}
