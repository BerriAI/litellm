import importlib
from typing import TYPE_CHECKING, Callable, Dict, Tuple

from litellm.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import Guardrail, LitellmParams


_HOOKS = "litellm.proxy.guardrails.guardrail_hooks"
_INITIALIZERS = "litellm.proxy.guardrails.guardrail_initializers"

# Every guardrail type the Rust engine implements, mapped to the Python
# initializer used as a fallback when the Rust wheel is not installed or the
# config uses a feature Rust does not support yet (module_path, function_name).
# This module is the single source of truth for routing these types; adding a
# provider is one entry here plus a branch in build_v2_config.
_PYTHON_FALLBACKS: Dict[str, Tuple[str, str]] = {
    SupportedGuardrailIntegrations.GENERIC_GUARDRAIL_API.value: (
        f"{_HOOKS}.generic_guardrail_api",
        "initialize_python_guardrail",
    ),
    SupportedGuardrailIntegrations.OPENAI_MODERATION.value: (
        f"{_HOOKS}.openai",
        "initialize_python_guardrail",
    ),
    SupportedGuardrailIntegrations.AZURE_PROMPT_SHIELD.value: (
        f"{_HOOKS}.azure",
        "initialize_python_guardrail",
    ),
    SupportedGuardrailIntegrations.AZURE_TEXT_MODERATIONS.value: (
        f"{_HOOKS}.azure",
        "initialize_python_guardrail",
    ),
    SupportedGuardrailIntegrations.PRESIDIO.value: (
        _INITIALIZERS,
        "initialize_presidio",
    ),
    SupportedGuardrailIntegrations.LAKERA_V2.value: (
        _INITIALIZERS,
        "initialize_lakera_v2",
    ),
    SupportedGuardrailIntegrations.BEDROCK.value: (
        _INITIALIZERS,
        "initialize_bedrock",
    ),
}

RUST_PROVIDERS = tuple(_PYTHON_FALLBACKS.keys())


def _initialize_rust(
    guardrail_type: str,
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
) -> "CustomGuardrail | None":
    """Build a GuardrailV2, or return None if the engine cannot handle this config."""
    import litellm
    from litellm._logging import verbose_proxy_logger

    from .guardrail_v2 import GuardrailV2, build_v2_config

    engine_config = build_v2_config(guardrail_type, litellm_params)
    if engine_config is None:
        return None

    instance = GuardrailV2(
        engine_config=engine_config,
        extra_headers=getattr(litellm_params, "extra_headers", None),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    verbose_proxy_logger.info(
        "Guardrail '%s' (%s) initialized with Rust engine",
        guardrail.get("guardrail_name", ""),
        guardrail_type,
    )
    return instance


def _initialize_python_fallback(
    guardrail_type: str,
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
) -> "CustomGuardrail":
    module_path, fn_name = _PYTHON_FALLBACKS[guardrail_type]
    initializer = getattr(importlib.import_module(module_path), fn_name)
    return initializer(litellm_params, guardrail)


def _make_initializer(guardrail_type: str) -> Callable:
    def initialize_guardrail(
        litellm_params: "LitellmParams", guardrail: "Guardrail"
    ) -> "CustomGuardrail":
        instance = _initialize_rust(guardrail_type, litellm_params, guardrail)
        if instance is not None:
            return instance
        return _initialize_python_fallback(guardrail_type, litellm_params, guardrail)

    return initialize_guardrail


guardrail_initializer_registry: Dict[str, Callable] = {
    guardrail_type: _make_initializer(guardrail_type)
    for guardrail_type in RUST_PROVIDERS
}
