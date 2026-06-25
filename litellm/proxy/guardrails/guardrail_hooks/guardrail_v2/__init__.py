import importlib
from typing import TYPE_CHECKING, Callable, Optional, cast

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
# provider is one entry here plus a branch in the Rust config_builder.
_PYTHON_FALLBACKS: dict[str, tuple[str, str]] = {
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
    """Build a GuardrailV2, or return None when the engine is unavailable or
    cannot handle this config; the caller then uses the Python implementation."""
    import litellm
    from litellm._logging import verbose_proxy_logger

    from .guardrail_v2 import GuardrailV2, _get_optional_param, config_supported

    params = (
        litellm_params.model_dump()
        if hasattr(litellm_params, "model_dump")
        else dict(litellm_params)
    )

    try:
        if not config_supported(guardrail_type, params):
            return None
    except ImportError:
        # Rust engine not built/installed; fall back to the Python implementation.
        return None

    instance = GuardrailV2(
        guardrail_type=guardrail_type,
        params=params,
        extra_headers=getattr(litellm_params, "extra_headers", None),
        streaming_end_of_stream_only=cast(
            Optional[bool],
            _get_optional_param(litellm_params, "streaming_end_of_stream_only"),
        ),
        streaming_sampling_rate=cast(
            Optional[int],
            _get_optional_param(litellm_params, "streaming_sampling_rate"),
        ),
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


guardrail_initializer_registry: dict[str, Callable] = {
    guardrail_type: _make_initializer(guardrail_type)
    for guardrail_type in RUST_PROVIDERS
}
