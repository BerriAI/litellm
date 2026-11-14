from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Literal

from pydantic import TypeAdapter, field_validator

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.types.guardrails import SupportedGuardrailIntegrations
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.proxy.guardrails.guardrail_hooks.straiker import StraikerOnFlaggedAction, StraikerStreamMode

from .straiker import StraikerGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


class _V3Routing(LiteLLMBaseModel):
    api_version: Literal["v1", "v3"] | None = None
    agent_ref: str | None = None
    client: str | None = None
    format_hint: Literal["anthropic.messages", "openai.chat"] | None = None

    @field_validator("api_version", mode="before")
    @classmethod
    def _unknown_api_version_is_unset(cls, value: object) -> object:
        if value is None or value in ("v1", "v3"):
            return value
        verbose_proxy_logger.warning(
            "Straiker guardrail: ignoring api_version %r, expected 'v1', 'v3' or unset; "
            "the route follows the api_key prefix",
            value,
        )
        return None


_STREAM_MODE: Final[TypeAdapter[StraikerStreamMode]] = TypeAdapter(StraikerStreamMode)
_CONFIG_ENTRIES: Final[TypeAdapter[Mapping[object, object]]] = TypeAdapter(Mapping[object, object])
# LitellmParams carries other guardrails' `on_flagged_action` defaults, 'monitor' at the top level and
# 'block' in optional_params, and a guardrail saved through the API stores both as if set. So 'passthrough'
# at either level opts in, and 'monitor' means "not set" here, not a typo.
_UNSET_ON_FLAGGED_ACTIONS: Final = ("monitor",)

_OPTIONAL_INIT_FIELDS: Final = (
    "timeout",
    "max_retries",
    "initial_backoff",
    "max_backoff",
    "unreachable_fallback",
    "fail_on_error",
    "max_payload_bytes",
    "custom_headers",
    "metadata",
    "verbose",
)


def _get_config_value(litellm_params: "LitellmParams", optional_params: object, attribute_name: str) -> object:
    if optional_params is not None:
        if isinstance(optional_params, dict):
            value = optional_params.get(attribute_name)
        else:
            value = getattr(optional_params, attribute_name, None)
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


def _explicit_value(params: object, attribute_name: str) -> object:
    """The value the operator set for `attribute_name`, or None when it is only a model default."""
    if isinstance(params, dict):
        return _CONFIG_ENTRIES.validate_python(params).get(attribute_name)
    fields_set: Final = getattr(params, "model_fields_set", None)
    if fields_set is None or attribute_name not in fields_set:
        return None
    return getattr(params, attribute_name, None)


def _on_flagged_action(litellm_params: "LitellmParams", optional_params: object) -> StraikerOnFlaggedAction:
    """'passthrough' only when the operator asked for it; anything else keeps the HTTP 400 block."""
    values: Final = (
        _explicit_value(optional_params, "on_flagged_action"),
        _explicit_value(litellm_params, "on_flagged_action"),
    )
    if "passthrough" in values:
        return "passthrough"
    unknown: Final = tuple(value for value in values if value not in (None, "block", *_UNSET_ON_FLAGGED_ACTIONS))
    if unknown:
        verbose_proxy_logger.warning(
            "Straiker guardrail: ignoring on_flagged_action %r, expected 'block' or 'passthrough'; blocking with HTTP 400",
            unknown[0],
        )
    return "block"


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    optional_params: Final = getattr(litellm_params, "optional_params", None)
    api_key: Final = litellm_params.api_key
    if not api_key:
        raise ValueError("api_key is required for straiker")

    api_base: Final = litellm_params.api_base or "https://api.prod.straiker.ai"
    default_app: Final = getattr(litellm_params, "default_app", None) or getattr(litellm_params, "source", None)
    source: Final = default_app if isinstance(default_app, str) and default_app else "LiteLLM Gateway"
    kwargs: Final[dict[str, object]] = {
        field: value
        for field in _OPTIONAL_INIT_FIELDS
        for value in [_get_config_value(litellm_params, optional_params, field)]
        if value is not None
    }
    routing: Final = _V3Routing.model_validate(
        {
            field: _get_config_value(litellm_params, optional_params, field)
            for field in ("api_version", "agent_ref", "client", "format_hint")
        }
    )
    _callback: Final = StraikerGuardrail(
        api_key=api_key,
        api_base=api_base if isinstance(api_base, str) else "https://api.prod.straiker.ai",
        source=source,
        guardrail_name=guardrail.get("guardrail_name", "straiker"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        api_version=routing.api_version,
        agent_ref=routing.agent_ref,
        client=routing.client,
        format_hint=routing.format_hint,
        stream_mode=_STREAM_MODE.validate_python(
            _get_config_value(litellm_params, optional_params, "stream_mode") or "buffered"
        ),
        on_flagged_action=_on_flagged_action(litellm_params, optional_params),
        **kwargs,
    )

    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.STRAIKER.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.STRAIKER.value: StraikerGuardrail,
}
