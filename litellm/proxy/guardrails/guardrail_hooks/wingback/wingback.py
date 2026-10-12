from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Final, Literal

from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.wingback import (
    WingbackGuardrailConfigModel,
)

GUARDRAIL_NAME: Final = "wingback"
DEFAULT_WINGBACK_API_BASE: Final = "https://api.wingback.ai/connectors"


class WingbackGuardrail(GenericGuardrailAPI):
    """Wingback runtime security via the LiteLLM Generic Guardrail API contract."""

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        wingback_app_id: str | None = None,
        additional_provider_specific_params: Mapping[str, object] | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        fail_on_error: bool | None = True,
        extra_headers: Sequence[str] | None = None,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | Sequence[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = False,
        timeout: float | None = None,
    ) -> None:
        resolved_api_base: Final = api_base or os.environ.get("WINGBACK_API_BASE") or DEFAULT_WINGBACK_API_BASE
        resolved_api_key: Final = api_key or os.environ.get("WINGBACK_INTEGRATION_API_KEY")

        existing_params: Final = additional_provider_specific_params
        additional_params: Final = (
            {
                **existing_params,
                "wingback_app_id": wingback_app_id,
            }
            if wingback_app_id and existing_params is not None and "wingback_app_id" not in existing_params
            else (
                {
                    "wingback_app_id": wingback_app_id
                }
                if wingback_app_id and existing_params is None
                else existing_params
            )
        )

        super().__init__(
            api_base=resolved_api_base,
            api_key=resolved_api_key,
            additional_provider_specific_params=additional_params,
            unreachable_fallback=unreachable_fallback,
            fail_on_error=fail_on_error,
            extra_headers=list(extra_headers) if extra_headers is not None else None,
            guardrail_name=guardrail_name or GUARDRAIL_NAME,
            event_hook=event_hook,
            default_on=default_on,
            timeout=timeout,
        )

    @classmethod
    def get_config_model(cls) -> type[WingbackGuardrailConfigModel] | None:
        return WingbackGuardrailConfigModel
