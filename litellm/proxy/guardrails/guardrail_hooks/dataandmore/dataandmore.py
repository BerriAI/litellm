from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Final, Literal

from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.types.guardrails import Mode
from litellm.types.proxy.guardrails.guardrail_hooks.dataandmore import (
    DataAndMoreGuardrailConfigModel,
)

GUARDRAIL_NAME: Final = "dataandmore"


class DataAndMoreGuardrail(GenericGuardrailAPI):
    """Data & More DLP via the LiteLLM Generic Guardrail API contract."""

    apply_guardrail = GenericGuardrailAPI.apply_guardrail  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # parent annotates request_data as a bare dict

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        additional_provider_specific_params: Mapping[str, object] | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_open",
        fail_on_error: bool | None = True,
        extra_headers: Sequence[str] | None = None,
        guardrail_name: str | None = None,
        event_hook: str | Sequence[str] | Mode | None = None,
        default_on: bool | None = False,
        timeout: float | None = None,
    ) -> None:
        resolved_api_base: Final = api_base or os.environ.get("DGM_GUARDRAIL_API_BASE")
        if not resolved_api_base:
            raise ValueError(
                "Data & More guardrail requires api_base "
                "(e.g. https://dgm.dataandmore.com/api/dlp/ai-gateway/litellm) "
                "or the DGM_GUARDRAIL_API_BASE environment variable"
            )
        resolved_api_key: Final = api_key or os.environ.get("DGM_GUARDRAIL_KEY")

        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # parent takes untyped extra_headers and **kwargs
            api_base=resolved_api_base,
            api_key=resolved_api_key,
            additional_provider_specific_params=additional_provider_specific_params,
            unreachable_fallback=unreachable_fallback,
            fail_on_error=fail_on_error,
            extra_headers=list(extra_headers) if extra_headers is not None else None,
            guardrail_name=guardrail_name or GUARDRAIL_NAME,
            event_hook=event_hook,
            default_on=default_on,
            timeout=timeout,
        )

    @staticmethod
    def get_config_model() -> type[DataAndMoreGuardrailConfigModel] | None:
        return DataAndMoreGuardrailConfigModel
