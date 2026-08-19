from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Optional

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.reco import validate_reco_tenant_id
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class RecoGuardrail(GenericGuardrailAPI):
    """Reco guardrail integration for LiteLLM, built on the Generic Guardrail API wire contract."""

    def __init__(
        self,
        reco_tenant_id: str | None,
        api_base: str | None,
        headers: Mapping[str, str] | None = None,
        **kwargs,  # noqa: ANN003  # forwards guardrail_name/event_hook/default_on to CustomGuardrail, whose types are narrower than LitellmParams' own field types
    ) -> None:
        if not reco_tenant_id:
            raise ValueError("reco_tenant_id is required for the Reco guardrail")
        if not api_base:
            raise ValueError("api_base is required for the Reco guardrail")

        validated_tenant_id = validate_reco_tenant_id(reco_tenant_id)
        base_headers = headers or {}  # mutable-ok: normalizes headers to a dict
        merged_headers = {**base_headers, "X-Reco-Tenant-Id": validated_tenant_id}  # mutable-ok: required by base class

        super().__init__(
            headers=merged_headers,
            api_base=api_base,
            unreachable_fallback="fail_open",
            fail_on_error=False,
            **kwargs,
        )

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,  # mutable-ok: must match GenericGuardrailAPI.apply_guardrail's own param type
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        # GenericGuardrailAPI raises with a hardcoded module-level guardrail_name on BLOCKED,
        # not the guardrail's own configured name. Re-raise with the right one instead of
        # patching the shared base class.
        try:
            return await super().apply_guardrail(inputs, request_data, input_type, logging_obj)
        except GuardrailRaisedException as e:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=e.message,
                should_wrap_with_default_message=False,
                status_code=e.status_code,
            ) from e

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.reco import RecoConfigModel

        return RecoConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: fixed by base class
        return [GuardrailEventHooks.pre_call]  # mutable-ok: fixed by base class
