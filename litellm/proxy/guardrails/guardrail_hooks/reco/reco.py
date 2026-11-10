from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, Optional

from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # decorator is untyped
)
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.reco import validate_reco_tenant_id
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
    from litellm.types.proxy.guardrails.guardrail_hooks.reco import RecoOptionalParams


class RecoGuardrail(GenericGuardrailAPI):
    """Reco guardrail integration for LiteLLM, built on the Generic Guardrail API wire contract."""

    def __init__(
        self,
        reco_tenant_id: str | None,
        api_base: str | None,
        headers: Mapping[str, str] | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        fail_on_error: bool = True,
        guardrail_name: str | None = None,
        event_hook: str | Sequence[str] | Mode | None = None,
        default_on: bool = False,
    ) -> None:
        if not reco_tenant_id:
            raise ValueError("reco_tenant_id is required for the Reco guardrail")
        if not api_base:
            raise ValueError("api_base is required for the Reco guardrail")

        validated_tenant_id: Final = validate_reco_tenant_id(reco_tenant_id)

        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # parent takes untyped extra_headers and **kwargs
            headers={**(headers or {}), "X-Reco-Tenant-Id": validated_tenant_id},
            api_base=api_base,
            unreachable_fallback=unreachable_fallback,
            fail_on_error=fail_on_error,
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: must match GenericGuardrailAPI.apply_guardrail's own param type
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        # GenericGuardrailAPI raises with a hardcoded module-level guardrail_name on BLOCKED,
        # not the guardrail's own configured name. Re-raise with the right one instead of
        # patching the shared base class.
        try:
            return await super().apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # parent annotates request_data as a bare dict
                inputs=inputs, request_data=request_data, input_type=input_type, logging_obj=logging_obj
            )
        except GuardrailRaisedException as e:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=e.message,
                should_wrap_with_default_message=False,
                status_code=e.status_code,
                blocked_content=e.blocked_content,
            ) from e

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel[RecoOptionalParams]"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.reco import RecoConfigModel

        return RecoConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: fixed by base class
        return [GuardrailEventHooks.pre_call]
