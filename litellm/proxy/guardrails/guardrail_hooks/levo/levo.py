"""Levo AI Gateway guardrail for LiteLLM.

Sends prompts and completions to a self-hosted Levo AI Gateway, which runs
Levo's policy engine — data-protection and content-safety scanners, CEL-based
access control, MCP tool policies — and answers allow / block / rewrite.

The gateway speaks LiteLLM's Basic Guardrail API, so the request and response
bodies here are the same shape `generic_guardrail_api` uses. This integration
exists on top of that for two reasons:

1. Streamed responses are buffered until moderated by default. The generic
   integration cannot enable that, so a response-side finding on a streaming
   call arrives after the client already has the content.
2. `guardrail: levo` with a typed config model, rather than a generic endpoint
   plus provider-specific parameters.

Gateway docs: https://docs.levo.ai/install-ai-gateway/ai-gateway-docker
"""

from typing import TYPE_CHECKING, Final, Literal

from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

#: Path the gateway serves; appended to the configured api_base.
LEVO_GUARDRAIL_PATH = "/beta/litellm_basic_guardrail_api"


class LevoGuardrail(GenericGuardrailAPI):
    """Guardrail backed by a self-hosted Levo AI Gateway."""

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        buffer_streaming_until_moderated: bool | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        **kwargs: object,
    ) -> None:
        super().__init__(
            api_base=api_base,
            api_key=api_key,
            unreachable_fallback=unreachable_fallback,
            **kwargs,
        )

        # Read by UnifiedLLMGuardrails.async_post_call_streaming_iterator_hook via
        # getattr(guardrail_to_apply, "streaming_*", default).
        #
        # Default to buffering: a response-side block is only meaningful if it
        # lands before the client sees the content. Without this, chunks are
        # emitted as they are produced and a violation is detected after the
        # fact. Operators who need time-to-first-token more than response-side
        # enforcement can set buffer_streaming_until_moderated=False.
        buffer: Final = True if buffer_streaming_until_moderated is None else bool(buffer_streaming_until_moderated)
        self.streaming_buffer_until_moderated = buffer
        if buffer:
            # Buffering can only moderate the assembled response, so it always
            # implies end-of-stream evaluation.
            self.streaming_end_of_stream_only = True

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        """Scan a request or response via the Levo AI Gateway.

        Delegates to the base implementation — the wire contract is identical.
        This override must exist **in this class body**: the proxy selects the
        unified guardrail path with
        ``"apply_guardrail" in type(callback).__dict__``, which inspects the
        class's own attributes and does not see inherited methods. Without it
        the guardrail is constructed and consulted, but never invoked, so every
        request passes unscanned.
        """
        return await super().apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type=input_type,
            logging_obj=logging_obj,
        )

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.levo import (
            LevoGuardrailConfigModel,
        )

        return LevoGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        # pre_call scans prompts, post_call scans completions. during_call is
        # deliberately excluded: it duplicates the pre_call input event without
        # adding a decision point.
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]
