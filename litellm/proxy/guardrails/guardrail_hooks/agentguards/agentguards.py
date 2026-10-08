# +-------------------------------------------------------------+
#
#           Use AgentGuards Guardrails for your LLM calls
#                   https://agentguards.co/
#
# +-------------------------------------------------------------+

import os
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypedDict, Unpack, override

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException, Timeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # decorator is untyped in custom_guardrail
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # helper is untyped in http_handler
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.agentguards import AgentGuardsGuardrailConfigModel
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DEFAULT_API_BASE: Final = "https://prod.agentguards.co"
DEFAULT_USE_CASE: Final = "check"
INPUT_PATH: Final = "/v1/guardrails/evaluate-input"
OUTPUT_PATH: Final = "/v1/outputs/validate"
TIMEOUT_SECONDS: Final = 10.0
# AgentGuards decisions that stop the request / response.
BLOCKING_INPUT_DECISIONS: Final = frozenset(("block", "escalate"))
BLOCKING_OUTPUT_DECISIONS: Final = frozenset(("reject", "escalate"))
BLOCKED_REASON: Final = "Blocked by AgentGuards"
UNREACHABLE_REASON: Final = "AgentGuards guardrail service unreachable"
AGENTGUARDS_ERRORS: Final = (httpx.RequestError, httpx.HTTPStatusError, Timeout, ValidationError, ValueError)
OBJECT_MAPPING: Final = TypeAdapter(Mapping[str, object])
EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
INPUT_HOOKS: Final = MappingProxyType(
    {
        "request": frozenset((GuardrailEventHooks.pre_call,)),
        "response": frozenset((GuardrailEventHooks.post_call,)),
    }
)


def as_mapping(value: object) -> Mapping[str, object]:
    """A tool call (pydantic model or dict) or its `function` as a read-only mapping; anything else is empty."""
    plain: Final = value.model_dump() if isinstance(value, BaseModel) else value
    try:
        return OBJECT_MAPPING.validate_python(plain)
    except ValidationError:
        return EMPTY


class _CustomGuardrailKwargs(TypedDict):
    guardrail_name: NotRequired[ReadOnly[str | None]]
    event_hook: NotRequired[  # mutable-ok: CustomGuardrail API
        ReadOnly[GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None]
    ]
    default_on: NotRequired[ReadOnly[bool]]


class AgentGuardsVerdict(BaseModel):
    """The fields this integration reads from an AgentGuards decision; the rest is ignored."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    decision: str = "allow"
    message: str | None = None
    redacted_text: str | None = None


class AgentGuardsGuardrail(CustomGuardrail):
    """
    AgentGuards guardrail integration for LiteLLM.

    Screens requests with `/v1/guardrails/evaluate-input` (jailbreaks, prompt injection,
    PII and secrets) and validates responses with `/v1/outputs/validate` (data exfiltration),
    according to the AgentGuards policy for the API key's tenant.
    """

    @staticmethod
    def get_config_model() -> type[AgentGuardsGuardrailConfigModel] | None:
        return AgentGuardsGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: overrides the base signature
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        agentguards_use_case: str | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        *,
        async_handler: AsyncHTTPHandler | None = None,
        **kwargs: Unpack[_CustomGuardrailKwargs],  # kwargs-ok: typed CustomGuardrail options passed through
    ) -> None:
        self.async_handler: AsyncHTTPHandler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        self.api_key: str = api_key or os.environ.get("AGENTGUARDS_API_KEY", "")
        if not self.api_key:
            raise ValueError("api_key is required. Set AGENTGUARDS_API_KEY or pass it in litellm_params.")
        self.api_base: str = (api_base or os.environ.get("AGENTGUARDS_API_BASE") or DEFAULT_API_BASE).rstrip("/")
        self.use_case: str = agentguards_use_case or DEFAULT_USE_CASE
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = unreachable_fallback

        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.__init__ is untyped
            supported_event_hooks=self.get_supported_event_hooks(),
            **kwargs,
        )

        verbose_proxy_logger.debug(
            "AgentGuards guardrail initialized: api_base=%s fallback=%s",
            self.api_base,
            self.unreachable_fallback,
        )

    def handles(self, input_type: Literal["request", "response"]) -> bool:
        if self.event_hook is None or isinstance(self.event_hook, Mode):
            return True
        configured: Final = self.event_hook if isinstance(self.event_hook, list) else (self.event_hook,)
        return any(GuardrailEventHooks(hook) in INPUT_HOOKS[input_type] for hook in configured)

    @staticmethod
    def tool_call_texts(inputs: GenericGuardrailAPIInputs) -> tuple[str, ...]:
        """`name(arguments)` for each tool call, so exfiltration in tool arguments is validated too."""
        functions: Final = (as_mapping(as_mapping(call).get("function")) for call in inputs.get("tool_calls") or ())
        return tuple(
            f"{function.get('name')}({function.get('arguments') or ''})"
            for function in functions
            if isinstance(function.get("name"), str) and function.get("name")
        )

    async def post(self, path: str, payload: Mapping[str, object]) -> AgentGuardsVerdict:
        response: Final = await self.async_handler.post(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler.post is untyped
            url=f"{self.api_base}{path}",
            headers={"Content-Type": "application/json", "X-API-Key": self.api_key},
            json=dict(payload),
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return AgentGuardsVerdict.model_validate(response.json())

    def unreachable(self, error: Exception, inputs: GenericGuardrailAPIInputs) -> GenericGuardrailAPIInputs:
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.critical("AgentGuards unreachable (fail-open): %s", str(error), exc_info=error)
            return inputs
        verbose_proxy_logger.error("AgentGuards unreachable (fail-closed): %s", str(error))
        raise GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=UNREACHABLE_REASON,
            should_wrap_with_default_message=False,
            status_code=503,
        )

    @staticmethod
    def blocked(verdict: AgentGuardsVerdict) -> HTTPException:
        return HTTPException(
            status_code=400,
            detail={
                "error": BLOCKED_REASON,
                "decision": verdict.decision,
                "message": verdict.message,
            },
        )

    @override
    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: overrides CustomGuardrail.apply_guardrail
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        """Screen the request (pre_call) or validate the response (post_call)."""
        if not self.handles(input_type):
            return inputs
        texts: Final[Sequence[str]] = tuple(text for text in inputs.get("texts") or () if text)
        if input_type == "request":
            return await self.screen_request(inputs, texts)
        return await self.validate_response(inputs, texts)

    async def screen_request(
        self, inputs: GenericGuardrailAPIInputs, texts: Sequence[str]
    ) -> GenericGuardrailAPIInputs:
        if not texts:
            return inputs
        try:
            verdict: Final = await self.post(
                INPUT_PATH,
                {"text": "\n\n".join(texts), "use_case": self.use_case, "channel": "api", "metadata": {}},
            )
        except AGENTGUARDS_ERRORS as error:
            return self.unreachable(error, inputs)
        if verdict.decision in BLOCKING_INPUT_DECISIONS:
            raise self.blocked(verdict)
        if verdict.decision == "redact" and verdict.redacted_text:
            # One redacted string maps back only onto a single text; with several, the request
            # was still screened as a whole, and is passed on unredacted rather than misaligned.
            if len(texts) == 1:
                redacted: Final[GenericGuardrailAPIInputs] = {**inputs, "texts": [verdict.redacted_text]}
                return redacted
            verbose_proxy_logger.warning("AgentGuards redaction spans several texts; passing them unredacted.")
        return inputs

    async def validate_response(
        self, inputs: GenericGuardrailAPIInputs, texts: Sequence[str]
    ) -> GenericGuardrailAPIInputs:
        output: Final = "\n".join((*texts, *self.tool_call_texts(inputs)))
        if not output:
            return inputs
        try:
            verdict: Final = await self.post(OUTPUT_PATH, {"output_text": output, "context_text": "", "metadata": {}})
        except AGENTGUARDS_ERRORS as error:
            return self.unreachable(error, inputs)
        if verdict.decision in BLOCKING_OUTPUT_DECISIONS:
            raise self.blocked(verdict)
        return inputs
