import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, NoReturn, Optional

import httpx
from pydantic import TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypeAliasType, TypedDict, Unpack, assert_never

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException, Timeout
from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME: Final = "beyondguard"
_TOKEN_HEADER: Final = "X-Guard-Token"
_DEFAULT_TIMEOUT_SECONDS: Final = 10.0


class _PiiOperations(TypedDict, total=False):
    content: ReadOnly[str | None]


class _CheckDetails(TypedDict, total=False):
    pii_operations: ReadOnly[_PiiOperations]


class _CheckItem(TypedDict, total=False):
    details: ReadOnly[_CheckDetails | None]


class _ChecksResult(TypedDict):
    passed: ReadOnly[bool]
    sanitized: NotRequired[ReadOnly[bool]]
    details: NotRequired[ReadOnly[tuple[_CheckItem, ...]]]


class _RejectionCode(TypedDict):
    code: ReadOnly[str]


_ContextGuardResponse = TypedDict(  # rebind-ok: functional form keeps the dunder-prefixed keys unmangled
    "_ContextGuardResponse",
    {
        "__beyond_guard_checks_result": ReadOnly[_ChecksResult],
        "__beyond_guard_rejection_codes": NotRequired[ReadOnly[tuple[_RejectionCode, ...]]],
    },
)

_RESPONSE: Final = TypeAdapter(_ContextGuardResponse)


@dataclass(frozen=True, slots=True)
class _Allow:
    pass


@dataclass(frozen=True, slots=True)
class _Block:
    codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Mask:
    text: str


@dataclass(frozen=True, slots=True)
class _Failure:
    reason: str


_Verdict = TypeAliasType("_Verdict", _Allow | _Block | _Mask | _Failure)


@dataclass(frozen=True, slots=True)
class _Target:
    endpoint_id: str
    api_key: str


class _GuardrailOptions(TypedDict, total=False, extra_items=object):
    guardrail_name: ReadOnly[str]
    default_on: ReadOnly[bool]
    timeout: ReadOnly[float | None]


class BeyondGuardMissingConfig(Exception):
    pass


def _masked_text(checks: _ChecksResult) -> str | None:
    for item in checks.get("details", ()):
        details: Final = item.get("details") or {}
        content: Final = details.get("pii_operations", {}).get("content")
        if content is not None:
            return content
    return None


def _verdict_from_response(body: _ContextGuardResponse) -> _Verdict:
    checks: Final = body["__beyond_guard_checks_result"]
    if checks["passed"]:
        if not checks.get("sanitized", False):
            return _Allow()
        masked: Final = _masked_text(checks)
        return _Mask(masked) if masked is not None else _Failure("sanitized_without_content")
    codes: Final = tuple(code["code"] for code in body.get("__beyond_guard_rejection_codes", ()))
    return _Block(codes) if codes else _Failure("rejected_without_codes")


class BeyondGuardGuardrail(CustomGuardrail):
    """Calls BeyondGuard Context Guard endpoints and maps their verdict to allow, block or mask.

    BeyondGuard checks the last user message of a request, so every text of one call is checked
    joined. When BeyondGuard masks the joined text, each text is re-checked on its own so the masked
    texts map back one to one. Technical failures always fail closed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        beyondguard_endpoint_id: str | None = None,
        beyondguard_output_endpoint_id: str | None = None,
        beyondguard_output_api_key: str | None = None,
        http_client: AsyncHTTPHandler | None = None,
        **kwargs: Unpack[_GuardrailOptions],  # kwargs-ok: CustomGuardrail options
    ) -> None:
        key: Final = (api_key or os.environ.get("BEYONDGUARD_API_KEY") or "").strip()
        base: Final = api_base or os.environ.get("BEYONDGUARD_API_BASE")
        endpoint_id: Final = beyondguard_endpoint_id or os.environ.get("BEYONDGUARD_ENDPOINT_ID")
        if not key or not base or not endpoint_id:
            raise BeyondGuardMissingConfig(
                "BeyondGuard needs api_key, api_base and beyondguard_endpoint_id "
                "(or BEYONDGUARD_API_KEY, BEYONDGUARD_API_BASE and BEYONDGUARD_ENDPOINT_ID)."
            )
        self.api_base = base.rstrip("/")
        self.request_target = _Target(endpoint_id=endpoint_id, api_key=key)
        self.response_target = _Target(
            endpoint_id=beyondguard_output_endpoint_id or endpoint_id,
            api_key=(beyondguard_output_api_key or "").strip() or key,
        )
        self.http_client = http_client or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        super().__init__(supported_event_hooks=self.get_supported_event_hooks(), **kwargs)

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.beyondguard import BeyondGuardConfigModel

        return BeyondGuardConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: CustomGuardrail signature
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,  # mutable-ok: CustomGuardrail.apply_guardrail signature
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = tuple(inputs.get("texts") or ())
        positions: Final = tuple(index for index, text in enumerate(texts) if text.strip())
        if not positions:
            return inputs
        target: Final = self.request_target if input_type == "request" else self.response_target
        checked: Final = tuple(texts[index] for index in positions)

        verdict: Final = await self._check(target, "\n".join(checked))
        match verdict:
            case _Allow():
                return inputs
            case _Block() | _Failure():
                self._raise(verdict)
            case _Mask(text=masked) if len(checked) == 1:
                return self._with_texts(inputs, texts, positions, (masked,))
            case _Mask():
                masked_each: Final = await self._mask_each(target, checked)
                return self._with_texts(inputs, texts, positions, masked_each)
            case _:
                assert_never(verdict)

    async def _mask_each(self, target: _Target, texts: tuple[str, ...]) -> tuple[str, ...]:
        verdicts: Final = await asyncio.gather(*(self._check(target, text) for text in texts))
        for verdict in verdicts:
            if isinstance(verdict, _Block | _Failure):
                self._raise(verdict)
        return tuple(
            verdict.text if isinstance(verdict, _Mask) else original
            for verdict, original in zip(verdicts, texts, strict=True)
        )

    async def _check(self, target: _Target, text: str) -> _Verdict:
        url: Final = f"{self.api_base}/{target.endpoint_id}/chat/completions"
        try:
            response: Final = await self.http_client.post(
                url=url,
                json={"model": GUARDRAIL_NAME, "messages": [{"role": "user", "content": text}]},
                headers={_TOKEN_HEADER: target.api_key, "Content-Type": "application/json"},
                timeout=self.timeout if self.timeout is not None else _DEFAULT_TIMEOUT_SECONDS,
            )
            body: Final = _RESPONSE.validate_python(response.json())
        except httpx.HTTPStatusError as e:
            return _Failure(f"http_{e.response.status_code}")
        except (httpx.HTTPError, Timeout, ValidationError, ValueError) as e:
            return _Failure(type(e).__name__)
        verdict: Final = _verdict_from_response(body)
        verbose_proxy_logger.debug(
            "BeyondGuard verdict=%s request_id=%s", type(verdict).__name__, response.headers.get("X-Request-ID")
        )
        return verdict

    def _raise(self, verdict: _Block | _Failure) -> NoReturn:
        match verdict:
            case _Block(codes=codes):
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
                    message=f"Blocked by BeyondGuard ({', '.join(codes)})",
                    should_wrap_with_default_message=False,
                    blocked_content=True,
                )
            case _Failure(reason=reason):
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
                    message=f"BeyondGuard unavailable ({reason})",
                    should_wrap_with_default_message=False,
                    status_code=503,
                )
            case _:
                assert_never(verdict)

    @staticmethod
    def _with_texts(
        inputs: GenericGuardrailAPIInputs,
        texts: tuple[str, ...],
        positions: tuple[int, ...],
        masked: tuple[str, ...],
    ) -> GenericGuardrailAPIInputs:
        replacements: Final = dict(zip(positions, masked, strict=True))
        return {**inputs, "texts": [replacements.get(index, text) for index, text in enumerate(texts)]}
