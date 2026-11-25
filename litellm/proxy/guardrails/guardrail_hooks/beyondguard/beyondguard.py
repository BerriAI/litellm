import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NoReturn, Optional

import httpx
from pydantic import TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypeAliasType, TypedDict, Unpack, assert_never

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException, Timeout
from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
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
_APPLICATION_HEADER: Final = "X-Application"
_DESTINATION_URL_HEADER: Final = "X-Destination-Url"
_DESTINATION_MODEL_HEADER: Final = "X-Destination-Model"
_DEFAULT_USER_EMAIL_HEADER: Final = "X-User-Email"
_DEFAULT_TIMEOUT_SECONDS: Final = 10.0
# Masking a multi-text call re-checks every text; cap the fan-out per call so one caller
# cannot turn a request into an unbounded number of BeyondGuard calls.
_MAX_MASK_SEGMENTS: Final = 16
_MASK_RECHECK_CONCURRENCY: Final = 4


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
class _RequestContext:
    model: str
    headers: Mapping[str, str]


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


class _CallerMetadata(TypedDict, total=False):
    user_api_key_user_email: ReadOnly[str | None]
    user_api_key_request_route: ReadOnly[str | None]


class _StreamFlag(TypedDict, total=False):
    stream: ReadOnly[object]


class _ProxyServerRequest(TypedDict, total=False):
    body: ReadOnly[_StreamFlag | None]


class _PolicyRequestData(TypedDict, total=False):
    model: ReadOnly[str | None]
    stream: ReadOnly[object]
    proxy_server_request: ReadOnly[_ProxyServerRequest | None]
    litellm_metadata: ReadOnly[_CallerMetadata | None]
    metadata: ReadOnly[_CallerMetadata | None]


_POLICY_REQUEST_DATA: Final = TypeAdapter(_PolicyRequestData)


def _policy_request_data(request_data: object) -> _PolicyRequestData:
    try:
        return _POLICY_REQUEST_DATA.validate_python(request_data)
    except ValidationError:
        return {}


def _stream_cannot_carry_a_mask(request_data: object, input_type: str) -> bool:
    """Whether a masked model response would be replaced by the original text on its way to the client.

    Streamed responses are rewritten in place only on chat completions (``incremental_diff``);
    every other streamed route replays the original chunks, so a mask must become a block there.
    A request this cannot read is treated as such a stream, so masking never fails open.
    """
    if input_type != "response":
        return False
    try:
        data: Final = _POLICY_REQUEST_DATA.validate_python(request_data)
    except ValidationError:
        return True
    body: Final = (data.get("proxy_server_request") or {}).get("body") or {}
    if data.get("stream") is not True and body.get("stream") is not True:
        return False
    routes: Final = (
        (data.get("metadata") or {}).get("user_api_key_request_route"),
        (data.get("litellm_metadata") or {}).get("user_api_key_request_route"),
    )
    route: Final = next((route for route in routes if route), None)
    call_types: Final = get_call_types_for_route(route) if route else None
    if not call_types:
        return True
    # Deferred: the translation handlers import the proxy, which imports the guardrail hooks.
    from litellm.llms import load_guardrail_translation_mappings  # noqa: PLC0415  # circular import
    from litellm.llms.openai.chat.guardrail_translation.handler import (  # noqa: PLC0415  # circular import
        OpenAIChatCompletionsHandler,
    )

    handler: Final[object] = load_guardrail_translation_mappings().get(call_types[0])
    return not (isinstance(handler, type) and issubclass(handler, OpenAIChatCompletionsHandler))


def _caller_email(data: _PolicyRequestData) -> str:
    buckets: Final = (data.get("litellm_metadata"), data.get("metadata"))  # new and legacy proxy metadata keys
    emails: Final = ((bucket or {}).get("user_api_key_user_email") or "" for bucket in buckets)
    return next((email.strip() for email in emails if email.strip()), "")


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
        beyondguard_application: str | None = None,
        beyondguard_destination_url: str | None = None,
        beyondguard_user_email_header: str | None = None,
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
        self.application = (beyondguard_application or "").strip()
        self.destination_url = (beyondguard_destination_url or "").strip()
        self.user_email_header = (beyondguard_user_email_header or "").strip() or _DEFAULT_USER_EMAIL_HEADER
        # Streamed responses are checked once, at end of stream. On chat completions a mask is written
        # into the stream as a diff; other streamed routes hold the text until the verdict and turn a
        # mask into a block (see _stream_cannot_carry_a_mask), so unmasked text is never released.
        self.streaming_transform_mode: Final[Literal["block_only", "incremental_diff"]] = "incremental_diff"
        self.streaming_end_of_stream_only: Final = True
        self.streaming_buffer_until_moderated: Final = True
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
            GuardrailEventHooks.post_call,
        ]

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = tuple(inputs.get("texts") or ())
        positions: Final = tuple(index for index, text in enumerate(texts) if text.strip())
        if not positions:
            return inputs
        target: Final = self.request_target if input_type == "request" else self.response_target
        checked: Final = tuple(texts[index] for index in positions)
        context: Final = self._request_context(request_data)

        verdict: Final = await self._check(target, "\n".join(checked), context)
        match verdict:
            case _Allow():
                return inputs
            case _Block() | _Failure():
                self._raise(verdict)
            case _Mask() if _stream_cannot_carry_a_mask(request_data, input_type):
                self._raise_unmaskable_stream()
            case _Mask(text=masked) if len(checked) == 1:
                return self._with_texts(inputs, texts, positions, (masked,))
            case _Mask():
                masked_each: Final = await self._mask_each(target, checked, context)
                return self._with_texts(inputs, texts, positions, masked_each)
            case _:
                assert_never(verdict)

    async def _mask_each(self, target: _Target, texts: tuple[str, ...], context: "_RequestContext") -> tuple[str, ...]:
        """Map a mask of the joined texts back to each text, or fail closed.

        The joined check found something to mask but cannot say where, so each text is
        re-checked. The rebuilt texts must then pass as a whole: a match that only exists
        across texts (masked jointly, allowed one by one) must never be forwarded unmasked.
        """
        if len(texts) > _MAX_MASK_SEGMENTS:
            self._raise(_Failure("too_many_texts_to_mask"))
        limit: Final = asyncio.Semaphore(_MASK_RECHECK_CONCURRENCY)

        async def recheck(text: str) -> _Verdict:
            async with limit:
                return await self._check(target, text, context)

        verdicts: Final = await asyncio.gather(*(recheck(text) for text in texts))
        for verdict in verdicts:
            if isinstance(verdict, _Block | _Failure):
                self._raise(verdict)
        masked: Final = tuple(
            verdict.text if isinstance(verdict, _Mask) else original
            for verdict, original in zip(verdicts, texts, strict=True)
        )
        match await self._check(target, "\n".join(masked), context):
            case _Allow():
                return masked
            case _Block() as blocked:
                self._raise(blocked)
            case _Failure() as failed:
                self._raise(failed)
            case _Mask():
                self._raise(_Failure("mask_not_mappable"))

    def _request_context(self, request_data: object) -> "_RequestContext":
        """Policy-selection headers for BeyondGuard: model, model URL, application and caller email."""
        data: Final = _policy_request_data(request_data)
        model_name: Final = (data.get("model") or "").strip()
        email: Final = _caller_email(data)
        optional: Final = (
            (_DESTINATION_MODEL_HEADER, model_name),
            (_DESTINATION_URL_HEADER, self.destination_url),
            (_APPLICATION_HEADER, self.application),
            (self.user_email_header, email),
        )
        return _RequestContext(
            model=model_name or GUARDRAIL_NAME,
            headers=MappingProxyType({name: value for name, value in optional if value}),
        )

    async def _check(self, target: _Target, text: str, context: "_RequestContext") -> _Verdict:
        url: Final = f"{self.api_base}/{target.endpoint_id}/chat/completions"
        try:
            response: Final = await self.http_client.post(
                url=url,
                json={"model": context.model, "messages": [{"role": "user", "content": text}]},
                headers={**context.headers, _TOKEN_HEADER: target.api_key, "Content-Type": "application/json"},
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
            case _Failure(reason="http_429"):
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
                    message="BeyondGuard rate limit exceeded",
                    should_wrap_with_default_message=False,
                    status_code=429,
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

    def _raise_unmaskable_stream(self) -> NoReturn:
        raise GuardrailRaisedException(
            guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
            message="Blocked by BeyondGuard (response needs masking, which this streamed route cannot apply)",
            should_wrap_with_default_message=False,
            blocked_content=True,
        )

    @staticmethod
    def _with_texts(
        inputs: GenericGuardrailAPIInputs,
        texts: tuple[str, ...],
        positions: tuple[int, ...],
        masked: tuple[str, ...],
    ) -> GenericGuardrailAPIInputs:
        replacements: Final = dict(zip(positions, masked, strict=True))
        return {**inputs, "texts": [replacements.get(index, text) for index, text in enumerate(texts)]}
