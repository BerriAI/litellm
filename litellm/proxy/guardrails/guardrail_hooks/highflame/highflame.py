import asyncio
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Optional, Protocol, TypeAlias

import httpx
from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict, Unpack

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException, Timeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    get_session_id_from_request_data,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME: Final = "highflame"

_DEFAULT_API_BASE: Final = "https://api.highflame.ai"
_DEFAULT_TOKEN_URL: Final = "https://auth.highflame.ai/oauth2/token"
_GUARD_PATH: Final = "/v1/shield/guard"
_DEFAULT_SHIELD_MODE: Final = "enforce"
_SHIELD_MODES: Final = frozenset({"enforce", "monitor", "alert", "modify"})
_DEFAULT_TIMEOUT_SECONDS: Final = 10.0
_TOKEN_REFRESH_MARGIN_SECONDS: Final = 30.0
_DEFAULT_TOKEN_LIFETIME_SECONDS: Final = 300.0
_MAX_CONCURRENT_EVALUATIONS: Final = 8

_DECISION_MODIFY: Final = "modify"
_PROCEED_DECISIONS: Final = frozenset({"allow", _DECISION_MODIFY})

_STATUS_UNAUTHORIZED: Final = 401
_STATUS_FORBIDDEN: Final = 403
_STATUS_TOO_EARLY: Final = 425
_STATUS_TOO_MANY_REQUESTS: Final = 429
_DEFER_PROBLEM_TYPE_MARKER: Final = "/defer/"

_JSON_OBJECT: Final = TypeAdapter(dict[str, object])

_SUPPORTED_EVENT_HOOKS: Final = (
    GuardrailEventHooks.pre_call,
    GuardrailEventHooks.during_call,
    GuardrailEventHooks.post_call,
    GuardrailEventHooks.pre_mcp_call,
    GuardrailEventHooks.post_mcp_call,
)


class _AsyncPoster(Protocol):
    async def post(
        self,
        url: str,
        *,
        json: dict[str, object] | None = None,  # mutable-ok: mirrors AsyncHTTPHandler.post's parameter type
        headers: dict[str, str] | None = None,  # mutable-ok: mirrors AsyncHTTPHandler.post's parameter type
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response: ...


class _CustomGuardrailOptions(TypedDict, total=False, extra_items=object):
    pass


class HighflameGuardrailMissingSecrets(Exception):
    pass


class HighflameGuardrailConfigurationError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _Evaluation:
    content: str
    content_type: Literal["prompt", "response", "tool_call"]
    action: Literal["process_prompt", "process_response", "call_tool"]
    text_index: int | None
    tool: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _Verdict:
    decision: str
    policy_reason: str | None
    redacted_content: str | None


@dataclass(frozen=True, slots=True)
class _Unreachable:
    reason: str


@dataclass(frozen=True, slots=True)
class _Misconfigured:
    reason: str


_Outcome: TypeAlias = _Verdict | _Unreachable | _Misconfigured


class HighflameGuardrail(CustomGuardrail):
    """
    Checks prompts, model responses, proposed tool calls, and MCP calls with Highflame Shield
    (`POST /v1/shield/guard`), then allows, redacts, or blocks based on Shield's decision
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        token_url: str | None = None,
        shield_mode: str | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        timeout: float | None = None,
        streaming_buffer_until_moderated: bool = True,
        async_handler: _AsyncPoster | None = None,
        **kwargs: Unpack[  # kwargs-ok: forwarded verbatim to CustomGuardrail.__init__, whose param list is wide and evolving
            _CustomGuardrailOptions
        ],
    ) -> None:
        resolved_api_key: Final = api_key or get_secret_str("HIGHFLAME_API_KEY")
        if not resolved_api_key:
            raise HighflameGuardrailMissingSecrets(
                "Highflame API key is required. Set the `HIGHFLAME_API_KEY` environment variable or "
                "pass `api_key` in the guardrail config."
            )
        resolved_mode: Final = shield_mode or _DEFAULT_SHIELD_MODE
        if resolved_mode not in _SHIELD_MODES:
            raise ValueError(
                f"Highflame guardrail: shield_mode must be one of {sorted(_SHIELD_MODES)}, got {resolved_mode!r}"
            )
        base: Final = (api_base or get_secret_str("HIGHFLAME_API_BASE") or _DEFAULT_API_BASE).rstrip("/")

        self.async_handler: _AsyncPoster = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self._api_key: str = resolved_api_key
        self.guard_url: str = f"{base}{_GUARD_PATH}"
        self.token_url: str = token_url or get_secret_str("HIGHFLAME_TOKEN_URL") or _DEFAULT_TOKEN_URL
        self.shield_mode: str = resolved_mode
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = unreachable_fallback
        self.streaming_buffer_until_moderated: bool = streaming_buffer_until_moderated
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        self._token_lock: asyncio.Lock = asyncio.Lock()

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = self.get_supported_event_hooks()  # rebind-ok: per-call kwargs dict

        super().__init__(timeout=timeout if timeout is not None else _DEFAULT_TIMEOUT_SECONDS, **kwargs)

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: parent's signature
        return list(_SUPPORTED_EVENT_HOOKS)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: overrides CustomGuardrail.apply_guardrail's plain-dict contract
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        evaluations: Final = _plan_evaluations(
            inputs, request_data, input_type, streamed=request_data.get("stream") is True
        )
        if not evaluations:
            return inputs

        token: Final = await self._token()
        if isinstance(token, _Unreachable):
            return self._on_unreachable(token, inputs)
        if isinstance(token, _Misconfigured):
            raise HighflameGuardrailConfigurationError(token.reason)

        session_id: Final = get_session_id_from_request_data(request_data)
        semaphore: Final = asyncio.Semaphore(_MAX_CONCURRENT_EVALUATIONS)

        async def bounded(evaluation: _Evaluation) -> _Outcome:
            async with semaphore:
                return await self._evaluate(evaluation, session_id, token)

        outcomes: Final = await asyncio.gather(*(bounded(e) for e in evaluations))
        return self._enforce(inputs, tuple(zip(evaluations, outcomes, strict=True)))

    def _enforce(
        self,
        inputs: GenericGuardrailAPIInputs,
        results: Sequence[tuple[_Evaluation, _Outcome]],
    ) -> GenericGuardrailAPIInputs:
        verdicts: Final = tuple((e, o) for e, o in results if isinstance(o, _Verdict))
        stopped: Final = next(((e, v) for e, v in verdicts if v.decision not in _PROCEED_DECISIONS), None)
        if stopped is not None:
            raise _blocked(*stopped)

        misconfigured: Final = next((o for _, o in results if isinstance(o, _Misconfigured)), None)
        if misconfigured is not None:
            raise HighflameGuardrailConfigurationError(misconfigured.reason)

        rewritten: Final = _apply_redactions(inputs, verdicts)
        unreachable: Final = next((o for _, o in results if isinstance(o, _Unreachable)), None)
        return rewritten if unreachable is None else self._on_unreachable(unreachable, rewritten)

    async def _evaluate(self, evaluation: _Evaluation, session_id: str | None, token: str) -> _Outcome:
        body: Final = self._guard_body(evaluation, session_id)
        response: Final = await self._post(self.guard_url, body, token)
        if isinstance(response, _Unreachable):
            return response
        if not _is_token_rejection(response):
            return _read_verdict(response)

        # A 401 that carries no decision is an expired or revoked token, while a step-up challenge
        # is a 401 that carries one. Exchange the key again and retry once
        self._forget_token(token)
        fresh: Final = await self._token()
        if not isinstance(fresh, str):
            return fresh
        retried: Final = await self._post(self.guard_url, body, fresh)
        return retried if isinstance(retried, _Unreachable) else _read_verdict(retried)

    def _guard_body(self, evaluation: _Evaluation, session_id: str | None) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "content": evaluation.content,
                "content_type": evaluation.content_type,
                "action": evaluation.action,
                "mode": self.shield_mode,
                **({"session_id": session_id} if session_id else {}),
                **({"tool": dict(evaluation.tool)} if evaluation.tool is not None else {}),
            }
        )

    async def _token(self) -> str | _Unreachable | _Misconfigured:
        cached: Final = self._cached_token()
        if cached is not None:
            return cached
        async with self._token_lock:
            refreshed: Final = self._cached_token()
            return refreshed if refreshed is not None else await self._exchange_token()

    def _cached_token(self) -> str | None:
        if self._access_token is not None and time.monotonic() < self._token_expires_at:
            return self._access_token
        return None

    def _forget_token(self, token: str) -> None:
        if self._access_token == token:
            self._access_token = None

    async def _exchange_token(self) -> str | _Unreachable | _Misconfigured:
        response: Final = await self._post(
            self.token_url, MappingProxyType({"grant_type": "api_key", "api_key": self._api_key}), token=None
        )
        if isinstance(response, _Unreachable):
            return response
        status: Final = response.status_code
        if _is_client_error(status):
            return _Misconfigured(
                f"Highflame rejected the service key at {self.token_url} (HTTP {status}). "
                "Check `api_key` and `token_url` in the guardrail config."
            )
        payload: Final = _body(response)
        access_token: Final = payload.get("access_token") if payload is not None else None
        if not response.is_success or not isinstance(access_token, str) or not access_token:
            return _Unreachable(f"Highflame token endpoint returned HTTP {status} without an access token")

        expires_in: Final = payload.get("expires_in") if payload is not None else None
        lifetime: Final = (
            float(expires_in)
            if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool)
            else _DEFAULT_TOKEN_LIFETIME_SECONDS
        )
        self._access_token = access_token
        self._token_expires_at = time.monotonic() + max(lifetime - _TOKEN_REFRESH_MARGIN_SECONDS, 0.0)
        return access_token

    async def _post(self, url: str, body: Mapping[str, object], token: str | None) -> httpx.Response | _Unreachable:
        headers: Final = {"Content-Type": "application/json"} | (
            {"Authorization": f"Bearer {token}"} if token is not None else {}
        )
        try:
            return await self.async_handler.post(url=url, json=dict(body), headers=headers, timeout=self.timeout)
        except httpx.HTTPStatusError as e:
            # The shared client raises on every non-2xx status, and Shield sends its step-up and
            # defer decisions as 401 and 425 responses, so the response is kept and read
            return e.response
        except (Timeout, httpx.TimeoutException, httpx.RequestError) as e:
            return _Unreachable(f"Highflame unreachable at {url}: {e}")

    def _on_unreachable(
        self, unreachable: _Unreachable, inputs: GenericGuardrailAPIInputs
    ) -> GenericGuardrailAPIInputs:
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.critical(
                "Highflame guardrail unreachable, allowing request per unreachable_fallback: %s",
                unreachable.reason,
            )
            return inputs
        raise GuardrailRaisedException(
            guardrail_name=GUARDRAIL_NAME,
            message="Highflame guardrail is unavailable and this request cannot be checked",
            should_wrap_with_default_message=False,
        )

    @staticmethod
    def get_config_model() -> type | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.highflame import (
            HighflameGuardrailConfigModel,
        )

        return HighflameGuardrailConfigModel


def _plan_evaluations(
    inputs: GenericGuardrailAPIInputs,
    request_data: Mapping[str, object],
    input_type: Literal["request", "response"],
    *,
    streamed: bool,
) -> tuple[_Evaluation, ...]:
    is_request: Final = input_type == "request"
    # LiteLLM's streaming path drops text rewrites and replays the original chunks, so a redaction
    # of a streamed response cannot be applied and is refused instead
    rewritable: Final = is_request or not streamed
    texts: Final = tuple(
        _Evaluation(
            content=text,
            content_type="prompt" if is_request else "response",
            action="process_prompt" if is_request else "process_response",
            text_index=index if rewritable else None,
        )
        for index, text in enumerate(inputs.get("texts") or [])
        if text
    )
    tool_calls: Final = tuple(
        _tool_call_evaluation(tool_call, proposed=not is_request) for tool_call in inputs.get("tool_calls") or []
    )
    mcp_tool_name: Final = request_data.get("mcp_tool_name")
    mcp_call: Final = (
        (_mcp_call_evaluation(mcp_tool_name, request_data),)
        if is_request and isinstance(mcp_tool_name, str) and mcp_tool_name
        else ()
    )
    return texts + tuple(e for e in tool_calls if e is not None) + mcp_call


def _tool_call_evaluation(tool_call: object, *, proposed: bool) -> _Evaluation | None:
    function: Final = _field(tool_call, "function")
    name: Final = _field(function, "name")
    if not isinstance(name, str) or not name:
        return None
    raw_arguments: Final = _field(function, "arguments")
    arguments_text: Final = (
        raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments or {}, default=str)
    )
    if not proposed:
        # Tool calls in the request are client-supplied history, so their arguments are checked as
        # untrusted prompt content rather than as a new call
        return _Evaluation(content=arguments_text, content_type="prompt", action="process_prompt", text_index=None)
    arguments: Final = _parse_json_object(arguments_text)
    return _Evaluation(
        content=arguments_text,
        content_type="tool_call",
        action="call_tool",
        text_index=None,
        tool=MappingProxyType({"name": name, **({"arguments": dict(arguments)} if arguments is not None else {})}),
    )


def _mcp_call_evaluation(name: str, request_data: Mapping[str, object]) -> _Evaluation:
    arguments: Final = _as_json_object(request_data.get("mcp_arguments"))
    description: Final = request_data.get("mcp_tool_description")
    server: Final = request_data.get("mcp_server_name")
    return _Evaluation(
        content=json.dumps(dict(arguments), default=str) if arguments is not None else name,
        content_type="tool_call",
        action="call_tool",
        text_index=None,
        tool=MappingProxyType(
            {
                "name": name,
                **({"arguments": dict(arguments)} if arguments is not None else {}),
                **({"description": description} if isinstance(description, str) and description else {}),
                **({"server_id": server} if isinstance(server, str) and server else {}),
            }
        ),
    )


def _apply_redactions(
    inputs: GenericGuardrailAPIInputs,
    verdicts: Sequence[tuple[_Evaluation, _Verdict]],
) -> GenericGuardrailAPIInputs:
    modified: Final = tuple((e, v) for e, v in verdicts if v.decision == _DECISION_MODIFY)
    if not modified:
        return inputs
    rewrites: Final = dict(_rewrites(modified))
    texts: Final = inputs.get("texts") or []
    return {**inputs, "texts": [rewrites.get(index, text) for index, text in enumerate(texts)]}


def _rewrites(modified: Sequence[tuple[_Evaluation, _Verdict]]) -> Iterator[tuple[int, str]]:
    for evaluation, verdict in modified:
        if evaluation.text_index is None or verdict.redacted_content is None:
            raise _blocked(evaluation, verdict)
        yield evaluation.text_index, verdict.redacted_content


def _blocked(evaluation: _Evaluation, verdict: _Verdict) -> GuardrailRaisedException:
    subject: Final = {"prompt": "request", "response": "response", "tool_call": "tool call"}[evaluation.content_type]
    return GuardrailRaisedException(
        guardrail_name=GUARDRAIL_NAME,
        message=f"Highflame blocked this {subject}: {verdict.policy_reason or f'decision={verdict.decision}'}",
        should_wrap_with_default_message=False,
        blocked_content=True,
    )


def _read_verdict(response: httpx.Response) -> _Outcome:
    status: Final = response.status_code
    payload: Final = _body(response)
    decision: Final = _decision(payload)
    if payload is not None and decision is not None:
        reason: Final = payload.get("policy_reason")
        redacted: Final = payload.get("redacted_content")
        return _Verdict(
            decision=decision,
            policy_reason=reason if isinstance(reason, str) and reason else None,
            redacted_content=redacted if isinstance(redacted, str) else None,
        )
    if status == _STATUS_TOO_EARLY or (status == _STATUS_FORBIDDEN and _is_defer_problem(payload)):
        return _Verdict(decision="defer", policy_reason=_problem_detail(payload), redacted_content=None)
    if status == _STATUS_TOO_MANY_REQUESTS or status >= 500:
        return _Unreachable(f"Highflame Shield returned HTTP {status}")
    if _is_client_error(status):
        return _Misconfigured(
            f"Highflame Shield refused the guard request (HTTP {status}): "
            f"{_problem_detail(payload) or response.text[:200]}"
        )
    return _Unreachable(f"Highflame Shield returned HTTP {status} without a decision")


def _is_token_rejection(response: httpx.Response) -> bool:
    return response.status_code == _STATUS_UNAUTHORIZED and _decision(_body(response)) is None


def _body(response: httpx.Response) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_json(response.content)
    except ValidationError:
        return None


def _decision(payload: Mapping[str, object] | None) -> str | None:
    decision: Final = payload.get("decision") if payload is not None else None
    return decision.lower() if isinstance(decision, str) and decision else None


def _is_defer_problem(payload: Mapping[str, object] | None) -> bool:
    problem_type: Final = payload.get("type") if payload is not None else None
    return isinstance(problem_type, str) and _DEFER_PROBLEM_TYPE_MARKER in problem_type


def _problem_detail(payload: Mapping[str, object] | None) -> str | None:
    if payload is None:
        return None
    details: Final = (payload.get(key) for key in ("policy_reason", "detail", "title", "message"))
    return next((d for d in details if isinstance(d, str) and d), None)


def _field(value: object, name: str) -> object:
    mapping: Final = _as_json_object(value)
    return mapping.get(name) if mapping is not None else getattr(value, name, None)


def _parse_json_object(text: str) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_json(text)
    except ValidationError:
        return None


def _as_json_object(value: object) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _is_client_error(status: int) -> bool:
    return 400 <= status < 500 and status != _STATUS_TOO_MANY_REQUESTS
