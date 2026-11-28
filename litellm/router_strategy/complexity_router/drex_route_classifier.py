import asyncio
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import suppress
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Literal, NamedTuple, TypeAlias

import anyio
import httpcore
import httpx
from pydantic import ConfigDict, Field, SecretStr, StrictFloat, TypeAdapter, ValidationError, model_validator

from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import request_contains_image_content
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared HTTP factory has an untyped params argument
)
from litellm.llms.nace.decisions.transformation import NaceDecisionsConfig
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.types.router import RoutingContext

_ROUTE_PATH: Final = "/v1/route"
_MAX_ROUTE_MESSAGES: Final = 1024
_MAX_ROUTE_CANDIDATES: Final = 255
_MAX_RESPONSE_BYTES: Final = 65_536
_ROUTE_ROLES: Final[Mapping[str, str]] = MappingProxyType(
    {"system": "system", "developer": "system", "user": "user", "assistant": "assistant"}
)
_HOSTED_TOOL_TEXT_PREFIX: Final = "Hosted web search: "
_HOSTED_TOOL_TEXT_PREFIX_LEN: Final = len(_HOSTED_TOOL_TEXT_PREFIX)
# litellm's Responses-API conversion replays a converted web_search_call item as assistant
# content starting with this prefix (litellm/responses/litellm_completion_transformation/
# transformation.py); converted tool traffic is not conversation text, so the projection
# drops it. Only assistant messages are filtered: a user who literally types the prefix
# is making a request, not replaying a tool call.
_NACE: Final = NaceDecisionsConfig()


class DrexRouteError(Exception):
    """Failure raised by DrexRouteClassifier. The message is fixed text plus numbers only,
    never a key, header, request body, or response body."""


class DrexRouteTimeoutError(DrexRouteError, TimeoutError):
    """The Drex call exceeded its timeout. Subclasses TimeoutError so the complexity router
    records failure reason 'timeout' instead of 'classifier_error'."""


class DrexRouteConfigError(Exception):
    """Invalid DrexRouteClassifier configuration. Raised instead of a pydantic
    ValidationError for model-level rules, because pydantic echoes the raw input into its
    error text and the input can carry the API key."""


class _EligibleTurn(NamedTuple):
    """One message projected to routing text. `text` is the whole turn when it fits the
    budget and the bounded character tail when it does not."""

    role: str
    text: str
    oversized: bool


def _malformed_response_error() -> ValidationError:
    """A clean ValidationError (input_value=None) so the complexity router records failure
    reason 'invalid_response' for malformed Drex bodies without echoing the body."""
    return ValidationError.from_exception_data(
        "DrexRouteResponse",
        [
            {
                "type": "value_error",
                "loc": ("response",),
                "input": None,
                "ctx": {"error": ValueError("drex route response did not match the expected shape")},
            }
        ],
    )


class DrexRouteCandidateProfile(LiteLLMBaseModel):
    """A full Drex Router candidate: any model ID Drex does not already know."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=256)
    provider: str = Field(min_length=1, max_length=128)
    price_in: float = Field(ge=0)
    price_out: float = Field(ge=0)
    context: int = Field(gt=0)
    tags: tuple[str, ...] = Field(default=(), max_length=64)


DrexRouteCandidate: TypeAlias = str | DrexRouteCandidateProfile


class DrexRouteClassifierConfig(LiteLLMBaseModel):
    """Tier-to-candidate wiring for DrexRouteClassifier. Dict keys are tier names, in the
    order the candidates are offered to Drex."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: Mapping[str, DrexRouteCandidate]
    selection_priority: Literal["quality", "balanced", "cheapest"] = "balanced"
    vision_tiers: frozenset[str] | None = None
    api_key: SecretStr | None = None
    api_base: str | None = None
    timeout_s: float = Field(default=2.5, gt=0, le=600, allow_inf_nan=False)
    cleanup_grace_s: float = Field(default=1.0, gt=0, le=60, allow_inf_nan=False)
    max_detached_exchanges: int = Field(default=4, ge=1, le=64)
    max_input_bytes: int = Field(default=24_000, gt=0)

    def __init__(
        self,
        **data: object,  # kwargs-ok: a pydantic model is constructed from its field names as keywords
    ) -> None:
        """Pydantic echoes the raw input into every field error, so a plaintext key never
        reaches it: it is converted to SecretStr (or rejected) before validation, and the raw
        input leaves this frame before anything is raised, so no traceback holds it."""
        key: Final = _safe_key(data.get("api_key"))
        fields: Final = {name: value for name, value in data.items() if name != "api_key"}
        del data  # rebind-ok: the raw key leaves this frame before anything can raise
        if isinstance(key, DrexRouteConfigError):
            raise key
        super().__init__(**fields, api_key=key)

    @model_validator(mode="after")
    def _validate_candidates(self) -> "DrexRouteClassifierConfig":
        if not 1 <= len(self.candidates) <= _MAX_ROUTE_CANDIDATES:
            raise DrexRouteConfigError(
                f"candidates must map between 1 and {_MAX_ROUTE_CANDIDATES} tiers, got {len(self.candidates)}"
            )
        ids: Final = tuple(_candidate_id(candidate) for candidate in self.candidates.values())
        duplicated: Final = sorted(frozenset(candidate_id for candidate_id in ids if ids.count(candidate_id) > 1))
        if duplicated:
            raise DrexRouteConfigError(
                f"candidate IDs must be unique across tiers, duplicated: {', '.join(duplicated)}"
            )
        if self.vision_tiers is not None:
            unknown: Final = tuple(sorted(frozenset(self.vision_tiers) - frozenset(self.candidates)))
            if unknown:
                raise DrexRouteConfigError(f"vision_tiers names tiers with no candidate: {', '.join(unknown)}")
        return self


class _RouteResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    selected: str
    probabilities: Mapping[str, Annotated[StrictFloat, Field(ge=0.0, le=1.0)]]
    request_id: str | None = None


def _parse_route_response(body: bytes) -> _RouteResponse | ValidationError:
    try:
        return _RouteResponse.model_validate_json(body)
    except ValidationError:
        return _malformed_response_error()


def _candidate_id(candidate: DrexRouteCandidate) -> str:
    return candidate if isinstance(candidate, str) else candidate.id


def _wire_candidate(candidate: DrexRouteCandidate) -> str | Mapping[str, object]:
    return candidate if isinstance(candidate, str) else candidate.model_dump(mode="json")


def _secret_value(secret: SecretStr | None) -> str | None:
    return secret.get_secret_value() if secret is not None else None


# RFC 6750 b64token, the Bearer credential syntax. Anything else (a space, a control or
# non-ASCII character) can be rejected by the HTTP stack, and httpcore logs the rejected
# header value at debug level before any error reaches this module.
_BEARER_TOKEN: Final = re.compile(r"[A-Za-z0-9\-._~+/]+=*")


def _safe_key(value: object) -> SecretStr | DrexRouteConfigError | None:
    """The config key as a SecretStr, with an empty string counting as unset. A rejection
    is returned, not raised, so callers raise it from a frame that no longer holds the key."""
    if value is None:
        return None
    if isinstance(value, SecretStr):
        return _safe_key(value.get_secret_value())
    if isinstance(value, str):
        return _header_safe_key(value) if value else None
    return DrexRouteConfigError("api_key must be a string when set")


def _header_safe_key(key: str) -> SecretStr | DrexRouteConfigError:
    """The key as a SecretStr once it is known to be a Bearer token. Checked at construction
    so a bad key never reaches header encoding or the transport, whose errors and debug logs
    carry the whole header value; the error names the rule, never the key."""
    if _BEARER_TOKEN.fullmatch(key) is None:
        return DrexRouteConfigError("api_key must be a Bearer token: letters, digits, -._~+/ and trailing =")
    return SecretStr(key)


def _validated_config(config: Mapping[str, object]) -> DrexRouteClassifierConfig | Exception:
    """Validate a caller-supplied mapping without ever handing a plaintext key to pydantic:
    model_validate bypasses __init__, so the key is scrubbed here and re-attached after. A
    failure is returned without its traceback, whose frames hold the mapping and its key."""
    key: Final = _safe_key(config.get("api_key"))
    if isinstance(key, DrexRouteConfigError):
        return key
    keyless: Final = {name: value for name, value in config.items() if name != "api_key"}
    try:
        return DrexRouteClassifierConfig.model_validate(keyless).model_copy(update={"api_key": key})
    except (ValidationError, DrexRouteConfigError) as e:
        return e.with_traceback(None)


def _shared_nace_handler() -> AsyncHTTPHandler:
    """LiteLLM's shared, event-loop-scoped NACE handler: pooled, cookie-blocked,
    configured with litellm's TLS/proxy settings, and closed by litellm's own cache
    eviction and async-client shutdown."""
    return get_async_httpx_client(llm_provider=LlmProviders.NACE)


def _handler_factory(http_client: AsyncHTTPHandler | None) -> Callable[[], AsyncHTTPHandler]:
    if http_client is None:
        return _shared_nace_handler
    return lambda: http_client


_CONTENT_PARTS: Final = TypeAdapter(tuple[object, ...])


def _text_part(part: object) -> str | None:
    match part:
        case {"type": "text", "text": str() as text}:
            return text
        case _:
            return None


def _joined_head(texts: tuple[str, ...]) -> str:
    """The opening characters of the joined parts, bounded to the hosted-tool prefix."""
    head = ""  # rebind-ok: bounded prefix accumulation
    for part_text in texts:
        if len(head) >= _HOSTED_TOOL_TEXT_PREFIX_LEN:
            break
        head = head + part_text[: _HOSTED_TOOL_TEXT_PREFIX_LEN - len(head)]
    return head


def _joined_tail(texts: tuple[str, ...], max_bytes: int) -> str:
    """The last max_bytes characters of the joined parts, without joining the whole turn:
    parts are scanned from the end and only the window that can reach the tail is joined."""
    tail_size = 0  # rebind-ok: characters accumulated from the end of the turn
    kept_from = len(texts)  # rebind-ok: first index of the tail window
    straddle = ""  # rebind-ok: the sliced part that spans the rest of the budget
    for index in range(len(texts) - 1, -1, -1):
        part_text = texts[index]
        remaining = max_bytes - tail_size
        if len(part_text) >= remaining:
            straddle = part_text[-remaining:]
            kept_from = index + 1
            break
        tail_size += len(part_text)
        kept_from = index
    return "".join((straddle, *texts[kept_from:]))[-max_bytes:]


def _turn_from_text(mapped: str, head: str, text: str, oversized: bool) -> _EligibleTurn | None:
    """`text` is always bounded by the budget; `head` carries the opening characters for
    the hosted-tool check. An oversized turn whose tail is blank is dropped, exactly as a
    truncated blank turn produced no routing text before."""
    if not text.strip():
        return None
    if mapped == "assistant" and head.startswith(_HOSTED_TOOL_TEXT_PREFIX):
        return None
    return _EligibleTurn(role=mapped, text=text, oversized=oversized)


def _route_turn(message: Mapping[str, object], max_bytes: int) -> _EligibleTurn | None:
    """The message as a routing turn, or None when it carries no conversation text.
    Ineligible roles are decided with a role lookup alone, and multipart content is
    processed within the byte budget, so no turn is joined or encoded whole."""
    role: Final = message.get("role")
    if not isinstance(role, str):
        return None
    mapped: Final = _ROUTE_ROLES.get(role)
    if mapped is None:
        return None
    content: Final = message.get("content")
    if isinstance(content, str):
        oversized: Final = len(content) > max_bytes
        return _turn_from_text(
            mapped,
            content[:_HOSTED_TOOL_TEXT_PREFIX_LEN],
            content[-max_bytes:] if oversized else content,
            oversized,
        )
    if not isinstance(content, list):
        return None
    parts: Final[tuple[object, ...]] = _CONTENT_PARTS.validate_python(content)
    texts: Final = tuple(part_text for part in parts if (part_text := _text_part(part)) is not None)
    total: Final = sum(len(part_text) for part_text in texts)
    if total <= max_bytes:
        joined: Final = "".join(texts)
        return _turn_from_text(mapped, joined, joined, False)
    return _turn_from_text(mapped, _joined_head(texts), _joined_tail(texts, max_bytes), True)


def _route_turns(messages: Sequence[Mapping[str, object]], max_bytes: int) -> tuple[Mapping[str, str], ...]:
    """The newest conversation turns that fit the byte budget, oldest-first. The history
    is walked newest-first; ineligible messages cost one role lookup each, so a long tool
    history never hides the newest eligible turns."""
    kept: tuple[Mapping[str, str], ...] = ()  # rebind-ok: newest-first projection bounded by the message cap
    received = 0  # rebind-ok: running UTF-8 size of the kept turns, enforcing the byte budget
    for message in reversed(messages):
        if len(kept) >= _MAX_ROUTE_MESSAGES:
            break
        turn = _route_turn(message, max_bytes)
        if turn is None:
            continue
        if turn.oversized:
            if not kept:
                tail = turn.text.encode()[-max_bytes:].decode(errors="ignore")
                kept = ({"role": turn.role, "content": tail},) if tail.strip() else ()
            break
        encoded = turn.text.encode()
        received += len(encoded)
        if received > max_bytes:
            if not kept:
                tail = encoded[-max_bytes:].decode(errors="ignore")
                kept = ({"role": turn.role, "content": tail},) if tail.strip() else ()
            break
        kept = (*kept, {"role": turn.role, "content": turn.text})
    return tuple(reversed(kept))


def _projected_turns(
    messages: Sequence[Mapping[str, object]], max_bytes: int
) -> tuple[Mapping[str, str], ...] | DrexRouteError:
    """The routing projection, or a sanitized error. Encoding a turn that holds a lone
    surrogate fails with the whole turn text attached, so no projection error leaves here."""
    try:
        return _route_turns(messages, max_bytes)
    except Exception as e:  # noqa: BLE001  # any projection failure is sanitized so no input text reaches router logs
        return DrexRouteError(f"drex route request could not be prepared ({type(e).__name__})")


_SEND_HEADERS_STARTED: Final = ".send_request_headers.started"
_PROXY_CONNECT_METHOD: Final = b"CONNECT"
# Strong references to exchanges left closing past the cleanup grace, dropped as each one finishes.
_DETACHED_EXCHANGES: Final[set["asyncio.Task[_RouteResponse]"]] = set()  # mutable-ok: see above


class _RetryRefusedError(Exception):
    """httpcore was about to send the route request a second time."""


def _header_send_is_proxy_connect(info: Mapping[str, object]) -> bool:
    """True when this header send is the proxy CONNECT that opens the tunnel.

    httpcore copies the caller's trace extension onto that CONNECT and reports it as
    `http11.send_request_headers.started`, the same event an origin HTTP/1.1 POST uses.
    The event prefix cannot tell them apart. The request method on the trace info can."""
    request: Final = info.get("request")
    return isinstance(request, httpcore.Request) and request.method == _PROXY_CONNECT_METHOD


class _SingleAttempt:
    """Per-request trace hook that keeps the exchange to one wire attempt. httpcore's pool
    resends a request on a new connection when an HTTP/2 peer's GOAWAY says it did not
    process the stream. httpcore reports each send of request headers through the
    per-request trace extension, and raising there aborts the second send before it starts.
    A proxy CONNECT reuses that event for tunnel setup and does not consume the one send."""

    def __init__(self) -> None:
        self.sends: int = 0

    async def trace(self, event: str, info: Mapping[str, object]) -> None:
        """Count origin request-header sends. httpcore calls this for every traced step."""
        if not event.endswith(_SEND_HEADERS_STARTED) or _header_send_is_proxy_connect(info):
            return
        if self.sends:
            raise _RetryRefusedError
        self.sends += 1


class _ReleasingStream(httpx.AsyncByteStream):
    """Wraps the response stream so every close of it, including the one HTTPX runs itself
    at end of body, finishes even when the exchange is being stopped: HTTPX marks a response
    closed before it awaits the transport's close, so an interrupted close could never be
    retried. httpcore shields its own internal closes the same way."""

    def __init__(self, stream: httpx.AsyncByteStream) -> None:
        self._stream: Final = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        with anyio.CancelScope(shield=True):
            await self._stream.aclose()


async def _read_capped(response: httpx.Response, max_bytes: int) -> bytes | None:
    """The response body, or None when it streamed past the cap; never buffers more. The
    request asks for identity encoding, so the capped bytes are the wire bytes."""
    parts: tuple[bytes, ...] = ()  # rebind-ok: bounded streaming accumulation under the cap
    received = 0  # rebind-ok: running byte count that enforces the cap while streaming
    async for chunk in response.aiter_bytes():
        received += len(chunk)
        if received > max_bytes:
            return None
        parts = (*parts, chunk)
    return b"".join(parts)


async def _exchange(
    client: httpx.AsyncClient, request: httpx.Request, timeout_s: float
) -> tuple[int, bytes | None] | DrexRouteError:
    """One streamed exchange. The request goes straight to the client's transport for its
    URL, so the response stream is wrapped before anything can close it: AsyncClient.send
    runs redirect and hook handling that can close a response it has not returned yet.
    Failures are returned, not raised, so the caller raises the sanitized error outside any
    except block and no original exception is chained onto it."""
    try:
        if client.is_closed:
            return DrexRouteError("drex route call failed (client closed)")
        transport: Final = client._transport_for_url(  # pyright: ignore[reportPrivateUsage]  # HTTPX exposes no public per-URL transport lookup; send() would close responses outside the wrapper
            request.url
        )
        response: Final = await transport.handle_async_request(request)
        response.request = request
        if isinstance(response.stream, httpx.AsyncByteStream):
            response.stream = _ReleasingStream(response.stream)
        try:
            encoding: Final = dict(response.headers).get("content-encoding", "identity").strip().lower()
            if encoding not in ("", "identity"):
                return DrexRouteError("drex route response used an unsupported content encoding")
            return response.status_code, await _read_capped(response, _MAX_RESPONSE_BYTES)
        finally:
            await response.aclose()
    except httpx.TimeoutException:
        return DrexRouteTimeoutError(f"drex route call timed out after {timeout_s} seconds")
    except Exception as e:  # noqa: BLE001  # any other failure is sanitized so no remote content reaches router logs
        return DrexRouteError(f"drex route call failed ({type(e).__name__})")


async def _post_route(
    client: httpx.AsyncClient, request: httpx.Request, timeout_s: float, scope: anyio.CancelScope
) -> _RouteResponse:
    """The exchange task. It runs inside `scope`, which the caller cancels to stop it: anyio
    delivers that cancellation at the exchange's next suspension outside a shielded close, so
    a close always finishes and nothing is sent after the stop. A CancelledError from
    anywhere else propagates and ends the task cancelled; the caller reads that through
    _exchange_outcome and never re-raises the task's own exception objects, so nothing they
    chain to can reach the caller."""
    stopped: Final = DrexRouteError("drex route call was stopped before completing")
    outcome: tuple[int, bytes | None] | DrexRouteError = stopped  # rebind-ok: replaced unless the scope stops it
    with scope:
        outcome = await _exchange(client, request, timeout_s)  # rebind-ok: see above
    if isinstance(outcome, DrexRouteError):
        raise outcome
    status, body = outcome
    if status != 200:
        raise DrexRouteError(f"drex route call returned status {status}")
    if body is None:
        raise DrexRouteError(f"drex route response exceeded {_MAX_RESPONSE_BYTES} bytes")
    parsed: Final = _parse_route_response(body)
    if isinstance(parsed, ValidationError):
        raise parsed
    return parsed


def _unchained(error: Exception) -> Exception:
    """The error with no reachable cause or context, raised by the caller outside any except
    block so Python does not attach a new context either."""
    error.__cause__ = None  # rebind-ok: the chain is cleared on the error itself; a copy could not keep its type
    error.__context__ = None  # rebind-ok: as above
    return error


def _exchange_outcome(exchange: "asyncio.Task[_RouteResponse]") -> _RouteResponse | Exception:
    """The finished exchange's answer, or its failure without any exception chain. A task
    that ended cancelled although the caller never cancelled it (the transport raised
    CancelledError, or cancelled its own task and returned without suspending) is an
    internal cancellation: it becomes a fresh error and falls back instead of surfacing as a
    caller cancellation."""
    if exchange.cancelled():
        return DrexRouteError("drex route call was cancelled before completing")
    failure: Final = exchange.exception()
    if failure is None:
        return exchange.result()
    if not isinstance(failure, Exception):
        raise failure
    return _unchained(failure)


def _release_detached(exchange: "asyncio.Task[_RouteResponse]") -> None:
    _DETACHED_EXCHANGES.discard(exchange)
    if not exchange.cancelled():
        _ = exchange.exception()


def _detached_on(loop: asyncio.AbstractEventLoop) -> int:
    """Detached exchanges still releasing on `loop`. Those of a closed loop never finish
    and are dropped here, so they never count against a later loop."""
    for exchange in tuple(_DETACHED_EXCHANGES):
        if exchange.get_loop().is_closed():
            _DETACHED_EXCHANGES.discard(exchange)
    return sum(1 for exchange in _DETACHED_EXCHANGES if exchange.get_loop() is loop)


async def _stop_exchange(exchange: "asyncio.Task[_RouteResponse]", scope: anyio.CancelScope, grace_s: float) -> None:
    """Stops the exchange for a caller cancellation or the plugin deadline by cancelling its
    scope; later cancellations of the caller are absorbed here instead of being forwarded
    into transport cleanup, and the caller re-raises its first one. If the exchange is still
    running after grace_s (a shielded close that never finishes), it is detached with a
    strong reference so the cancellation can proceed, and it finishes on its own;
    max_detached_exchanges bounds how many can be pending at once."""
    scope.cancel()
    loop: Final = asyncio.get_running_loop()
    give_up_at: Final = loop.time() + grace_s
    while not exchange.done():
        remaining = give_up_at - loop.time()
        if remaining <= 0:
            _DETACHED_EXCHANGES.add(exchange)
            exchange.add_done_callback(_release_detached)
            verbose_router_logger.debug(
                "Drex Router exchange still releasing its response after %ss cleanup grace, detached", grace_s
            )
            return
        with suppress(asyncio.CancelledError):
            _ = await asyncio.wait((exchange,), timeout=remaining)
    with suppress(asyncio.CancelledError, Exception):
        await exchange  # already done: retrieves its outcome so asyncio never logs it as unretrieved


async def _pending_cancellation() -> asyncio.CancelledError | None:
    """A caller cancellation requested in the same loop turn as the exchange's own
    failure, probed with a zero-length yield."""
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as pending:
        return pending
    return None


def _route_request(
    client: httpx.AsyncClient,
    url: str,
    body: Mapping[str, object],
    api_key: SecretStr,
    timeout_s: float,
    attempt: _SingleAttempt,
) -> httpx.Request | DrexRouteError:
    """The prepared request, or a sanitized error. Header and body encoding errors carry the
    whole header value or the whole message text, so none of them leaves this function."""
    try:
        return client.build_request(
            "POST",
            url,
            json=body,
            headers={
                **dict(_NACE.validate_environment({}, "", api_key.get_secret_value(), {})),
                "Accept-Encoding": "identity",
            },
            timeout=timeout_s,
            extensions={"trace": attempt.trace},
        )
    except Exception as e:  # noqa: BLE001  # any preparation failure is sanitized so no key or input reaches router logs
        return DrexRouteError(f"drex route request could not be prepared ({type(e).__name__})")


class DrexRouteClassifier:
    """ClassifierPlugin that asks Drex Router `POST /v1/route` which configured candidate
    should serve the request, and returns the tier whose candidate Drex picked.

    One attempt on the httpx client of LiteLLM's shared, event-loop-scoped NACE handler
    (redirects off per request, identity encoding, a 64 KiB streamed cap, status 200 only,
    probabilities in [0, 1]); the handler is held for the request's duration so cache
    eviction cannot orphan its pooled connections, and nothing is stored on the classifier.
    Every failure declines with None or raises a sanitized error so the complexity
    router's classifier_fallback decides the destination, and the returned string is
    always a tier name from the construction-time snapshot."""

    def __init__(
        self,
        config: DrexRouteClassifierConfig | Mapping[str, object],
        http_client: AsyncHTTPHandler | None = None,
    ) -> None:
        """`http_client` injects the handler whose underlying httpx client serves tests;
        production resolves LiteLLM's shared NACE handler per call. A mapping config and a
        key from the environment leave this frame before a rejection is raised, so no
        traceback holds a plaintext key."""
        validated: Final = config if isinstance(config, DrexRouteClassifierConfig) else _validated_config(config)
        del config  # rebind-ok: a mapping config can hold the plaintext key
        if isinstance(validated, Exception):
            raise validated
        api_base, resolved_key = _NACE.resolve_connection(validated.api_base, _secret_value(validated.api_key))
        key: Final = None if resolved_key is None else _header_safe_key(resolved_key)
        del resolved_key  # the plaintext key leaves this frame before a rejection is raised
        if isinstance(key, DrexRouteConfigError):
            raise key
        self._candidates: Final = tuple(validated.candidates.items())
        self._selection_priority: Final = validated.selection_priority
        self._vision_tiers: Final = validated.vision_tiers
        self._max_input_bytes: Final = validated.max_input_bytes
        self._timeout_s: Final = validated.timeout_s
        self._cleanup_grace_s: Final = validated.cleanup_grace_s
        self._max_detached_exchanges: Final = validated.max_detached_exchanges
        self._api_base: Final = api_base
        self._api_key: Final = key
        self._handler_factory: Final = _handler_factory(http_client)

    def _eligible_tiers(
        self, raw: Sequence[Mapping[str, object]], structured: Sequence[Mapping[str, object]]
    ) -> tuple[tuple[str, DrexRouteCandidate], ...]:
        if self._vision_tiers is None:
            return self._candidates
        surface: Final = raw or structured
        if not request_contains_image_content(surface):
            return self._candidates
        return tuple(pair for pair in self._candidates if pair[0] in self._vision_tiers)

    async def classify(self, context: "RoutingContext") -> str | None:
        """The tier Drex picked, None to decline, or a sanitized error. The error is raised
        here with no traceback of its own, so no frame that held the request, its key, the
        input or the response body is reachable from it."""
        outcome: Final = await self._route(context)
        del context  # rebind-ok: the routing input leaves this frame before the outcome is raised
        if isinstance(outcome, BaseException):
            raise outcome.with_traceback(None)
        return outcome

    async def _route(self, context: "RoutingContext") -> str | BaseException | None:
        if self._api_base is None or self._api_key is None:
            return None
        eligible: Final = self._eligible_tiers(context.raw_messages, context.structured_messages)
        turns: Final = _projected_turns(context.structured_messages or context.raw_messages, self._max_input_bytes)
        if isinstance(turns, DrexRouteError):
            return turns
        if not eligible or not turns:
            return None
        detached: Final = _detached_on(asyncio.get_running_loop())
        if detached >= self._max_detached_exchanges:
            verbose_router_logger.debug(
                "Drex Router skipped: %d earlier exchanges are still releasing their responses", detached
            )
            return DrexRouteError(f"drex route call skipped: {detached} earlier exchanges are still closing")
        handler: Final = self._handler_factory()
        client: Final = handler.client
        request: Final = _route_request(
            client,
            f"{self._api_base.rstrip('/').removesuffix('/v1')}{_ROUTE_PATH}",
            {
                "messages": list(turns),
                "candidates": [_wire_candidate(candidate) for _, candidate in eligible],
                "selection_priority": self._selection_priority,
            },
            self._api_key,
            self._timeout_s,
            _SingleAttempt(),
        )
        if isinstance(request, DrexRouteError):
            return request
        scope: Final = anyio.CancelScope()
        exchange: Final = asyncio.ensure_future(_post_route(client, request, self._timeout_s, scope))
        caller_cancelled: asyncio.CancelledError | None = None  # rebind-ok: returned once the exchange stops
        try:
            _ = await asyncio.wait((exchange,))
        except asyncio.CancelledError as cancelled:
            caller_cancelled = cancelled  # rebind-ok: the caller's cancellation, held while the exchange stops
        if caller_cancelled is not None:
            await _stop_exchange(exchange, scope, self._cleanup_grace_s)
            return caller_cancelled
        answer: Final = _exchange_outcome(exchange)
        if isinstance(answer, Exception):
            pending: Final = await _pending_cancellation()
            return answer if pending is None else pending
        tier: Final = next((name for name, candidate in eligible if _candidate_id(candidate) == answer.selected), None)
        if tier is None:
            verbose_router_logger.warning(
                "Drex Router selected an ID outside the offered candidates (length %d), declining",
                len(answer.selected),
            )
            return None
        verbose_router_logger.debug(
            "Drex Router chose tier %s from %d candidates, probability %s",
            tier,
            len(eligible),
            answer.probabilities.get(answer.selected),
        )
        return tier
