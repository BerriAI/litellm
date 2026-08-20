"""Needlepath guardrail: query-conditioned extractive context selection.

Bulky message content (tool outputs by default) is sent to the Needlepath
selection service before the request reaches the LLM. The service returns the
spans of that content which carry the answer to a query, and the guardrail
writes those spans back over the message they came from. Nothing is
paraphrased or rewritten: the returned block is made of extracts of the text
that was submitted.

Selection is per message and query-conditioned. The query for a tool output is
the intent of the tool call that produced it (``name`` plus ``arguments``,
found through ``tool_call_id``); anything else uses the last user message. Each
message is selected independently, so one message's outcome never changes
another's.

**This guardrail is unconditionally fail-open.** Every path that does not
produce a usable selection returns the caller's messages untouched. A proxy
that silently blanks a tool output is far worse than a proxy that does nothing,
so there is no configuration in which a selection failure becomes a request
failure. See ``_selected_text`` for the full list of declines.
"""

from __future__ import annotations

import asyncio
import ipaddress
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal
from urllib.parse import urlparse

import httpx
from httpx import Response as HttpxResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # helper is untyped in http_handler
    httpxSpecialProvider,
)
from litellm.proxy.guardrails.guardrail_hooks.content_text import (
    content_to_text,
    is_all_text_parts,
    merge_rewritten_text_parts,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )

DEFAULT_API_BASE: Final = "https://api.nextmoca.com"
SELECT_PATH: Final = "/v1/context/select"
# Immutable engine label. Pinned rather than inherited from the service default
# so an upgrade on the service side cannot change what this proxy sends without
# an operator changing this config.
DEFAULT_OPERATING_POINT: Final = "np-2026-07-r2"
DEFAULT_MAX_CONTEXT_TOKENS: Final = 4000
DEFAULT_MIN_CHARS_TO_SELECT: Final = 500
# The shared client's read timeout is measured in minutes, which is far too long
# to hold an inbound LLM request behind an optional optimisation. A stall past
# this bound is a decline, and the original message is forwarded.
_SELECT_TIMEOUT_SECONDS: Final = 30.0
# A single proxy request can carry arbitrarily many eligible messages, and each
# one becomes an outbound selection call. These two bounds keep a pathological
# request (say, thousands of minimally qualifying tool outputs) from
# monopolising the proxy's shared HTTP pool or burning selection quota: at most
# _MAX_TARGETS_PER_REQUEST messages are selected per request -- the largest
# first, where selection pays off most -- and at most
# _MAX_CONCURRENT_SELECTIONS calls are in flight at once. Messages past the
# cap are forwarded untouched, the same fail-open outcome as any decline.
_MAX_TARGETS_PER_REQUEST: Final = 16
_MAX_CONCURRENT_SELECTIONS: Final = 4
# The service reports a deliberate no-op through the gate. Any reason under this
# prefix means "the engine chose not to select"; the original content is what
# the caller should send.
_STANDDOWN_PREFIX: Final = "standdown:"
_BLOCKED_METADATA_HOSTS: Final = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
        "metadata.azure.com",
        "metadata.azure.internal",
    }
)
_BLOCKED_METADATA_IPS: Final = frozenset(
    ipaddress.ip_address(ip) for ip in ("169.254.169.254", "fd00:ec2::254", "100.100.100.200", "168.63.129.16")
)
# Record kinds the service publishes. A system prompt has no dedicated kind, so
# it is submitted as external_data rather than invented as a new one: an
# unrecognised kind is a 400 for the whole call.
_KIND_TOOL_RESULT: Final = "tool_result"
_KIND_USER_INPUT: Final = "user_input"
_KIND_EXTERNAL_DATA: Final = "external_data"


def _parse_ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse ``host`` as an IP literal, including the alternate spellings the
    socket layer accepts (single-integer IPv4, IPv4-mapped IPv6), so a blocked
    address cannot be smuggled past a plain string comparison."""
    try:
        addr = ipaddress.ip_address(host)  # rebind-ok: reassigned in the except fallback below
    except ValueError:
        try:
            addr = ipaddress.ip_address(int(host, 0))  # rebind-ok: fallback re-parse as an int literal
        except (TypeError, ValueError):
            return None
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def _validate_api_base(url: str) -> str:
    """Return ``url`` if it passes basic outbound-target checks, else raise.

    Defense in depth against a mistyped or hostile ``api_base``: non-http(s)
    schemes and cloud-metadata hosts/IPs are refused. Private ranges stay
    allowed so on-prem deployments work. This is not a complete SSRF control:
    there is no DNS resolution here and the shared client follows redirects.
    ``api_base`` is operator config, so that is an accepted limit.
    """
    parsed: Final = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Needlepath guardrail api_base must be http or https, got scheme={parsed.scheme!r}")
    host: Final = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Needlepath guardrail api_base has no host")
    ip_literal: Final = _parse_ip_literal(host)
    if host in _BLOCKED_METADATA_HOSTS or (ip_literal is not None and ip_literal in _BLOCKED_METADATA_IPS):
        raise ValueError(f"Needlepath guardrail api_base {host!r} is a blocked cloud-metadata host")
    return url


def _write_text_back(content: object, new_text: str) -> object:
    """Put ``new_text`` into a ``content`` value without changing its shape.

    A string is replaced directly. An all-text part list collapses to a single
    part carrying the last declared cache_control breakpoint. Anything else is
    returned untouched: breakpoints are positional, so one selected block cannot
    be written across a non-text part without moving text past it.
    """
    if isinstance(content, str):
        return new_text
    if isinstance(content, list) and is_all_text_parts(content):
        return merge_rewritten_text_parts(content, new_text)
    return content


def _render_tool_intent(fn: Mapping[str, object]) -> str:
    """A tool call rendered as the query its output should be selected against."""
    name: Final = str(fn.get("name") or "").strip()
    raw_args: Final = fn.get("arguments")
    arg_text: Final = "" if raw_args is None else str(raw_args).strip()
    if name and arg_text:
        return f"{name}: {arg_text}"
    return name or arg_text


def _query_for_target(messages: Sequence[Mapping[str, object]], target_idx: int, fallback: str) -> str:
    """The query ``messages[target_idx]`` should be selected against.

    A tool or function result is selected against the intent of the call that
    produced it, located by ``tool_call_id`` on an earlier assistant message.
    Everything else, and any tool result whose call cannot be found, uses the
    last user message.
    """
    msg: Final = messages[target_idx]
    if msg.get("role") not in ("tool", "function"):
        return fallback

    tool_call_id: Final = msg.get("tool_call_id")
    fn_name: Final = msg.get("name")
    for idx in range(target_idx - 1, -1, -1):
        previous = messages[idx]
        if previous.get("role") != "assistant":
            continue
        tool_calls = previous.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict) or not tool_call_id or call.get("id") != tool_call_id:
                    continue
                fn = call.get("function")
                intent = _render_tool_intent(fn if isinstance(fn, dict) else MappingProxyType({}))
                if intent:
                    return intent
        # Legacy function_call turns carry no id, so require a name match.
        # Without it an older, unrelated call would supply the wrong intent.
        legacy = previous.get("function_call")
        if isinstance(legacy, dict) and fn_name and legacy.get("name") == fn_name:
            intent = _render_tool_intent(legacy)
            if intent:
                return intent
    return fallback


def _title_for(messages: Sequence[Mapping[str, object]], target_idx: int) -> str | None:
    """The tool name behind a message, used as the record title."""
    msg: Final = messages[target_idx]
    name: Final = msg.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()[:120]
    tool_call_id: Final = msg.get("tool_call_id")
    if not tool_call_id:
        return None
    for idx in range(target_idx - 1, -1, -1):
        previous = messages[idx]
        if previous.get("role") != "assistant":
            continue
        tool_calls = previous.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict) or call.get("id") != tool_call_id:
                continue
            fn = call.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                return str(fn["name"])[:120]
    return None


def _record_kind(role: object) -> str:
    if role in ("tool", "function"):
        return _KIND_TOOL_RESULT
    if role == "user":
        return _KIND_USER_INPUT
    return _KIND_EXTERNAL_DATA


def _safe_int(value: object) -> int | None:
    """Read an integer counter from an untrusted body without raising.

    A field that is missing or not a number is reported as ``None`` so the
    caller can treat it as "unknown" rather than as zero, which is a decline.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_response_text(response: object, limit: int = 500) -> str:
    """Read a response body for a log line without letting the read itself raise.

    A corrupt ``Content-Encoding`` makes httpx's ``.text`` raise, which would
    turn an already-handled decline into an unhandled 500.
    """
    try:
        text: Final = getattr(response, "text", "")
    except httpx.DecodingError:
        return "<undecodable response body>"
    return (text or "")[:limit]


class NeedlepathGuardrail(CustomGuardrail):
    """Select the spans of a message that answer the current query.

    Every knob is optional and the defaults are deliberately narrow: tool
    outputs only, nothing under ``min_chars_to_select`` characters, and a pinned
    operating point.
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        select_tool_outputs: bool | None = None,
        select_history: bool | None = None,
        select_system: bool | None = None,
        min_chars_to_select: int | None = None,
        max_context_tokens: int | None = None,
        operating_point: str | None = None,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | Sequence[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = False,
    ) -> None:
        raw_api_base: Final = (api_base or get_secret_str("NEEDLEPATH_API_BASE") or DEFAULT_API_BASE).rstrip("/")
        self.needlepath_api_base = _validate_api_base(raw_api_base)
        self.needlepath_api_key = api_key or get_secret_str("NEEDLEPATH_API_KEY")
        if not self.needlepath_api_key:
            raise ValueError(
                "Needlepath guardrail requires an API key. Set `api_key` in the "
                "guardrail config or the NEEDLEPATH_API_KEY env var."
            )
        self.select_tool_outputs = True if select_tool_outputs is None else select_tool_outputs
        self.select_history = False if select_history is None else select_history
        self.select_system = False if select_system is None else select_system
        self.min_chars_to_select = (
            DEFAULT_MIN_CHARS_TO_SELECT if min_chars_to_select is None else int(min_chars_to_select)
        )
        self.max_context_tokens = DEFAULT_MAX_CONTEXT_TOKENS if max_context_tokens is None else int(max_context_tokens)
        # Pinned, not inherited. The labels are immutable, so a pinned one makes
        # what this guardrail sends reproducible across service releases.
        self.operating_point = operating_point or DEFAULT_OPERATING_POINT
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.__init__ is untyped
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
        )

    def _request_headers(self) -> Mapping[str, str]:
        # httpx's `headers=` accepts any Mapping (it iterates .items()), so this can
        # be a genuine read-only view rather than a dict a caller could mutate.
        return MappingProxyType(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.needlepath_api_key or ''}",
            }
        )

    def _decline(self, reason: str, detail: Mapping[str, object] | None = None) -> None:
        """Record a decline. The caller then forwards the original content.

        Declining is the normal outcome for a stand-down or a service problem,
        so it is logged at debug and never raised. Details can include upstream
        response bytes and stay in the proxy's own logs.
        """
        # The dict fallback below is never a value a caller could mutate: `or {}` only
        # feeds the debug-log %s formatting, and swapping it for a frozen mapping would
        # change the logged repr from `{}` to `mappingproxy({})`.
        verbose_proxy_logger.debug("Needlepath: declined (%s) detail=%s", reason, detail or {})  # mutable-ok: see above

    def _select_targets(self, messages: Sequence[Mapping[str, object]], query_idx: int | None) -> tuple[int, ...]:
        """Indices of the messages whose text content is eligible for selection."""

        def _eligible(idx: int, msg: Mapping[str, object]) -> bool:
            # The message carrying the query is never rewritten: selecting the
            # question against itself is meaningless and it is what the rest of
            # the selection is conditioned on.
            if idx == query_idx:
                return False
            role: Final = msg.get("role")
            if role in ("tool", "function"):
                if not self.select_tool_outputs:
                    return False
            elif role == "system":
                if not self.select_system:
                    return False
            elif role == "user":
                if not self.select_history:
                    return False
            else:
                return False
            content: Final = msg.get("content")
            if isinstance(content, list) and not is_all_text_parts(content):
                return False
            return len(content_to_text(content)) >= self.min_chars_to_select

        return tuple(idx for idx, msg in enumerate(messages) if _eligible(idx, msg))

    @staticmethod
    def _last_user_message(messages: Sequence[Mapping[str, object]]) -> tuple[str, int | None]:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                return content_to_text(messages[idx].get("content")), idx
        return "", None

    async def _post_select(self, payload: Mapping[str, object]) -> HttpxResponse | None:
        """POST one selection request. Returns None on any transport failure."""
        try:
            return await self.async_handler.post(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler.post is untyped
                url=f"{self.needlepath_api_base}{SELECT_PATH}",
                json=payload,
                headers=self._request_headers(),
                timeout=_SELECT_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            # The shared handler calls raise_for_status(), so 402, 403, 429 and
            # every other non-2xx arrives here carrying the upstream body and
            # our Authorization header. It is a decline and nothing about it
            # reaches the client.
            response: Final = getattr(e, "response", None)
            self._decline(
                "http_status",
                {  # mutable-ok: log detail dict, consumed by _decline's debug log immediately
                    "status_code": getattr(response, "status_code", None),
                    "body": _safe_response_text(response),
                },
            )
            return None
        except (httpx.RequestError, litellm.Timeout) as e:
            # Every request-side httpx failure, timeouts included, is a
            # RequestError. Catching the whole class is what keeps a network
            # problem from escaping as a 500.
            self._decline("transport", {"detail": str(e)})  # mutable-ok: log detail dict, consumed immediately
            return None

    async def _selected_text(self, text: str, query: str, title: str | None, source: object, kind: str) -> str | None:
        """The selected block for one message, or None to keep the original.

        None is returned, and the original content is forwarded, when:

        * the service could not be reached, timed out, or answered non-2xx
          (402, 403, 429 and everything else);
        * the body is not JSON, or is JSON of an unexpected shape;
        * the gate reports a stand-down (any ``gate.reason`` under
          ``standdown:``), which is the service saying the full content is what
          the caller should send;
        * ``records_selected`` is 0, ``tokens_after`` is 0, or
          ``rendered_context`` is missing or blank;
        * the returned block is not shorter than the text it would replace, so a
          rewrite could never grow a message.
        """
        # `record` and `payload` are both real, mutable dicts by necessity: `record`
        # gains optional keys after construction (below), and both are handed straight
        # to httpx's `json=`, which needs a genuine dict/list -- a MappingProxyType
        # is not JSON-serializable (json.dumps raises TypeError on one), so the
        # freezing wrappers used elsewhere in this file cannot be used here.
        record: Final[dict[str, object]] = {  # mutable-ok: see comment above
            "id": "m0",
            "kind": kind,
            "text": text,
        }
        if title:
            record["title"] = title
        if isinstance(source, str) and source:
            record["source"] = source
        payload: Final[dict[str, object]] = {  # mutable-ok: see comment above
            "records": (record,),
            "task": {"prompt": query},  # mutable-ok: nested JSON payload dict, see comment above
            "budget": {  # mutable-ok: nested JSON payload dict, see comment above
                "max_context_tokens": self.max_context_tokens,
                "operating_point": self.operating_point,
            },
            "render": True,
            "render_format": "plain",
        }

        response: Final = await self._post_select(payload)
        if response is None:
            return None
        if not 200 <= response.status_code < 300:
            self._decline(
                "http_status",
                {  # mutable-ok: log detail dict, consumed by _decline's debug log immediately
                    "status_code": response.status_code,
                    "body": _safe_response_text(response),
                },
            )
            return None

        try:
            body: Final[object] = response.json()
        except (ValueError, httpx.DecodingError, RecursionError):
            # RecursionError: a deeply nested body overflows the JSON parser.
            # It is a decline like any other malformed answer, not a 500.
            self._decline("unreadable_body", {"body": _safe_response_text(response)})  # mutable-ok: log detail dict
            return None
        if not isinstance(body, dict):
            self._decline("unexpected_shape", {"body": _safe_response_text(response)})  # mutable-ok: log detail dict
            return None

        gate: Final = body.get("gate")
        reason: Final = gate.get("reason") if isinstance(gate, dict) else None
        if isinstance(reason, str) and reason.startswith(_STANDDOWN_PREFIX):
            self._decline("gate_standdown", {"reason": reason})  # mutable-ok: log detail dict
            return None

        records_selected: Final = _safe_int(body.get("records_selected"))
        tokens_after: Final = _safe_int(body.get("tokens_after"))
        if records_selected == 0 or tokens_after == 0:
            self._decline(
                "empty_selection",
                {  # mutable-ok: log detail dict, consumed by _decline's debug log immediately
                    "records_selected": records_selected,
                    "tokens_after": tokens_after,
                },
            )
            return None

        rendered: Final = body.get("rendered_context")
        if not isinstance(rendered, str) or not rendered.strip():
            self._decline("empty_rendered_context", MappingProxyType({}))
            return None
        if len(rendered) >= len(text):
            # A selection that is not smaller is not a selection worth making,
            # and applying it could only add tokens.
            self._decline("no_reduction", {"before": len(text), "after": len(rendered)})  # mutable-ok: log detail
            return None
        return rendered

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "request":
            return inputs

        structured_messages: Final = inputs.get("structured_messages")
        if not isinstance(structured_messages, list) or not structured_messages:
            return inputs
        # The static type says every entry is an AllMessageValues TypedDict, but this
        # is the proxy's client-facing boundary: the list arrives from arbitrary
        # request JSON, so the runtime check is the guarantee the annotation only
        # promises.
        messages: Final = tuple(
            m
            for m in structured_messages
            if isinstance(m, dict)  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime boundary check on client JSON
        )
        if len(messages) != len(structured_messages):
            return inputs

        fallback_query, query_idx = self._last_user_message(messages)
        # One (idx, query) pair per eligible message whose query is non-blank.
        # Selection is conditioned on a query; without one there is nothing to
        # select against, so that message is left exactly as it arrived.
        eligible_pairs: Final = tuple(
            (idx, query)
            for idx in self._select_targets(messages, query_idx)
            if (query := _query_for_target(messages, idx, fallback_query)).strip()
        )
        if not eligible_pairs:
            verbose_proxy_logger.debug("Needlepath: no messages eligible for selection")
            return inputs

        # Cap the fan-out: the largest messages are kept because that is where
        # selection saves the most, and the choice must not depend on message
        # order in the request. Everything past the cap is forwarded untouched.
        selected_pairs: Final = (
            eligible_pairs
            if len(eligible_pairs) <= _MAX_TARGETS_PER_REQUEST
            else tuple(
                sorted(
                    eligible_pairs,
                    key=lambda pair: len(content_to_text(messages[pair[0]].get("content"))),
                    reverse=True,
                )[:_MAX_TARGETS_PER_REQUEST]
            )
        )
        if len(selected_pairs) < len(eligible_pairs):
            verbose_proxy_logger.debug(
                "Needlepath: %d eligible messages, selecting only the %d largest",
                len(eligible_pairs),
                _MAX_TARGETS_PER_REQUEST,
            )

        semaphore: Final = asyncio.Semaphore(_MAX_CONCURRENT_SELECTIONS)

        async def _bounded_selected_text(idx: int, query: str) -> str | None:
            # The semaphore bounds how many selection calls this one proxy
            # request holds open at a time; see _MAX_CONCURRENT_SELECTIONS.
            async with semaphore:
                return await self._selected_text(
                    text=content_to_text(messages[idx].get("content")),
                    query=query,
                    title=_title_for(messages, idx),
                    source=messages[idx].get("tool_call_id"),
                    kind=_record_kind(messages[idx].get("role")),
                )

        start_time: Final = time.monotonic()
        # One request per message: each carries its own query, and the service
        # renders one block per call. Running them concurrently (up to the
        # semaphore's bound) keeps the added latency near one round trip
        # rather than one per message.
        selections: Final = await asyncio.gather(*(_bounded_selected_text(idx, query) for idx, query in selected_pairs))
        end_time: Final = time.monotonic()

        # Needs in-place index assignment below to build the edited copy without
        # touching the frozen `messages` tuple; only replaced indices differ from
        # the original list -- see the identity note below.
        selected_messages: Final = list(messages)  # mutable-ok: see comment above
        messages_selected = 0  # rebind-ok: loop accumulator, incremented per selection below
        chars_before = 0  # rebind-ok: loop accumulator, incremented per selection below
        chars_after = 0  # rebind-ok: loop accumulator, incremented per selection below
        for (idx, _query), selection in zip(selected_pairs, selections):
            original = selected_messages[idx]
            original_text = content_to_text(original.get("content"))
            if selection is None:
                continue
            messages_selected += 1
            chars_before += len(original_text)
            chars_after += len(selection)
            selected_messages[idx] = {  # mutable-ok: plain AllMessageValues dict, see comment above return
                **original,
                "content": _write_text_back(original.get("content"), selection),
            }

        if messages_selected == 0:
            # Nothing was replaced, so hand back the exact inputs object. The
            # handlers detect a guardrail edit by identity, and a fresh list
            # would force a write-back of an untouched request (on Anthropic
            # that reconversion strips cache_control from thinking blocks).
            verbose_proxy_logger.debug("Needlepath: no selection applied; request unchanged")
            return inputs

        # Serialized as guardrail_json_response for standard logging (DataDog,
        # Langfuse, ...); must stay a plain dict for json.dumps compatibility.
        stats: Final[dict[str, object]] = {  # mutable-ok: see comment above
            "messages_selected": messages_selected,
            "messages_considered": len(selected_pairs),
            "chars_before": chars_before,
            "chars_after": chars_after,
            "operating_point": self.operating_point,
        }
        verbose_proxy_logger.debug(
            "Needlepath: selected %s of %s message(s), %s -> %s chars",
            messages_selected,
            len(selected_pairs),
            chars_before,
            chars_after,
        )
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=stats,
            request_data=request_data,
            guardrail_status="success",
            guardrail_provider="needlepath",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )
        return {  # pyright: ignore[reportReturnType]  # plain dicts satisfy AllMessageValues at runtime  # mutable-ok: TypedDict, treated as plain dict downstream
            **inputs,
            "structured_messages": selected_messages,
        }

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel[object]] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.needlepath import (
            NeedlepathGuardrailConfigModel,
        )

        return NeedlepathGuardrailConfigModel
