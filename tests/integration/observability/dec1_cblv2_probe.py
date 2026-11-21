"""Callback module the dec1 CustomLoggerV2 contract tests load into the proxy as `module.instance`.

Two families of loggers live here.

The legacy `Chain` loggers (`chain_first`, `chain_second`, `chain_third`) are plain CustomLogger instances that
record every hook call to the wire server named by CBLV2_PROBE_URL, with a per-process sequence number, so tests
can read the order the proxy called them in. They change the request or response only when the request text
carries one of the `cblv2-...` directives below, which lets one proxy serve every ordering scenario.

The V2 loggers subclass CustomLoggerV2 and declare their hooks with `@on`. `v2_gate` writes and may abort at
`pre_call`, writes response headers, and patches the response at `post_call_success`. `v2_reader` is an off-path
`attempt_ended` reader with the default Selection (so it does not honour no-log), `v2_no_log_reader` declares
`honors_no_log=True` and caller opt-out, `v2_crashing_reader` raises on every event, and `v2_awaited_reader` is an
awaited reader. Every V2 record carries the request text so tests can find it without a call id: the guide says
`ctx.call` can be None on proxy hooks, so the call id is recorded only when present.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import AsyncGenerator, Mapping
from typing import Final

import httpx
from fastapi import HTTPException

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2 import (
    Abort,
    Activation,
    CustomLoggerV2,
    Field,
    Hook,
    OptOutBy,
    Patch,
    Selection,
    on,
)

CHAIN: Final = "cblv2-chain"
REJECT_AT: Final = "cblv2-reject-at-"
CRASH_LOG_AT: Final = "cblv2-crash-log-at-"
CRASH_PRE_AT: Final = "cblv2-crash-pre-at-"
ITERATOR: Final = "cblv2-iterator"
V2_ABORT: Final = "cblv2-v2-abort"
V2_ABORT_CUSTOM_EXCEPTION: Final = "cblv2-v2-abort-exception"
V2_PATCH: Final = "cblv2-v2-patch"
V2_RAISE: Final = "cblv2-v2-raise"
V2_POST_PATCH: Final = "cblv2-v2-post-patch"
V2_METADATA_PATCH: Final = "cblv2-v2-metadata-patch"

V2_ABORT_STATUS: Final = 451
V2_ABORT_MESSAGE: Final = "blocked by the cblv2 v2 gate"
V2_ABORT_HEADER: Final = "x-cblv2-blocked-by"
V2_EXCEPTION_STATUS: Final = 409
V2_EXCEPTION_MESSAGE: Final = "conflict raised by the cblv2 v2 gate"
V2_REWRITTEN_REQUEST: Final = "rewritten by the cblv2 v2 gate"
V2_REWRITTEN_RESPONSE: Final = "rewritten by the cblv2 v2 post-call handler"
V2_RESPONSE_HEADER: Final = "x-cblv2-v2-gate"
CHAIN_REJECTION: Final = "rejected by the cblv2 chain logger"

_SEQUENCE: Final = itertools.count()


def _text(container: object) -> str:
    if not isinstance(container, Mapping):
        return ""
    return json.dumps({name: container.get(name) for name in ("messages", "input")}, default=str, ensure_ascii=False)


async def _send(record: Mapping[str, object]) -> None:
    body: Final = json.dumps({**record, "seq": next(_SEQUENCE), "pid": os.getpid()}, default=str).encode()
    async with httpx.AsyncClient(trust_env=False) as client:
        (await client.post(os.environ["CBLV2_PROBE_URL"], content=body)).raise_for_status()


def _tag_last_message(data: dict[str, object], name: str) -> dict[str, object]:
    messages: Final = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return data
    last: Final = dict(messages[-1])
    last["content"] = f"{last.get('content')} [{name}]"
    return {**data, "messages": [*messages[:-1], last]}


class Chain(CustomLogger):
    """Legacy logger whose hooks record their order and, on a directive, tag or reject the request or reply."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    async def _record(self, hook: str, data: object, **extra: object) -> None:
        fields: Final = data if isinstance(data, Mapping) else {}
        await _send(
            {
                "logger": self.name,
                "hook": hook,
                "litellm_call_id": fields.get("litellm_call_id"),
                "text": _text(fields),
                **extra,
            }
        )

    async def async_pre_call_hook(
        self, user_api_key_dict: object, cache: object, data: dict[str, object], call_type: str
    ) -> dict[str, object] | None:
        await self._record("async_pre_call_hook", data)
        text: Final = _text(data)
        if f"{CRASH_PRE_AT}{self.name}" in text:
            raise ValueError(f"the cblv2 chain logger {self.name} crashed")
        if f"{REJECT_AT}{self.name}" in text:
            raise HTTPException(status_code=400, detail={"error": f"{CHAIN_REJECTION} {self.name}"})
        if CHAIN in text:
            return _tag_last_message(data, self.name)
        return None

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: object, response: object
    ) -> object:
        await self._record("async_post_call_success_hook", data)
        if CHAIN not in _text(data):
            return response
        choices: Final = getattr(response, "choices", None)
        if choices:
            message: Final = choices[0].message
            message.content = f"{message.content} [{self.name}]"
        return response

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: object,
        traceback_str: str | None = None,
    ) -> None:
        await self._record("async_post_call_failure_hook", request_data, error=type(original_exception).__name__)

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: object, response: object, request_data: dict[str, object]
    ) -> AsyncGenerator[object, None]:
        await self._record("async_post_call_streaming_iterator_hook", request_data)
        tagging: Final = ITERATOR in _text(request_data)
        first = True
        async for item in response:  # pyright: ignore[reportGeneralTypeIssues]  # the proxy hands an async iterator
            if tagging and first:
                delta = item.choices[0].delta  # pyright: ignore[reportAttributeAccessIssue]  # ModelResponseStream
                if delta.content:
                    delta.content = f"[{self.name}]{delta.content}"
                    first = False
            yield item

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        payload: Final = kwargs.get("standard_logging_object")
        await self._record(
            "async_log_success_event",
            kwargs,
            standard_id=payload.get("id") if isinstance(payload, Mapping) else None,
        )
        if f"{CRASH_LOG_AT}{self.name}" in _text(kwargs):
            raise RuntimeError(f"the cblv2 chain logger {self.name} crashed while logging")

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._record("async_log_failure_event", kwargs)


def _messages_text(view: object) -> str:
    messages: Final = getattr(view, "messages", None)
    return json.dumps({"messages": messages, "input": getattr(view, "input", None)}, default=str)


def _call_id(ctx: object) -> str | None:
    call: Final = getattr(ctx, "call", None)
    ids: Final = getattr(call, "ids", None)
    return getattr(ids, "call_id", None)


async def _send_v2(logger: str, hook: str, view: object, ctx: object, **extra: object) -> None:
    await _send(
        {
            "logger": logger,
            "hook": hook,
            "litellm_call_id": _call_id(ctx),
            "text": _messages_text(view),
            **extra,
        }
    )


class V2Gate(CustomLoggerV2):
    """Writes and may abort, so it is required for every request and cannot honour no-log or caller opt-out."""

    selection = Selection(name="cblv2_v2_gate")

    @on(Hook.pre_call, reads={Field.messages}, writes={Field.messages}, may_abort=True)
    async def gate(self, view: object, ctx: object) -> object:
        await _send_v2("v2_gate", "pre_call", view, ctx)
        text: Final = _messages_text(view)
        if V2_ABORT_CUSTOM_EXCEPTION in text:
            return Abort(
                status=V2_EXCEPTION_STATUS,
                message=V2_EXCEPTION_MESSAGE,
                exception=HTTPException(status_code=V2_EXCEPTION_STATUS, detail={"error": V2_EXCEPTION_MESSAGE}),
            )
        if V2_ABORT in text:
            return Abort(status=V2_ABORT_STATUS, message=V2_ABORT_MESSAGE, headers={V2_ABORT_HEADER: "cblv2_v2_gate"})
        if V2_RAISE in text:
            raise RuntimeError("the cblv2 v2 gate raised")
        if V2_PATCH in text:
            return Patch().set(Field.messages, [{"role": "user", "content": V2_REWRITTEN_REQUEST}])
        return None

    @on(Hook.response_headers, writes={Field.response_headers})
    async def headers(self, view: object, ctx: object) -> object:
        return Patch().merge(Field.response_headers, {V2_RESPONSE_HEADER: "seen"})


class V2PostGate(CustomLoggerV2):
    """Rewrites the reply text at post_call_success when the request carries V2_POST_PATCH."""

    selection = Selection(name="cblv2_v2_post_gate")

    @on(Hook.post_call_success, reads={Field.messages, Field.response}, writes={Field.response})
    async def rewrite(self, view: object, ctx: object) -> object:
        await _send_v2("v2_post_gate", "post_call_success", view, ctx)
        if V2_POST_PATCH not in _messages_text(view):
            return None
        response: Final = getattr(view, "response", None)
        choices: Final = getattr(response, "choices", None)
        if not choices:
            return None
        choices[0].message.content = V2_REWRITTEN_RESPONSE
        return Patch().set(Field.response, response)


class V2MetadataGate(CustomLoggerV2):
    """Declares a metadata write the proxy pre_call cannot apply yet (InvalidPatchError per the user guide)."""

    selection = Selection(name="cblv2_v2_metadata_gate")

    @on(Hook.pre_call, reads={Field.messages}, writes={Field.metadata})
    async def tag(self, view: object, ctx: object) -> object:
        if V2_METADATA_PATCH not in _messages_text(view):
            return None
        return Patch().merge(Field.metadata, {"cblv2": "patched"})


class V2Reader(CustomLoggerV2):
    """Off-path attempt_ended reader with the default Selection, which does not honour no-log."""

    def __init__(self, name: str) -> None:
        self.name = name

    @on(Hook.attempt_ended, reads={Field.model, Field.outcome, Field.messages, Field.usage, Field.cost})
    def record(self, view: object, ctx: object) -> None:
        outcome: Final = getattr(view, "outcome", None)
        usage: Final = getattr(view, "usage", None)
        logged: Final = getattr(usage, "logged", None)
        cost: Final = getattr(view, "cost", None)
        body: Final = json.dumps(
            {
                "logger": self.name,
                "hook": "attempt_ended",
                "seq": next(_SEQUENCE),
                "pid": os.getpid(),
                "litellm_call_id": _call_id(ctx),
                "text": _messages_text(view),
                "model": getattr(view, "model", None),
                "outcome": getattr(outcome, "value", outcome),
                "total_tokens": getattr(logged, "total_tokens", None),
                "response_cost": getattr(cost, "response_cost", None),
            },
            default=str,
        ).encode()
        with httpx.Client(trust_env=False) as client:
            client.post(os.environ["CBLV2_PROBE_URL"], content=body).raise_for_status()


class V2NoLogReader(V2Reader):
    selection = Selection(
        name="cblv2_no_log_reader",
        activation=Activation.default_on,
        opt_out_by=frozenset({OptOutBy.admin, OptOutBy.caller}),
        honors_no_log=True,
    )


class V2CrashingReader(CustomLoggerV2):
    @on(Hook.attempt_ended, reads={Field.outcome})
    def explode(self, view: object, ctx: object) -> None:
        raise RuntimeError("the cblv2 crashing reader always fails")


class V2AwaitedReader(CustomLoggerV2):
    selection = Selection(name="cblv2_awaited_reader")

    @on(Hook.attempt_ended, reads={Field.outcome, Field.messages}, awaited=True)
    async def record(self, view: object, ctx: object) -> None:
        outcome: Final = getattr(view, "outcome", None)
        await _send_v2("v2_awaited_reader", "attempt_ended", view, ctx, outcome=getattr(outcome, "value", outcome))


chain_first: Final = Chain("chain_first")
chain_second: Final = Chain("chain_second")
chain_third: Final = Chain("chain_third")
v2_gate: Final = V2Gate()
v2_post_gate: Final = V2PostGate()
v2_metadata_gate: Final = V2MetadataGate()
v2_reader: Final = V2Reader("v2_reader")
v2_no_log_reader: Final = V2NoLogReader("v2_no_log_reader")
v2_crashing_reader: Final = V2CrashingReader()
v2_awaited_reader: Final = V2AwaitedReader()
