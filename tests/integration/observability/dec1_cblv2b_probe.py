"""Callback module the dec1 cblv2b contract tests load into the proxy as `module.instance`.

Every logger reports each hook call to the probe sink named by CBLV2_PROBE_URL (the sink `cblv2_rig` boots) as one
JSON record with `logger`, `hook`, `text` and a per-process `seq`, so tests can read who ran, in what order, and
what each one saw. `text` is the request messages as JSON when the hook has them; hooks without the request (the
per-chunk streaming hook, the pre-body refusal) put the key alias there instead, so tests correlate by a unique
key alias.

Request-path V2 gates act only on directives written in the request text:

- `OrderGate` (instances `gate_y`, `gate_x`): pre_call, reads and writes `messages`, `may_abort=True`. With
  `cblv2b-order` it appends ` [<name>]` to the last message; with `cblv2b-abort-at-<name>` it returns Abort 452;
  `cblv2b-abort-code-at-<name>` adds `code` and `body` to that Abort; `cblv2b-respond-at-<name>` returns Respond;
  `cblv2b-reroute-at-<name>` returns Reroute to `cblv2-chat-rejected`.
- `LegacyGate` (instance `legacy_gate`): a plain CustomLogger pre-call hook that tags the same way and rejects
  with HTTPException 400 on `cblv2b-reject-at-legacy_gate`.
- `PostGate` aborts 453 at post_call_success, `DuringGate` aborts 454 at during_call, `FailureGate` (instances
  `failure_gate_a` 503 and `failure_gate_b` 504) aborts or raises at post_call_failure, and `HeadersRaiser` raises
  at response_headers.

Off-path V2 readers (`attempt_ended`) differ only in their Selection, for the selection tests. `FailureLedger` and
`ChunkLedger` are legacy loggers that record failure hooks and per-chunk streaming hooks.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Mapping
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
from litellm.integrations.custom_logger_v2.outcomes import Reroute, Respond

PROBE_MODULE: Final = "integration.observability.dec1_cblv2b_probe"

ORDER: Final = "cblv2b-order"
ABORT_AT: Final = "cblv2b-abort-at-"
ABORT_CODE_AT: Final = "cblv2b-abort-code-at-"
RESPOND_AT: Final = "cblv2b-respond-at-"
REROUTE_AT: Final = "cblv2b-reroute-at-"
REJECT_AT: Final = "cblv2b-reject-at-"
POST_ABORT: Final = "cblv2b-post-abort"
DURING_ABORT: Final = "cblv2b-during-abort"
FAILURE_ABORT_AT: Final = "cblv2b-failure-abort-at-"
FAILURE_RAISE: Final = "cblv2b-failure-raise"
HEADERS_RAISE: Final = "cblv2b-headers-raise"

ABORT_STATUS: Final = 452
ABORT_HEADER: Final = "x-cblv2b-aborted-by"
ABORT_CODE: Final = "cblv2b_policy"
ABORT_BODY: Final = {"reason": "cblv2b"}
POST_ABORT_STATUS: Final = 453
DURING_ABORT_STATUS: Final = 454
FAILURE_STATUSES: Final = {"failure_gate_a": 503, "failure_gate_b": 504}
REJECT_STATUS: Final = 400
RESPOND_TEXT: Final = "answered by the cblv2b gate without a provider"
REROUTE_TARGET: Final = "cblv2-chat-rejected"

_SEQUENCE: Final = itertools.count()


def abort_message(name: str) -> str:
    return f"aborted by the cblv2b gate {name}"


def failure_message(name: str) -> str:
    return f"replaced by the cblv2b failure gate {name}"


def _post(record: Mapping[str, object]) -> None:
    body: Final = json.dumps({**record, "seq": next(_SEQUENCE), "pid": os.getpid()}, default=str).encode()
    with httpx.Client(trust_env=False) as client:
        client.post(os.environ["CBLV2_PROBE_URL"], content=body).raise_for_status()


async def _send(record: Mapping[str, object]) -> None:
    body: Final = json.dumps({**record, "seq": next(_SEQUENCE), "pid": os.getpid()}, default=str).encode()
    async with httpx.AsyncClient(trust_env=False) as client:
        (await client.post(os.environ["CBLV2_PROBE_URL"], content=body)).raise_for_status()


def _view_text(view: object) -> str:
    return json.dumps({"messages": getattr(view, "messages", None)}, default=str, ensure_ascii=False)


def _data_text(data: object) -> str:
    fields: Final = data if isinstance(data, Mapping) else {}
    return json.dumps({"messages": fields.get("messages")}, default=str, ensure_ascii=False)


def _tagged(messages: object, name: str) -> list[object]:
    listed: Final = list(messages) if isinstance(messages, list) else []
    if not listed or not isinstance(listed[-1], Mapping):
        return listed
    last: Final = dict(listed[-1])
    last["content"] = f"{last.get('content')} [{name}]"
    return [*listed[:-1], last]


def _status_of(error: object) -> object:
    return getattr(error, "status_code", None) or getattr(error, "code", None)


class OrderGate(CustomLoggerV2):
    """Abortable pre_call writer; two instances show the serial config-order contract."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.selection = Selection(name=f"cblv2b_{name}")

    @on(Hook.pre_call, reads={Field.messages}, writes={Field.messages}, may_abort=True)
    async def gate(self, view: object, ctx: object) -> object:
        text: Final = _view_text(view)
        await _send({"logger": self.name, "hook": "pre_call", "text": text})
        if f"{ABORT_CODE_AT}{self.name}" in text:
            return Abort(status=ABORT_STATUS, message=abort_message(self.name), code=ABORT_CODE, body=ABORT_BODY)
        if f"{ABORT_AT}{self.name}" in text:
            return Abort(status=ABORT_STATUS, message=abort_message(self.name), headers={ABORT_HEADER: self.name})
        if f"{RESPOND_AT}{self.name}" in text:
            return Respond(response={"choices": [{"message": {"role": "assistant", "content": RESPOND_TEXT}}]})
        if f"{REROUTE_AT}{self.name}" in text:
            return Reroute(model=REROUTE_TARGET)
        if ORDER in text:
            return Patch().set(Field.messages, _tagged(getattr(view, "messages", None), self.name))
        return None


class LegacyGate(CustomLogger):
    """Legacy pre-call hook in the same list; the dispatcher treats it as a writer that may abort."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    async def async_pre_call_hook(
        self, user_api_key_dict: object, cache: object, data: dict[str, object], call_type: str
    ) -> dict[str, object] | None:
        text: Final = _data_text(data)
        await _send({"logger": self.name, "hook": "async_pre_call_hook", "text": text})
        if f"{REJECT_AT}{self.name}" in text:
            raise HTTPException(status_code=REJECT_STATUS, detail={"error": f"rejected by {self.name}"})
        if ORDER in text:
            return {**data, "messages": _tagged(data.get("messages"), self.name)}
        return None


class PostGate(CustomLoggerV2):
    selection = Selection(name="cblv2b_post_gate")

    @on(Hook.post_call_success, reads={Field.messages}, may_abort=True)
    async def check(self, view: object, ctx: object) -> object:
        text: Final = _view_text(view)
        await _send({"logger": "post_gate", "hook": "post_call_success", "text": text})
        if POST_ABORT in text:
            return Abort(status=POST_ABORT_STATUS, message=abort_message("post_gate"))
        return None


class DuringGate(CustomLoggerV2):
    selection = Selection(name="cblv2b_during_gate")

    @on(Hook.during_call, reads={Field.messages}, may_abort=True)
    async def check(self, view: object, ctx: object) -> object:
        text: Final = _view_text(view)
        await _send({"logger": "during_gate", "hook": "during_call", "text": text})
        if DURING_ABORT in text:
            return Abort(status=DURING_ABORT_STATUS, message=abort_message("during_gate"))
        return None


class FailureGate(CustomLoggerV2):
    """post_call_failure handler that may replace the error, or raise to show the hook swallows it."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.selection = Selection(name=f"cblv2b_{name}")

    @on(Hook.post_call_failure, reads={Field.messages, Field.error}, may_abort=True)
    async def replace(self, view: object, ctx: object) -> object:
        text: Final = _view_text(view)
        error: Final = getattr(view, "error", None)
        await _send(
            {
                "logger": self.name,
                "hook": "post_call_failure",
                "text": text,
                "error": type(error).__name__,
                "status": _status_of(error),
            }
        )
        if f"{FAILURE_ABORT_AT}{self.name}" in text:
            return Abort(status=FAILURE_STATUSES[self.name], message=failure_message(self.name))
        if FAILURE_RAISE in text:
            raise RuntimeError(f"the cblv2b failure gate {self.name} raised")
        return None


class HeadersRaiser(CustomLoggerV2):
    selection = Selection(name="cblv2b_headers_raiser")

    @on(Hook.response_headers, reads={Field.messages}, writes={Field.response_headers})
    async def headers(self, view: object, ctx: object) -> object:
        if HEADERS_RAISE in _view_text(view):
            raise RuntimeError("the cblv2b headers handler raised")
        return Patch().merge(Field.response_headers, {"x-cblv2b-headers": "seen"})


class Reader(CustomLoggerV2):
    """Off-path attempt_ended reader; subclasses differ only in their Selection."""

    def __init__(self, name: str) -> None:
        self.name = name

    @on(Hook.attempt_ended, reads={Field.messages, Field.outcome})
    def record(self, view: object, ctx: object) -> None:
        outcome: Final = getattr(view, "outcome", None)
        _post(
            {
                "logger": self.name,
                "hook": "attempt_ended",
                "text": _view_text(view),
                "outcome": getattr(outcome, "value", outcome),
            }
        )


class RequiredReader(Reader):
    selection = Selection(
        name="cblv2b_required_reader",
        activation=Activation.required,
        opt_out_by=frozenset({OptOutBy.admin, OptOutBy.caller}),
        honors_no_log=True,
    )


class OptInReader(Reader):
    selection = Selection(name="cblv2b_opt_in_reader", activation=Activation.opt_in)


class AdminOnlyReader(Reader):
    selection = Selection(name="cblv2b_admin_only_reader")


class CallerReader(Reader):
    selection = Selection(name="cblv2b_caller_reader", opt_out_by=frozenset({OptOutBy.admin, OptOutBy.caller}))


class AliasReader(Reader):
    selection = Selection(
        name="cblv2b_alias_reader",
        aliases=frozenset({"cblv2b_alias"}),
        opt_out_by=frozenset({OptOutBy.admin, OptOutBy.caller}),
    )


class CamelReader(Reader):
    selection = Selection(name="Cblv2bCamelReader", opt_out_by=frozenset({OptOutBy.admin, OptOutBy.caller}))


class NoLogReader(Reader):
    selection = Selection(name="cblv2b_no_log_reader", honors_no_log=True)


class FailureLedger(CustomLogger):
    """Legacy logger recording every failure hook with the key alias, so pre-body refusals can be found too."""

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: object,
        traceback_str: str | None = None,
    ) -> None:
        await _send(
            {
                "logger": "failure_ledger",
                "hook": "async_post_call_failure_hook",
                "text": _data_text(request_data),
                "key_alias": getattr(user_api_key_dict, "key_alias", None),
                "error": type(original_exception).__name__,
                "status": _status_of(original_exception),
            }
        )

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        exception: Final = kwargs.get("exception")
        await _send(
            {
                "logger": "failure_ledger",
                "hook": "async_log_failure_event",
                "text": _data_text(kwargs),
                "error": type(exception).__name__,
                "status": _status_of(exception),
            }
        )


class ChunkLedger(CustomLogger):
    """Legacy per-chunk streaming hook; the proxy hands it the text so far, not the request."""

    async def async_post_call_streaming_hook(self, user_api_key_dict: object, response: str) -> None:
        await _send(
            {
                "logger": "chunk_ledger",
                "hook": "async_post_call_streaming_hook",
                "text": str(getattr(user_api_key_dict, "key_alias", None)),
                "so_far": response,
            }
        )


gate_x: Final = OrderGate("gate_x")
gate_y: Final = OrderGate("gate_y")
legacy_gate: Final = LegacyGate("legacy_gate")
post_gate: Final = PostGate()
during_gate: Final = DuringGate()
failure_gate_a: Final = FailureGate("failure_gate_a")
failure_gate_b: Final = FailureGate("failure_gate_b")
headers_raiser: Final = HeadersRaiser()
required_reader: Final = RequiredReader("required_reader")
opt_in_reader: Final = OptInReader("opt_in_reader")
admin_only_reader: Final = AdminOnlyReader("admin_only_reader")
caller_reader: Final = CallerReader("caller_reader")
alias_reader: Final = AliasReader("alias_reader")
camel_reader: Final = CamelReader("camel_reader")
no_log_reader: Final = NoLogReader("no_log_reader")
failure_ledger: Final = FailureLedger()
chunk_ledger: Final = ChunkLedger()
