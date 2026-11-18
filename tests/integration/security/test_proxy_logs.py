"""S6: the owned proxy's own stdout and stderr never carry a credential canary.

Each leg boots its own proxy (slot B1 lives in its config), sends one successful and one
provider-rejected chat completion, stops the proxy so every buffered write reaches the log
file, and then searches the whole captured log. The ``default`` leg runs with ``LITELLM_LOG``
unset, the level an operator gets out of the box; the ``debug`` leg runs with
``LITELLM_LOG=DEBUG``, which prints request data, router decisions and provider calls.

Positive control: the provider double must receive ``Authorization: Bearer <B1 canary>`` for
both requests. Sensitivity control: the provider double echoes the rejected message in its
error text, and the proxy logs that error at every level, so the marker must be found in the
log; a capture that misses the log file or reads it before the writes land fails there.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from integration._support.client import string_value
from integration._support.wire import Reply, Request
from integration.security._canary import MARKER, Canary, canary, find_canary
from integration.security._sinks import CONFIG_MODEL, PROVIDER_4XX, canary_rig, chat_upstream, settle, team_caller
from integration.security._sweeps import Hit, assert_no_hits

LEGS: Final = MappingProxyType({"default": MappingProxyType({}), "debug": MappingProxyType({"LITELLM_LOG": "DEBUG"})})


def echoing_upstream(request: Request) -> Reply:
    """``chat_upstream``, except a rejection repeats the rejected message in its error text."""
    body: Final = json.loads(request.body or b"{}")
    text: Final = str((body.get("messages") or [{}])[-1].get("content", ""))
    if PROVIDER_4XX not in text:
        return chat_upstream(request)
    return Reply(
        status=400,
        body=json.dumps(
            {"error": {"type": "invalid_request_error", "code": "canary_rejected", "message": f"rejected: {text}"}}
        ).encode(),
    )


def sweep_log(path: Path, canaries: tuple[Canary, ...]) -> tuple[Hit, ...]:
    """Every canary in the captured log, attributed to the line that holds it."""
    data: Final = path.read_bytes()
    if not find_canary(data, canaries):
        return ()
    return tuple(
        Hit("S6", f"{path.name} line {number}: {line[:160]!r}", match.slot, match.encoding)
        for number, line in enumerate(data.splitlines(), start=1)
        for match in find_canary(line, canaries)
    )


@pytest.mark.parametrize("leg", tuple(LEGS))
def test_proxy_log_carries_no_credential(leg: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_LOG", raising=False)
    marker: Final = canary(MARKER)
    with canary_rig(tmp_path, environment=LEGS[leg], upstream=echoing_upstream) as rig:
        b1: Final = rig.canaries["B1"]
        with rig.proxy.scenario() as scenario:
            caller: Final = team_caller(scenario)
            responses: Final = tuple(
                rig.proxy.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": CONFIG_MODEL, "messages": [{"role": "user", "content": f"slot B1 {suffix}"}]},
                    key=caller.key,
                )
                for suffix in (marker.value, f"{marker.value} {PROVIDER_4XX}")
            )
            assert [response.status_code for response in responses] == [200, 400], [r.text for r in responses]
            delivered: Final = rig.provider.carrying(marker.value)
            assert [request.headers.get("authorization") for request in delivered] == [f"Bearer {b1.value}"] * 2, (
                "Positive control: the provider double never received the B1 canary"
            )
            settle(rig, string_value(responses[0].json()["id"]), marker)
        log: Final = rig.owned.log
    hits: Final = sweep_log(log, (marker, b1))
    assert any(hit.slot == MARKER for hit in hits), f"Sensitivity control: the marker never reached {log}"
    assert_no_hits(tuple(hit for hit in hits if hit.slot != MARKER), f"slot B1, proxy log, {leg} level")
