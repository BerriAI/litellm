import asyncio
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel, TypeAdapter

from litellm.proxy.control_plane.common.fanout import Failure, Fanout, Missed, parse_reply
from litellm.proxy.control_plane.dataplanes.client import DataplaneReply, DataplaneUnreachable
from litellm.proxy.control_plane.dataplanes.credentials import CredentialUnavailable, IssuedCredential
from litellm.proxy.control_plane.dataplanes.registry import Dataplane


class _NoCredential:
    async def headers(self, caller) -> IssuedCredential:
        raise AssertionError("fanout never issues credentials itself")


def _dataplane(dataplane_id: str) -> Dataplane:
    return Dataplane(dataplane_id, dataplane_id, f"https://{dataplane_id}.test", _NoCredential(), ())


@dataclass
class _InFlight:  # mutable-ok: test probe counting concurrent fetches
    now: int = 0
    peak: int = 0
    seen: list[str] = field(default_factory=list)


async def test_gather_bounds_concurrency_and_keeps_dataplane_order() -> None:
    probe = _InFlight()

    async def fetch(dataplane: Dataplane) -> str | Failure:
        probe.now += 1
        probe.peak = max(probe.peak, probe.now)
        await asyncio.sleep(0)
        probe.now -= 1
        probe.seen.append(dataplane.id)
        return dataplane.id.upper()

    dataplanes = tuple(_dataplane(name) for name in ("a", "b", "c", "d", "e"))
    gathered = await Fanout(concurrency=2).gather(dataplanes, fetch)
    assert probe.peak == 2, f"semaphore of 2 let {probe.peak} fetches run at once"
    assert [reached.value for reached in gathered.reached] == ["A", "B", "C", "D", "E"]
    assert gathered.missed == () and not gathered.partial


async def test_a_failing_dataplane_is_a_missed_value_and_marks_the_result_partial() -> None:
    async def fetch(dataplane: Dataplane) -> int | Failure:
        return Failure("boom") if dataplane.id == "b" else 1

    gathered = await Fanout(concurrency=4).gather(tuple(_dataplane(name) for name in "abc"), fetch)
    assert [reached.dataplane.id for reached in gathered.reached] == ["a", "c"]
    assert gathered.missed == (Missed(gathered.missed[0].dataplane, "boom"),)
    assert gathered.missed[0].dataplane.id == "b"
    assert gathered.partial and not gathered.nothing_answered


async def test_nothing_answered_only_when_every_dataplane_missed() -> None:
    async def fetch(dataplane: Dataplane) -> int | Failure:
        return Failure("down")

    all_down = await Fanout(concurrency=1).gather((_dataplane("a"),), fetch)
    none_asked = await Fanout(concurrency=1).gather((), fetch)
    assert all_down.nothing_answered and not none_asked.nothing_answered


class _Shape(BaseModel):
    n: int


_SHAPE = TypeAdapter(_Shape)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (DataplaneReply(200, {"n": 3}, "service_token"), _Shape(n=3)),
        (DataplaneReply(200, {"n": "x"}, "service_token"), Failure("unexpected response shape: 1 validation error(s)")),
        (DataplaneReply(403, {"error": "no"}, "oidc_exchange"), Failure("HTTP 403 with the oidc_exchange credential")),
        (DataplaneUnreachable("ConnectError", 3), Failure("unreachable after 3 attempt(s): ConnectError")),
        (CredentialUnavailable("service_token", "no token"), Failure("no usable credential: no token")),
    ],
)
def test_parse_reply_turns_every_bad_outcome_into_a_failure(result, expected) -> None:
    assert parse_reply(result, _SHAPE) == expected
