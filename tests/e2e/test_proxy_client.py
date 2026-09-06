"""Harness coverage for the barriers that gate on every replica.

No proxy needed and no ``e2e`` marker: this pins that a model registered through
the control plane only counts as servable once every configured replica lists it
on /v1/models, and that a management write only counts as read back once every
replica that serves the route reflects it, which is what keeps a multi-replica
stack from handing a test a replica the write has not reached yet. The fakes are
plain pollers and an injected clock, so nothing here monkeypatches anything.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import chain, repeat
from typing import Final

import pytest

from e2e_config import parse_replica_urls
from e2e_http import Success
from models import ModelListEntry, ModelsListResponse
from proxy_client import (
    Converged,
    ModelsPoller,
    NeverConvergedOn,
    NotServableOn,
    ReplicaRead,
    Servable,
    await_everywhere,
    await_servable_everywhere,
    build_proxy_client,
)

MODEL: Final = "gpt-under-test"
TIMEOUT: Final = 10.0
INTERVAL: Final = 2.0


@dataclass
class FakeClock:
    elapsed: float = 0.0

    def now(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


def _listing(*model_ids: str) -> Success[ModelsListResponse]:
    entries: Final = tuple(ModelListEntry(id=model_id) for model_id in model_ids)
    return Success(status_code=200, data=ModelsListResponse(data=entries))


def _poller(results: Iterable[Success[ModelsListResponse]]) -> ModelsPoller:
    it: Final = iter(results)
    return lambda _timeout: next(it)


def _await(pollers: Mapping[str, ModelsPoller]) -> Servable | NotServableOn:
    clock: Final = FakeClock()
    return await_servable_everywhere(
        pollers,
        model_name=MODEL,
        timeout=TIMEOUT,
        interval=INTERVAL,
        request_timeout=5.0,
        db_sync_seconds=0.0,
        now=clock.now,
        sleep=clock.sleep,
    )


class TestAwaitServableEverywhere:
    @pytest.mark.parametrize("missing", ["gateway-1", "gateway-2"])
    def test_fails_on_the_replica_that_never_lists_the_model(self, missing: str) -> None:
        pollers: Final = {
            "gateway-1": _poller(repeat(_listing(MODEL))),
            "gateway-2": _poller(repeat(_listing(MODEL))),
        } | {missing: _poller(repeat(_listing()))}
        assert _await(pollers) == NotServableOn(replica=missing, last_result=_listing())

    def test_passes_once_every_replica_lists_the_model(self) -> None:
        pollers: Final = {
            "gateway-1": _poller(repeat(_listing(MODEL))),
            "gateway-2": _poller(chain(repeat(_listing(), 2), repeat(_listing(MODEL)))),
        }
        assert _await(pollers) == Servable()


class TestParseReplicaUrls:
    def test_splits_and_trims_the_gateway_addresses(self) -> None:
        raw: Final = " http://127.0.0.1:4010/, http://127.0.0.1:4011 "
        assert parse_replica_urls(raw, "http://lb") == ("http://127.0.0.1:4010", "http://127.0.0.1:4011")

    def test_falls_back_to_the_data_plane_address_when_unset(self) -> None:
        assert parse_replica_urls("", "http://lb") == ("http://lb",)


def _answers(answers: Iterable[str]) -> ReplicaRead[str]:
    it: Final = iter(answers)
    return lambda _timeout: next(it)


def _await_everywhere(reads: Mapping[str, ReplicaRead[str]]) -> Converged[str] | NeverConvergedOn[str]:
    clock: Final = FakeClock()
    return await_everywhere(
        reads,
        settled=lambda answer: answer == "renamed",
        timeout=TIMEOUT,
        interval=INTERVAL,
        request_timeout=5.0,
        now=clock.now,
        sleep=clock.sleep,
    )


class TestAwaitEverywhere:
    def test_waits_for_the_lagging_replica_and_returns_every_settled_answer(self) -> None:
        reads: Final = {
            "gateway-1": _answers(repeat("renamed")),
            "gateway-2": _answers(chain(repeat("stale", 2), repeat("renamed"))),
        }
        outcome: Final = _await_everywhere(reads)
        assert isinstance(outcome, Converged)
        assert dict(outcome.answers) == {"gateway-1": "renamed", "gateway-2": "renamed"}

    def test_names_the_replica_that_never_converges_with_what_it_last_served(self) -> None:
        reads: Final = {
            "gateway-1": _answers(repeat("renamed")),
            "gateway-2": _answers(repeat("stale")),
        }
        assert _await_everywhere(reads) == NeverConvergedOn(replica="gateway-2", last="stale")

    def test_polls_until_the_deadline_before_giving_up(self) -> None:
        lagging: Final = chain(repeat("stale", int(TIMEOUT / INTERVAL)), repeat("renamed"))
        outcome: Final = _await_everywhere({"gateway-1": _answers(lagging)})
        assert isinstance(outcome, Converged), outcome


class TestReplicasFor:
    def test_split_deployment_reads_management_routes_back_from_the_control_plane(self) -> None:
        client: Final = build_proxy_client(
            base_url="http://lb",
            control_plane_base_url="http://backend",
            replica_urls=("http://gateway-1", "http://gateway-2"),
        )
        assert set(client.replicas_for("/v1/mcp/server/abc")) == {"http://backend"}
        assert set(client.replicas_for("/v1/models")) == {"http://gateway-1", "http://gateway-2"}

    def test_monolith_reads_management_routes_back_from_every_replica(self) -> None:
        client: Final = build_proxy_client(
            base_url="http://lb",
            control_plane_base_url="http://lb",
            replica_urls=("http://pod-1", "http://pod-2"),
        )
        assert set(client.replicas_for("/v1/mcp/server/abc")) == {"http://pod-1", "http://pod-2"}
