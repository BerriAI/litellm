"""Harness coverage for the model barrier that gates on every replica.

No proxy needed and no ``e2e`` marker: this pins that a model registered through
the control plane only counts as servable once every configured replica lists it
on /v1/models, which is what keeps a two-gateway stack from handing a test a
model that one gateway has not reloaded yet, and that a read-back after a write
converges only once every replica serves the written state. The fakes are plain
pollers and an injected clock, so nothing here monkeypatches anything.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import chain, repeat
from typing import Final

import pytest

from e2e_config import parse_replica_urls
from e2e_http import Success
from models import ModelInfoEntry, ModelInfoResponse, ModelListEntry, ModelsListResponse
from proxy_client import (
    BodyReader,
    Converged,
    ModelsPoller,
    NeverConvergedOn,
    NotServableOn,
    Servable,
    await_converged_everywhere,
    await_servable_everywhere,
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


def _info(*model_names: str) -> Success[ModelInfoResponse]:
    entries: Final = [ModelInfoEntry(model_name=model_name) for model_name in model_names]
    return Success(status_code=200, data=ModelInfoResponse(data=entries))


def _reader(results: Iterable[Success[ModelInfoResponse]]) -> BodyReader[ModelInfoResponse]:
    it: Final = iter(results)
    return lambda _timeout: next(it)


def _lists_model(body: ModelInfoResponse) -> bool:
    return any(entry.model_name == MODEL for entry in body.data)


def _read_back(
    readers: Mapping[str, BodyReader[ModelInfoResponse]],
) -> tuple[Converged[ModelInfoResponse] | NeverConvergedOn[ModelInfoResponse], FakeClock]:
    clock: Final = FakeClock()
    outcome: Final = await_converged_everywhere(
        readers,
        predicate=_lists_model,
        timeout=TIMEOUT,
        interval=INTERVAL,
        request_timeout=5.0,
        now=clock.now,
        sleep=clock.sleep,
    )
    return outcome, clock


class TestAwaitConvergedEverywhere:
    def test_waits_for_the_lagging_replica_and_returns_every_body(self) -> None:
        readers: Final = {
            "gateway-1": _reader(repeat(_info(MODEL))),
            "gateway-2": _reader(chain(repeat(_info(), 2), repeat(_info(MODEL)))),
        }
        outcome, clock = _read_back(readers)
        assert outcome == Converged(bodies={"gateway-1": _info(MODEL).data, "gateway-2": _info(MODEL).data})
        assert clock.elapsed == 2 * INTERVAL

    @pytest.mark.parametrize("lagging", ["gateway-1", "gateway-2"])
    def test_fails_naming_the_replica_that_never_converges(self, lagging: str) -> None:
        readers: Final = {
            "gateway-1": _reader(repeat(_info(MODEL))),
            "gateway-2": _reader(repeat(_info(MODEL))),
        } | {lagging: _reader(repeat(_info()))}
        outcome, clock = _read_back(readers)
        assert outcome == NeverConvergedOn(replica=lagging, last_result=_info())
        assert clock.elapsed >= TIMEOUT
